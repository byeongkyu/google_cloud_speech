#!/usr/bin/python
#-*- encoding: utf8 -*-

import contextlib
import sys
import threading
import copy
import numpy as np
import yaml
import os
from array import array
from sys import byteorder
from struct import pack
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fftpack import rfft, irfft, fftfreq
import signal

import pyaudio
import audioop
import wave
from google.cloud import speech

import rospy
from std_msgs.msg import Bool, String, Float64, Empty
from dynamic_reconfigure.server import Server
from google_cloud_speech.msg import RecognizedWord
from google_cloud_speech.cfg import RecognitionConfig


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextlib.contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


AUDIO_FILE = "record.wav"


class GoogleCloudSpeech:
    def __init__(self):
        self.client = speech.SpeechClient()
        self.pub_recognized_word = rospy.Publisher('recognized_word', RecognizedWord, queue_size=10)
        self.language_code = 'en_US'
        self.vocabulary_file = rospy.get_param('~vocabulary_file', "")
        self.vocabulary = []
        if self.vocabulary_file != '':
            target_file = os.path.abspath(self.vocabulary_file)
            target_file = os.path.expanduser(self.vocabulary_file)
            print target_file
            with open(target_file) as f:
                self.vocabulary = yaml.load(f)
                rospy.loginfo('load user vocabulary...')

    def set_language_code(self, lang):
        self.language_code = lang

    def recognize(self):
        with open(AUDIO_FILE, 'rb') as stream:
            sample = self.client.sample(
                stream=stream, encoding=speech.Encoding.LINEAR16,
                sample_rate_hertz=16000)

            rospy.loginfo('Request to cloud.google.com...')
            results = sample.streaming_recognize(
                language_code=self.language_code,
                interim_results=True,
                single_utterance=False,
                speech_contexts=self.vocabulary)

            try:
                rospy.loginfo('Response from cloud.google.com...')
                for result in results:
                    for alternative in result.alternatives:
                        if result.is_final:
                            msg = RecognizedWord()
                            msg.recognized_word = alternative.transcript
                            msg.confidence = alternative.confidence
                            rospy.loginfo(
                                '%s recognized...(%s), confidence (%s) ...' %
                                (rospy.get_name(),
                                    alternative.transcript.encode('utf-8'),
                                    alternative.confidence))
                            self.pub_recognized_word.publish(msg)
            except:
                pass


class RecordDetectAudio:
    CHUNK_SIZE = 1024
    RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    SILENT_CHUNK_SIZE = (RATE/CHUNK_SIZE)*1.5
    FRAME_MAX_VALUE = 2 ** 15 - 1
    NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)

    def __init__(self):
        with noalsaerr():
            self.p = pyaudio.PyAudio()

        self.fig = plt.figure()
        self.fig.canvas.toolbar.pack_forget()
        self.ax = self.fig.add_subplot(111)

        x = [1]
        y = [1]

        self.li, = self.ax.plot(x, y)
        self.fig.canvas.draw()
        plt.yticks([])
        self.fig.tight_layout()
        plt.show(block=False)

        signal.signal(signal.SIGUSR2, self.update_fig)
        signal.siginterrupt(signal.SIGUSR2, False)

        self.detect_count = 0
        self.enable_recognition = True
        self.threshold = 900
        rospy.Subscriber('enable_recognition', Bool, self.handle_enable_recognition)

        self.pub_start_speech = rospy.Publisher('start_of_speech', Empty, queue_size=10)
        self.pub_end_speech = rospy.Publisher('end_of_speech', Empty, queue_size=10)
        self.pub_silency_detected = rospy.Publisher('silency_detected', Empty, queue_size=10)

    def handle_enable_recognition(self, msg):
        self.enable_recognition = msg.data

    def set_threshold(self, value):
        self.threshold = value
        rospy.loginfo('%s threadhold is %d' % (rospy.get_name(), self.threshold))

    def update_fig(self, signum, frame):
        self.fig.canvas.draw()

    def record(self):
        stream = self.p.open(format=RecordDetectAudio.FORMAT, channels=RecordDetectAudio.CHANNELS,
            rate=RecordDetectAudio.RATE, input=True, output=False, frames_per_buffer=RecordDetectAudio.CHUNK_SIZE)

        rms = []
        for i in range(10):
            data = stream.read(RecordDetectAudio.CHUNK_SIZE)
            rms.append(audioop.rms(data, 2))
        self.threshold2 = np.mean(rms) * 80
        if self.threshold2 > 2500:
            self.threshold2 = 2500
        print self.threshold2

        data_all = array('h')
        audio_started = False
        silent_chunks = 0
        start_index = -1

        while not rospy.is_shutdown() and self.enable_recognition:
            data_chunk = array('h', stream.read(RecordDetectAudio.CHUNK_SIZE))
            if byteorder == 'big':
                data_chunk.byteswap()

            data, is_silent = self.fft(data_chunk)
            try:
                data_all.extend(data)
            except OverflowError as e:
                continue

            if audio_started:
                if is_silent:
                    silent_chunks += 1
                    if silent_chunks > RecordDetectAudio.SILENT_CHUNK_SIZE:
                        rospy.loginfo('\033[95m[%s] \033[92mRecording stoped...\033[0m' % rospy.get_name())
                        self.pub_end_speech.publish()
                        break

            elif not is_silent and self.enable_recognition:
                audio_started = True
                start_index = len(data_all)
                silent_chunks = 0
                rospy.loginfo('\033[95m[%s] \033[92mRecording started...\033[0m' % rospy.get_name())
                self.pub_start_speech.publish()

            else:
                silent_chunks += 1
                if silent_chunks > RecordDetectAudio.SILENT_CHUNK_SIZE:
                    self.pub_silency_detected.publish()
                    silent_chunks = 0

        sample_width = self.p.get_sample_size(RecordDetectAudio.FORMAT)
        stream.stop_stream()
        stream.close()

        if self.enable_recognition:
            data_all = self.trim(data_all, start_index)
            data_all = self.normalize(data_all)

        return self.enable_recognition, sample_width, data_all

    def trim(self, data, start_index):
        _from = max(start_index - RecordDetectAudio.RATE, 0)
        return copy.deepcopy(data[_from:-1])

    def normalize(self, data):
        r = array('h')
        if len(data) == 0:
            return r

        normalize_factor = (float(RecordDetectAudio.NORMALIZE_MINUS_ONE_dB * RecordDetectAudio.FRAME_MAX_VALUE)
            / max(abs(i) for i in data))
        for i in data:
            r.append(int(i * normalize_factor))
        return r

    def fft(self, data):
        # FFT
        W = fftfreq(len(data), 1.0/RecordDetectAudio.RATE*2.0)
        f_signal = rfft(data)

        # Doing some filter?
        f_signal[(W<16)] = 0

        # Visualization
        self.li.set_xdata(W[1:len(W)/2])
        self.li.set_ydata(np.abs(f_signal[1:len(W)/2]))
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        os.kill(os.getpid(), signal.SIGUSR2)

        m_signal = f_signal.copy()
        # print np.mean(np.abs(m_signal[16:96]))
        if np.mean(np.abs(m_signal[16:96])) > self.threshold2:
            self.detect_count += 1
        else:
            self.detect_count = 0

        if self.detect_count > 0:
            self.detect_count = 0
            return irfft(m_signal).astype(np.int32), False
        else:
            return irfft(m_signal).astype(np.int32), True


class GoogleCloudSpeechNode:
    def __init__(self):
        self.client = GoogleCloudSpeech()
        self.recoder = RecordDetectAudio()
        self.conf_srv = Server(RecognitionConfig, self.callback_config)

        self.t1 = threading.Thread(target=self.handle_speech_recognition)
        self.t1.start()

        rospy.loginfo('%s initialized...' % rospy.get_name())
        rospy.spin()

    def callback_config(self, config, level):
        self.recoder.set_threshold(config['audio_threshold'])
        self.client.set_language_code(config['language'])
        return config

    def handle_speech_recognition(self):
        while not rospy.is_shutdown():
            result, sample_width, data = self.recoder.record()
            if not result:
                rospy.sleep(0.1)
                continue

            if not rospy.is_shutdown():
                data = pack('<' + ('h' * len(data)), *data)
                wave_file = wave.open(AUDIO_FILE, 'wb')
                wave_file.setnchannels(RecordDetectAudio.CHANNELS)
                wave_file.setsampwidth(sample_width)
                wave_file.setframerate(RecordDetectAudio.RATE)
                wave_file.writeframes(data)
                wave_file.close()

                self.client.recognize()
            rospy.sleep(0.5)


if __name__ == '__main__':
    rospy.init_node('google_speech_node', anonymous=False)
    try:
        m = GoogleCloudSpeechNode()
    except rospy.ROSInterruptException: pass
