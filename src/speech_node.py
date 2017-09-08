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

import pyaudio
import wave
from google.cloud import speech

import rospy
from std_msgs.msg import Bool, String, Float64, Empty
from google_cloud_speech.msg import RecognizedWord


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

AUDIO_FILE="record.wav"


class GoogleCloudSpeech:
    def __init__(self):
        self.client = speech.Client()
        self.pub_recognized_word = rospy.Publisher('recognized_word', RecognizedWord, queue_size=10)
        self.vocabulary_file = rospy.get_param('~vocabulary_file', "")
        self.vocabulary = []
        if self.vocabulary_file != '':
            target_file = os.path.abspath(self.vocabulary_file)
            target_file = os.path.expanduser(self.vocabulary_file)
            with open(target_file) as f:
                self.vocabulary = yaml.load(f)
                rospy.loginfo('load user vocabulary...')

    def recognize(self):
        with open(AUDIO_FILE, 'rb') as stream:
            sample = self.client.sample(stream=stream, encoding=speech.Encoding.LINEAR16, sample_rate_hertz=16000)
            results = sample.streaming_recognize(
                    language_code='en-US',
                    interim_results=True,
                    single_utterance=False,
                    speech_contexts=self.vocabulary)

            try:
                for result in results:
                    for alternative in result.alternatives:
                        if result.is_final:
                            msg = RecognizedWord()
                            msg.recognized_word = alternative.transcript
                            msg.confidence = alternative.confidence
                            rospy.loginfo('%s recognized...(%s), confidence (%s) ...'%
                                (rospy.get_name(), alternative.transcript.encode('utf-8'), alternative.confidence))
                            self.pub_recognized_word.publish(msg)
            except:
                pass

class RecordWav:
    CHUNK_SIZE = 1024
    RATE = 16000
    CHANNELS = 1
    THRESHOLD = 1600  # audio levels not normalised.
    BUFFERSIZE = 2 ** 12

    SILENT_CHUNKS = 2 * RATE / CHUNK_SIZE  # about 1sec
    FORMAT = pyaudio.paInt16
    FRAME_MAX_VALUE = 2 ** 15 - 1
    NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
    TRIM_APPEND = RATE / 4

    def __init__(self):
        with noalsaerr():
            self.p = pyaudio.PyAudio()
        self.audio_data = np.empty((self.CHUNK_SIZE * self.BUFFERSIZE), dtype=np.int16)

        self.enable_recognition = True

        rospy.Subscriber('enable_recognition', Bool, self.handle_enable_recognition)
        self.pub_start_speech = rospy.Publisher('start_of_speech', Empty, queue_size=10)
        self.pub_end_speech = rospy.Publisher('end_of_speech', Empty, queue_size=10)
        self.pub_silency_detected = rospy.Publisher('silency_detected', Empty, queue_size=10)

    def handle_enable_recognition(self, msg):
        if msg.data:
            rospy.logdebug('enable_recognition...')
            self.enable_recognition = True
        else:
            rospy.logdebug('disable_recognition...')
            self.enable_recognition = False

    def is_silent(self, data_chunk):
        """Returns 'True' if below the 'silent' threshold"""
        return max(data_chunk) < RecordWav.THRESHOLD

    def normalize(self, data_all):
        """Amplify the volume out to max -1dB"""
        # MAXIMUM = 16384
        r = array('h')
        if len(data_all) == 0:
            return r

        normalize_factor = (float(RecordWav.NORMALIZE_MINUS_ONE_dB * RecordWav.FRAME_MAX_VALUE)
            / max(abs(i) for i in data_all))
        for i in data_all:
            r.append(int(i * normalize_factor))
        return r

    def trim(self, data_all):
        _from = 0
        _to = len(data_all) - 1

        for i, b in enumerate(data_all):
            if abs(b) > RecordWav.THRESHOLD:
                _from = max(0, i - RecordWav.TRIM_APPEND)
                break

        for i, b in enumerate(reversed(data_all)):
            if abs(b) > RecordWav.THRESHOLD:
                _to = min(len(data_all) - 1, len(data_all) - 1 - i + RecordWav.TRIM_APPEND)
                break

        _from = int(_from)
        _to = int(_to)

        return copy.deepcopy(data_all[_from:(_to + 1)])

    def record(self):
        stream = self.p.open(format=RecordWav.FORMAT, channels=RecordWav.CHANNELS, rate=RecordWav.RATE,
            input=True, output=False, frames_per_buffer=RecordWav.CHUNK_SIZE)

        silent_chunks = 0
        audio_started = False
        data_all = array('h')
        silent_count = 0

        while not rospy.is_shutdown() and self.enable_recognition:
            data_chunk = array('h', stream.read(RecordWav.CHUNK_SIZE))
            if byteorder == 'big':
                data_chunk.byteswap()
            data_all.extend(data_chunk)

            silent = self.is_silent(data_chunk)

            if audio_started:
                if silent:
                    silent_chunks += 1
                    if silent_chunks > RecordWav.SILENT_CHUNKS:
                        rospy.logdebug('Recording Stoped...')
                        self.pub_end_speech.publish()
                        break
                else:
                    silent_chunks = 0

            elif not silent:
                audio_started = True
                if self.enable_recognition:
                    rospy.logdebug('Recording started...')
                    self.pub_start_speech.publish()
            else:
                silent_count += 1
                if silent_count > 10:
                    silent_count = 0;
                    rospy.logdebug("Silency detected...")
                    self.pub_silency_detected.publish()

        sample_width = self.p.get_sample_size(RecordWav.FORMAT)
        stream.stop_stream()
        stream.close()

        if self.enable_recognition:
            data_all = self.trim(data_all)
            data_all = self.normalize(data_all)

        return self.enable_recognition, sample_width, data_all


class GoogleCloudSpeechNode:
    def __init__(self):
        rospy.init_node('google_speech_node', anonymous=False)

        self.is_speaking_started = False
        self.published_started = False
        self.count_silency_time = 0

        self.client = GoogleCloudSpeech()
        self.recoder = RecordWav()

        self.t1 = threading.Thread(target=self.handle_speech_recognition)
        self.t1.start()

        rospy.loginfo('%s initialized...'%rospy.get_name())
        rospy.spin()

    def handle_speech_recognition(self):
        while not rospy.is_shutdown():
            result, sample_width, data = self.recoder.record()
            if not result:
                rospy.sleep(0.1)
                continue

            if not rospy.is_shutdown():
                data = pack('<' + ('h' * len(data)), *data)
                wave_file = wave.open(AUDIO_FILE, 'wb')
                wave_file.setnchannels(RecordWav.CHANNELS)
                wave_file.setsampwidth(sample_width)
                wave_file.setframerate(RecordWav.RATE)
                wave_file.writeframes(data)
                wave_file.close()

                self.client.recognize()

if __name__ == '__main__':
    m = GoogleCloudSpeechNode()
