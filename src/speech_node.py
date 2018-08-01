#!/usr/bin/python
#-*- encoding: utf8 -*-

from __future__ import division

import os
import yaml
import pyaudio
import contextlib
from ctypes import *
import pygame

import rospy
import rospkg
from dynamic_reconfigure.server import Server
from google_cloud_speech.msg import RecognizedWord
from google_cloud_speech.cfg import RecognitionConfig
from std_msgs.msg import Bool, Empty
from google_cloud_speech.msg import RecognizedWord

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from six.moves import queue


# [START ignore_error_message_about_libsound]
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

asound = cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(c_error_handler)
# [END ignore_error_message_about_libsound]


RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def stop_recording(self):
        self.closed = True

    def generator(self):
        while not self.closed and not rospy.is_shutdown():
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)


class GoogleCloudSpeechNode:
    def __init__(self):
        pygame.init()

        self.begin_sound = os.path.join(
            rospkg.RosPack().get_path('google_cloud_speech'), 'resources', 'begin_record.wav')
        self.end_sound = os.path.join(
            rospkg.RosPack().get_path('google_cloud_speech'), 'resources', 'end_record.wav')

        rospy.Subscriber('enable_recognition', Bool, self.handle_enable_recognition)
        self.pub_recognized_word = rospy.Publisher('recognized_word', RecognizedWord, queue_size=10)
        self.pub_start_speech = rospy.Publisher('start_of_speech', Empty, queue_size=10)
        self.pub_end_speech = rospy.Publisher('end_of_speech', Empty, queue_size=10)

        self.language_code = 'en-US' #default language code
        self.conf_srv = Server(RecognitionConfig, self.callback_config)
        self.vocabulary_file = rospy.get_param('~vocabulary_file', '')
        self.vocabulary = []
        if self.vocabulary_file != '':
            target_file = os.path.abspath(self.vocabulary_file)
            target_file = os.path.expanduser(self.vocabulary_file)
            with open(target_file) as f:
                self.vocabulary = yaml.load(f)
                rospy.loginfo('load and set user vocabulary...')
        self.enable_recognition = False

        client = speech.SpeechClient()

        while not rospy.is_shutdown():
            config = types.RecognitionConfig(
                encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code=self.language_code)
            streaming_config = types.StreamingRecognitionConfig(
                config=config,
                single_utterance=True,
                interim_results=True)

            if not self.enable_recognition:
                rospy.sleep(0.1)
                continue

            pygame.mixer.music.load(self.begin_sound)
            pygame.mixer.music.play()

            with MicrophoneStream(RATE, CHUNK) as self.stream:
                audio_generator = self.stream.generator()
                requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

                responses = client.streaming_recognize(streaming_config, requests)
                self.listen_and_loop(responses)

            self.enable_recognition = False

    def handle_enable_recognition(self, msg):
        self.enable_recognition = msg.data
        if not msg.data:
            pygame.mixer.music.load(self.end_sound)
            pygame.mixer.music.play()
            self.stream.stop_recording()

    def listen_and_loop(self, responses):
        is_started_speech = False

        for response in responses:
            if response.speech_event_type == 1:
                pygame.mixer.music.load(self.end_sound)
                pygame.mixer.music.play()

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            if not result.is_final:
                if not is_started_speech:
                    is_started_speech = True
                    self.pub_start_speech.publish()
            else:
                is_started_speech = False
                self.pub_end_speech.publish()

                msg = RecognizedWord()
                msg.recognized_word = result.alternatives[0].transcript
                msg.confidence = result.alternatives[0].confidence

                self.pub_recognized_word.publish(msg)
                return


    def callback_config(self, config, level):
        self.language_code = config['language']
        rospy.loginfo('set recognition language code to [%s]...'%self.language_code)
        return config


if __name__ == '__main__':
    rospy.init_node('google_cloud_speech', anonymous=False)
    try:
        m = GoogleCloudSpeechNode()
        rospy.spin()
    except rospy.ROSInterruptException: pass
