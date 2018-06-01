#!/usr/bin/python
#-*- encoding: utf8 -*-

from __future__ import division

import os
import yaml
import pyaudio
import contextlib
from ctypes import *

import rospy
from dynamic_reconfigure.server import Server
from google_cloud_speech.msg import RecognizedWord
from google_cloud_speech.cfg import RecognitionConfig

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from six.moves import queue


RATE = 8000
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
        self.language_code = 'en_US' #default language code
        self.conf_srv = Server(RecognitionConfig, self.callback_config)
        self.vocabulary_file = rospy.get_param('~vocabulary_file', '')
        self.vocabulary = []
        if self.vocabulary_file != '':
            target_file = os.path.abspath(self.vocabulary_file)
            target_file = os.path.expanduser(self.vocabulary_file)
            with open(target_file) as f:
                self.vocabulary = yaml.load(f)
                rospy.loginfo('load and set user vocabulary...')

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

            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

                responses = client.streaming_recognize(streaming_config, requests)
                self.listen_and_loop(responses)


    def listen_and_loop(self, responses):
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            if not result.is_final:
                pass

            else:
                print(transcript)
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
