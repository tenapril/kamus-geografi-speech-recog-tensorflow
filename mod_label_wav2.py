# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import audioop
from collections import deque
import os
import time
import math
import tensorflow as tf
import pyaudio
import wave
# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 512
# RECORD_SECONDS = 1
# WAVE_OUTPUT_FILENAME = "file.wav"
CHUNK = 1400  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
THRESHOLD = 1000  # The threshold intensity that defines silence
                  # and noise signal (an int. lower than THRESHOLD is silence).

SILENCE_LIMIT = 0.375  # Silence limit in seconds. The max ammount of seconds where
                   # only silence is recorded. When this time passes the
                   # recording finishes and the file is delivered.

PREV_AUDIO = 0.125  # Previous audio (in seconds) to prepend. When noise
                  # is detected, how much of previously recorded audio is
                  # prepended. This helps to prevent chopping the beggining
                  # of the phrase.



def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0

def listen_for_speech(labels, graph, input_name, output_name, how_many_labels,threshold=THRESHOLD, num_phrases=-1):
    """
    Listens to Microphone, extracts phrases from it and sends it to
    Google's TTS service and returns response. a "phrase" is sound
    surrounded by silence (according to threshold). num_phrases controls
    how many phrases to process before finishing the listening process
    (-1 for infinite).
    """

    #Open stream
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print ("* Listening mic. ")
    audio2send = []
    cur_data = ''  # current chunk  of audio data
    rel = RATE/CHUNK
    rel = int(rel)
    slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
    #Prepend audio from 0.5 seconds before noise was detected
    prev_audio = deque(maxlen=int(PREV_AUDIO * rel))
    started = False
    n = num_phrases
    response = []

    while (num_phrases == -1 or n > 0):
        stream.close()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        cur_data = stream.read(CHUNK)
        slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
        #print slid_win[-1]
        if(sum([x > THRESHOLD for x in slid_win]) > 0):
            if(not started):
                print ("Starting record of phrase")
                started = True
            audio2send.append(cur_data)
        elif (started is True):
            print ("Finished")
            # The limit was reached, finish capture and deliver.
            filename = save_speech(labels, graph, input_name, output_name, how_many_labels,list(prev_audio) + audio2send, p)
            # Send file to Google and get response
            # r = stt_google_wav(filename)
            # if num_phrases == -1:
            #     print ("Response", r)
            # else:
            #     response.append(r)
            # Remove temp file. Comment line to review.
            # os.remove(filename)
            # Reset all
            started = False
            slid_win = deque(maxlen=int(SILENCE_LIMIT * rel))
            prev_audio = deque(maxlen=int(rel/2))
            audio2send = []
            n -= 1
            print ("Listening ...")
        else:
            prev_audio.append(cur_data)

    print ("* Done recording")
    stream.close()
    p.terminate()

    return 0


def save_speech(labels, graph, input_name, output_name, how_many_labels,data, p):
    """ Saves mic data to temporary WAV file. Returns filename of saved
        file """
    labels_list = load_labels(labels)
    # filename = 'output_'+str(int(time.time()))
    # writes data to WAV file
    data = b"".join(data)
    wf = wave.open('tensorflow-master-233/tensorflow/examples/speech_commands/file.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)  # TODO make this value a function parameter?
    wf.writeframes(data)
    wf.close()
    with open('tensorflow-master-233/tensorflow/examples/speech_commands/file.wav', 'rb') as wav_file:
        wav_data = wav_file.read()
    run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
    return 'file.wav'


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  listen_for_speech(labels=labels, graph=graph, input_name=input_name, output_name=output_name, how_many_labels=how_many_labels)
  # audio = pyaudio.PyAudio()
  # for x in range(1,10):
  #   try:
  #       input("Press enter to continue")
  #   except SyntaxError:
  #       pass
  #   # start Recording
  #   stream = audio.open(format=FORMAT, channels=CHANNELS,rate=RATE, input=True,frames_per_buffer=CHUNK)
  #   print ('recording...')
  #   frames = []
  #   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
  #     data = stream.read(CHUNK)
  #     frames.append(data)
  #   print ('finished recording')
  #   # stop Recording
  #   stream.stop_stream()
  #   stream.close()
  #   waveFile = wave.open("tensorflow-master/tensorflow/examples/speech_commands/"+WAVE_OUTPUT_FILENAME, 'wb')
  #   waveFile.setnchannels(CHANNELS)
  #   waveFile.setsampwidth(audio.get_sample_size(FORMAT))
  #   waveFile.setframerate(RATE)
  #   waveFile.writeframes(b''.join(frames))
  #   waveFile.close()
    # with open(wav, 'rb') as wav_file:
    #   wav_data = wav_file.read()
    # run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
  # audio.terminate()


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
