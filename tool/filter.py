import glob
import json
import os

from absl import app
from absl import flags
from absl import logging
from dateutil import relativedelta
from scipy.io import wavfile
import simpleaudio

FLAGS = flags.FLAGS

flags.DEFINE_string('transcription_dir', '.',
                    'Directory to read transcription files from.')
flags.DEFINE_string('audio_dir', '.', 'Directory to read audio files from.')
flags.DEFINE_string('output_dir', '.', 'Directory to write output files to.')


def humanize_duration(seconds: float) -> str:
  delta = relativedelta.relativedelta(microseconds=round(seconds * 1e6))
  milliseconds = round(delta.microseconds / 1000)
  return (f'{delta.hours:02}:{delta.minutes:02}:{delta.seconds:02}'
          f'.{milliseconds:03}')


def print_segment(segment: dict, idx: int, num_segments: int):
  start_seconds = segment['start']
  end_seconds = segment['end']
  print(f'[{idx + 1}/{num_segments}] {humanize_duration(start_seconds)} - '
        f'{humanize_duration(end_seconds)}: {segment["text"]}')
  print(f'      avg_logprob: {segment["avg_logprob"]:0.3f}')
  print(f'compression_ratio: {segment["compression_ratio"]:0.3f}')
  print(f'   no_speech_prob: {segment["no_speech_prob"]:0.3f}')
  word_probs = [word['probability'] for word in segment['words']]
  word_probs_humanized = ', '.join([f'{p:0.3f}' for p in word_probs])
  print(f'       word_probs: [{word_probs_humanized}]')


def do_filtering(transcription_path: str, audio_path: str, output_path: str,
                 temp_path: str):
  with open(transcription_path, 'r', encoding='utf-8') as f:
    transcription = json.load(f)
  segments = transcription['segments']
  sample_rate, audio = wavfile.read(audio_path)

  if os.path.exists(temp_path):
    logging.info('Resuming %s...', transcription_path)
    with open(temp_path, 'r', encoding='utf-8') as f:
      temp = json.load(f)
    filtered_indices = temp['segments']
    start_from = temp['next']
  else:
    logging.info('Starting %s...', transcription_path)
    filtered_indices = []
    start_from = 0

  idx = start_from
  while idx < len(segments):
    segment = segments[idx]
    start_seconds = segment['start']
    end_seconds = segment['end']
    start_sample = int(start_seconds * sample_rate)
    end_sample = int(end_seconds * sample_rate)
    segment_audio = audio[start_sample:end_sample]

    print_segment(segment, idx, len(segments))

    # Play audio
    play_obj = simpleaudio.play_buffer(segment_audio, 1, 2, sample_rate)

    # Ask user for input
    while True:
      try:
        keep = input('Keep? [y/n/r/b/q/?] ')
        if keep == 'y':
          # Yes
          filtered_indices.append(idx)
          idx += 1
          break
        elif keep == 'n':
          # No
          idx += 1
          break
        elif keep == 'r':
          # Replay
          break
        elif keep == 'b':
          # Back
          if idx > 0:
            idx -= 1
            if idx in filtered_indices:
              filtered_indices.remove(idx)
          break
        elif keep == 'q':
          # Quit
          raise KeyboardInterrupt
        elif keep == '?':
          # Help
          print('y: Keep this segment')
          print('n: Skip this segment')
          print('r: Replay this segment')
          print('b: Back to previous segment')
          print('q: Quit')
          print('?/h: Print help')
      except KeyboardInterrupt:
        print('Aborting...')
        return

    play_obj.stop()

    with open(temp_path, 'w', encoding='utf-8') as f:
      json.dump({
          'segments': filtered_indices,
          'next': idx,
      }, f)

  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'segments': [segments[i] for i in filtered_indices],
    }, f)
  os.remove(temp_path)


def main(argv):
  transcription_paths = glob.glob(
      os.path.join(FLAGS.transcription_dir, '*.json'))
  for transcription_path in transcription_paths:
    name = os.path.splitext(os.path.basename(transcription_path))[0]
    audio_path = os.path.join(FLAGS.audio_dir, name + '.wav')

    output_path = os.path.join(FLAGS.output_dir, name + '.json')
    if os.path.exists(output_path):
      logging.info('Skipping %s...', transcription_path)
      continue

    temp_path = os.path.join(FLAGS.output_dir, name + '.temp.json')

    do_filtering(transcription_path, audio_path, output_path, temp_path)


if __name__ == '__main__':
  app.run(main)
