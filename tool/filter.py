import glob
import json
import os

from absl import app
from absl import flags
from absl import logging
from dateutil import relativedelta
from scipy.io import wavfile
import simpleaudio
import termcolor

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


def print_segment(segment: dict, edits: dict, idx: int, num_segments: int):
  start_seconds_humanized = humanize_duration(segment['start'])
  start_delta_seconds = edits['start_delta'][idx]
  if start_delta_seconds != 0:
    start_seconds_humanized += termcolor.colored(
        f' ({start_delta_seconds:+0.3f}s)', 'yellow')
  end_seconds_humanized = humanize_duration(segment['end'])
  end_delta_seconds = edits['end_delta'][idx]
  if end_delta_seconds != 0:
    end_seconds_humanized += termcolor.colored(f' ({end_delta_seconds:+0.3f}s)',
                                               'yellow')

  text_humanized = segment['text']
  if edits['text'][idx] is not None:
    text_humanized = termcolor.colored(edits['text'][idx], 'yellow')
    text_humanized += f' (edited){os.linesep}'
    text_humanized += f'Original text: {segment["text"]}'

  print(f'[{idx + 1}/{num_segments}] {start_seconds_humanized} - '
        f'{end_seconds_humanized}: {text_humanized}')
  print(f'      avg_logprob: {segment["avg_logprob"]:0.3f}')
  print(f'compression_ratio: {segment["compression_ratio"]:0.3f}')
  print(f'   no_speech_prob: {segment["no_speech_prob"]:0.3f}')
  word_probs = [word['probability'] for word in segment['words']]
  word_probs_humanized = ', '.join([f'{p:0.3f}' for p in word_probs])
  print(f'       word_probs: [{word_probs_humanized}]')


def find_next_undecided_segment_idx(edits: dict, start_idx: int) -> int:
  for i in range(start_idx, len(edits['keep'])):
    if edits['keep'][i] is None:
      return i
  for i in range(0, start_idx):
    if edits['keep'][i] is None:
      return i
  return -1


def input_int(prompt: str) -> int:
  while True:
    try:
      return int(input(prompt).strip())
    except ValueError:
      print('Invalid input. Please enter an integer.')


def save_backup(temp_path: str, segments: list, edits: dict):
  contents = json.dumps({
      'segments': segments,
      'edits': edits,
  })
  with open(temp_path, 'w', encoding='utf-8') as f:
    f.write(contents)


def save_segments(output_path: str, segments: list):
  contents = json.dumps({
      'segments': segments,
  })
  with open(output_path, 'w', encoding='utf-8') as f:
    f.write(contents)


def filter_segments(segments: list, edits: dict) -> list:
  filtered_segments = []
  for i, segment in enumerate(segments):
    if not edits['keep'][i]:
      continue
    segment_to_add = {
        'id':
            segment['id'],
        'start':
            segment['start'] + edits['start_delta'][i],
        'end':
            segment['end'] + edits['end_delta'][i],
        'text':
            edits['text'][i]
            if edits['text'][i] is not None else segment['text'],
    }
    filtered_segments.append(segment_to_add)
  return filtered_segments


def do_filtering(transcription_path: str, audio_path: str, output_path: str,
                 temp_path: str):
  if os.path.exists(temp_path):
    logging.info('Resuming %s...', transcription_path)
    with open(temp_path, 'r', encoding='utf-8') as f:
      temp = json.load(f)
    segments = temp['segments']
    edits = temp['edits']
  else:
    logging.info('Starting %s...', transcription_path)
    with open(transcription_path, 'r', encoding='utf-8') as f:
      transcription = json.load(f)
    segments = transcription['segments']
    edits = {
        'start_delta': [0] * len(segments),
        'end_delta': [0] * len(segments),
        'text': [None] * len(segments),
        'keep': [None] * len(segments),
    }

  sample_rate, audio = wavfile.read(audio_path)
  whole_duration_seconds = audio.shape[0] / sample_rate

  idx = find_next_undecided_segment_idx(edits, 0)
  while idx != -1:
    segment = segments[idx]
    start_seconds = segment['start'] + edits['start_delta'][idx]
    end_seconds = segment['end'] + edits['end_delta'][idx]
    start_sample = int(start_seconds * sample_rate)
    end_sample = int(end_seconds * sample_rate)
    segment_audio = audio[start_sample:end_sample]

    print_segment(segment, edits, idx, len(segments))

    # Play audio
    if segment_audio.shape[0] > 0:
      play_obj = simpleaudio.play_buffer(segment_audio, 1, 2, sample_rate)
    else:
      play_obj = None

    # Ask user for input
    while True:
      try:
        keep = input('Keep? [y/n/r/b/s/e/t/f/q/?] ').strip().lower()
        if keep == 'y':
          # Keep this segment
          edits['keep'][idx] = True
          idx = find_next_undecided_segment_idx(edits, idx)
          break
        elif keep == 'n':
          # Skip this segment
          edits['keep'][idx] = False
          idx = find_next_undecided_segment_idx(edits, idx)
          break
        elif keep == 'r':
          # Replay this segment
          break
        elif keep == 'b':
          # Back to previous segment
          if idx > 0:
            idx -= 1
            edits['keep'][idx] = None
          break
        elif keep == 's':
          # Shift start time of this segment
          delta = input_int('Start delta (ms): ') / 1000
          edits['start_delta'][idx] = delta
          new_start = segment['start'] + delta
          if new_start < 0:
            termcolor.cprint(f'Start time {new_start} cannot be negative',
                             'red')
            continue
          if idx > 0:
            prev_end = segments[idx - 1]['end'] + edits['end_delta'][idx - 1]
            if new_start < prev_end:
              termcolor.cprint(
                  f'Start time {new_start} is before previous segment end '
                  f'time {prev_end}', 'yellow')
          break
        elif keep == 'e':
          # Shift end time of this segment
          delta = input_int('End delta (ms): ') / 1000
          edits['end_delta'][idx] = delta
          new_end = segment['end'] + delta
          if new_end > whole_duration_seconds:
            termcolor.cprint(f'End time {new_end} is after end of audio file',
                             'red')
            continue
          if idx < len(segments) - 1:
            next_start = segments[idx + 1]['start'] + edits['start_delta'][idx +
                                                                           1]
            if new_end > next_start:
              termcolor.cprint(
                  f'End time {new_end} is after next segment start time '
                  f'{next_start}', 'yellow')
          break
        elif keep == 't':
          # Edit the text of this segment
          new_text = input('New text: ')
          if new_text.strip() == '':
            new_text = None
          edits['text'][idx] = new_text
          break
        elif keep == 'f':
          filtered_segments = filter_segments(segments, edits)
          save_segments(output_path, filtered_segments)
          raise KeyboardInterrupt
        elif keep == 'q':
          # Quit
          raise KeyboardInterrupt
        elif keep == '?':
          # Print help
          print('y: Keep this segment')
          print('n: Skip this segment')
          print('r: Replay this segment')
          print('b: Back to previous segment')
          print('s: Shift start time of this segment')
          print('e: Shift end time of this segment')
          print('t: Edit the text of this segment')
          print('f: Finish editing and export to JSON file')
          print('q: Quit')
          print('?/h: Print help')
      except KeyboardInterrupt:
        print('Aborting...')
        return

    print()

    if play_obj is not None:
      play_obj.stop()

    save_backup(temp_path, segments, edits)

  filtered_segments = filter_segments(segments, edits)
  save_segments(output_path, filtered_segments)


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

    temp_path = os.path.join(FLAGS.output_dir, name + '.json.temp')

    do_filtering(transcription_path, audio_path, output_path, temp_path)


if __name__ == '__main__':
  app.run(main)
