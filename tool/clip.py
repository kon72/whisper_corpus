import glob
import json
import os

from absl import app
from absl import flags
from scipy.io import wavfile

FLAGS = flags.FLAGS

flags.DEFINE_string('transcription_dir', '.',
                    'Directory to read transcription files from.')
flags.DEFINE_string('audio_dir', '.', 'Directory to read audio files from.')
flags.DEFINE_string('output_dir', '.', 'Directory to write audio files to.')


def main(argv):
  transcription_paths = glob.glob(
      os.path.join(FLAGS.transcription_dir, '*.json'))
  for transcription_path in transcription_paths:
    with open(transcription_path, 'r', encoding='utf-8') as f:
      transcription = json.load(f)
    segments = transcription['segments']

    name = os.path.splitext(os.path.basename(transcription_path))[0]
    audio_path = os.path.join(FLAGS.audio_dir, name + '.wav')
    sample_rate, audio = wavfile.read(audio_path)

    for segment in segments:
      start_seconds = segment['start']
      end_seconds = segment['end']
      start = round(start_seconds * sample_rate)
      end = round(end_seconds * sample_rate)
      segment_audio = audio[start:end]

      segment_id = segment['id']
      output_path = os.path.join(FLAGS.output_dir, f'{name}_{segment_id}.wav')
      wavfile.write(output_path, sample_rate, segment_audio)


if __name__ == '__main__':
  app.run(main)
