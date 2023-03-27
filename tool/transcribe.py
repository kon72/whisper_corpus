import glob
import os

from absl import app
from absl import flags
from absl import logging
import whisper
from whisper import utils as whisper_utils

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', 'medium', whisper.available_models(),
                  'Name of the Whisper model to use.')
flags.DEFINE_string('input_dir', '.', 'Directory to read audio files from.')
flags.DEFINE_string('output_dir', '.', 'Directory to write output files to.')


def main(argv):
  logging.info('Loading model %s...', FLAGS.model)
  model = whisper.load_model(FLAGS.model)
  logging.info('Model loaded.')

  audio_paths = glob.glob(os.path.join(FLAGS.input_dir, '*.wav'))
  writer = whisper_utils.get_writer('json', FLAGS.output_dir)
  for audio_path in audio_paths:
    logging.info('Loading audio %s...', audio_path)
    audio = whisper.load_audio(audio_path)
    # audio = audio[:whisper.audio.SAMPLE_RATE * 120]
    logging.info('Transcribing %s...', audio_path)
    result = model.transcribe(audio,
                              verbose=False,
                              language='en',
                              word_timestamps=True)
    writer(result, audio_path)  # type: ignore


if __name__ == '__main__':
  app.run(main)
