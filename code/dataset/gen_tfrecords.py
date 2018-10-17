# generate tfrecords

# TODO: clean up
# TODO: update the readme
# TODO: better CLI descriptions/syntax
# TODO: restoring a model without recreating it (use constants / op names in the code?)
# TODO: move all the training parameters inside the training parser
# TODO: switch to https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn instead of buckets

import sys
import argparse
import logging

import tensorflow as tf
from defaults_dataset import Config
import dataset_multi_tfrecord as dataset

tf.logging.set_verbosity(tf.logging.ERROR)


def process_args(args, defaults):
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-path', dest="log_path",
                        metavar=defaults.LOG_PATH,
                        type=str, default=defaults.LOG_PATH,
                        help=('log file path (default: %s)'
                              % (defaults.LOG_PATH)))

    parser.add_argument('--max-samples', dest='max_samples',
                        type=int, default=defaults.MAX_SAMPLES_PER_TFRECORD,
                        metavar=defaults.MAX_SAMPLES_PER_TFRECORD,
                        help=('max samples number per tfrecord dataset (default: %s)'
                              % defaults.MAX_SAMPLES_PER_TFRECORD))

    parser.add_argument('--dataset', dest='dataset', default=defaults.DATASET,
                        help=('dataset'))
    parser.add_argument('--data_home', dest='data_home', default=defaults.DATA_HOME,
                        help=('path to the dataset'))
    parser.add_argument('--annotations_path', dest='annotations_path',
                        default=defaults.ANNOTATION_PATH,
                        help=('path to the annotation file'))
    parser.add_argument('--output_path', dest='output_path',
                        default=defaults.OUTPUT_PATH,
                        help='output path')

    parser.add_argument('--log-step', dest='log_step',
                        type=int, default=defaults.LOG_STEP,
                        metavar=defaults.LOG_STEP,
                        help=('print log messages every N steps (default: %s)'
                              % defaults.LOG_STEP))
    parser.add_argument('--no-force-uppercase', dest='force_uppercase',
                        action='store_false', default=defaults.FORCE_UPPERCASE,
                        help='do not force uppercase on label values')
    parser.add_argument('--save-filename', dest='save_filename',
                        action='store_true', default=defaults.SAVE_FILENAME,
                        help='save filename as a field in the dataset')

    parameters = parser.parse_args(args)
    return parameters


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    print(parameters)

    dataset.generate(parameters.max_samples,
                     parameters.data_home,
                     parameters.dataset,
                     parameters.annotations_path,
                     parameters.output_path,
                     parameters.log_step,
                     parameters.force_uppercase,
                     parameters.save_filename)
    # return


if __name__ == "__main__":
    main()
