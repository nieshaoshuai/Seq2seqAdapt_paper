# TODO: better CLI descriptions/syntax
# TODO: move all the training parameters inside the training parser

'''
Required:
tensorflow 1.4.1
python 3.0, where the dict, map and decode function are different from python 2.0

Usage:
(0) There must be a pretrained source model
  if there is none, please run " python main_baseline.py" first

(1)Training
   python main_baseline.py --phase='train'

(2)Test
   python main_baseline.py --phase='test'

'''

import argparse
import logging
import sys

import numpy as np
import tensorflow as tf

from defaults import Config
from model_baseline import Model



tf.logging.set_verbosity(tf.logging.ERROR)


def process_args(args, defaults):

    parser = argparse.ArgumentParser()
    # Global arguments
    parser.add_argument('--log-path', dest="log_path",
                             metavar=defaults.LOG_PATH,
                             type=str, default=defaults.LOG_PATH,
                             help=('log file path (default: %s)'
                                   % (defaults.LOG_PATH)))

    parser.add_argument('--phase', dest="phase",
                              type=str, default='train',
                              help=('train,test,predict,export'))
    parser.add_argument('--is_da', dest='is_da', default=False,
                        help=('True,False'))


    parser.set_defaults(visualize=defaults.VISUALIZE)

    parser.set_defaults(load_model=defaults.LOAD_MODEL)
    # parser.set_defaults(load_model=False)



    parser.add_argument('--max-width', dest="max_width",
                              metavar=defaults.MAX_WIDTH,
                              type=int, default=defaults.MAX_WIDTH,
                              help=('max image width (default: %s)'
                                    % (defaults.MAX_WIDTH)))
    parser.add_argument('--max-height', dest="max_height",
                              metavar=defaults.MAX_HEIGHT,
                              type=int, default=defaults.MAX_HEIGHT,
                              help=('max image height (default: %s)'
                                    % (defaults.MAX_HEIGHT)))
    parser.add_argument('--max-prediction', dest="max_prediction",
                              metavar=defaults.MAX_PREDICTION,
                              type=int, default=defaults.MAX_PREDICTION,
                              help=('max length of predicted strings (default: %s)'
                                    % (defaults.MAX_PREDICTION)))
    # parser.add_argument('--full-ascii', dest='full_ascii', action='store_true',
    #                           help=('use lowercase in addition to uppercase'))
    # parser.set_defaults(full_ascii=defaults.FULL_ASCII)
    parser.add_argument('--color', dest="channels", action='store_const', const=3,
                              default=defaults.CHANNELS,
                              help=('do not convert source images to grayscale'))
    parser.add_argument('--no-distance', dest="use_distance", action="store_false",
                              default=defaults.USE_DISTANCE,
                              help=('require full match when calculating accuracy'))
    parser.add_argument('--gpu-id', dest="gpu_id", metavar=defaults.GPU_ID,
                              type=int, default=defaults.GPU_ID,
                              help='specify a GPU ID')


    parser.add_argument('--use-gru', dest='use_gru', action='store_true',
                              help='use GRU instead of LSTM')

    parser.add_argument('--attn-num-layers', dest="attn_num_layers",
                              type=int, default=defaults.ATTN_NUM_LAYERS,
                              metavar=defaults.ATTN_NUM_LAYERS,
                              help=('hidden layers in attention decoder cell (default: %s)'
                                    % (defaults.ATTN_NUM_LAYERS)))
    parser.add_argument('--attn-num-hidden', dest="attn_num_hidden",
                              type=int, default=defaults.ATTN_NUM_HIDDEN,
                              metavar=defaults.ATTN_NUM_HIDDEN,
                              help=('hidden units in attention decoder cell (default: %s)'
                                    % (defaults.ATTN_NUM_HIDDEN)))
    parser.add_argument('--target-embedding-size', dest="target_embedding_size",
                        type=int, default=defaults.TARGET_EMBEDDING_SIZE,
                        metavar=defaults.TARGET_EMBEDDING_SIZE,
                        help=('embedding dimension for each target (default: %s)'
                              % (defaults.TARGET_EMBEDDING_SIZE)))


    parser.add_argument('--initial-learning-rate', dest="initial_learning_rate",
                              type=float, default=defaults.INITIAL_LEARNING_RATE,
                              metavar=defaults.INITIAL_LEARNING_RATE,
                              help=('initial learning rate (default: %s)'
                                    % (defaults.INITIAL_LEARNING_RATE)))
    parser.add_argument('--max-gradient-norm', dest="max_gradient_norm",
                        type=int, default=defaults.MAX_GRADIENT_NORM,
                        metavar=defaults.MAX_GRADIENT_NORM,
                        help=('clip gradients to this norm (default: %s)'
                              % (defaults.MAX_GRADIENT_NORM)))
    parser.add_argument('--no-gradient-clipping', dest='clip_gradients', action='store_false',
                        help=('do not perform gradient clipping'))
    parser.set_defaults(clip_gradients=defaults.CLIP_GRADIENTS)




    parser.add_argument('--output-dir', dest="output_dir",
                              type=str, default=defaults.OUTPUT_DIR,
                              metavar=defaults.OUTPUT_DIR,
                              help=('output directory (default: %s)'
                                    % (defaults.OUTPUT_DIR)))
    parser.add_argument('--model-dir', dest="model_dir",
                              type=str, default=defaults.MODEL_DIR,
                              metavar=defaults.MODEL_DIR,
                              help=('directory for the model '
                                    '(default: %s)' %(defaults.MODEL_DIR)))

    parser.add_argument('--dataset_a', dest="dataset_a",
                        type=str, default=defaults.SOURCE_DATASET,
                        help=('train,test,predict,vi export'))
    parser.add_argument('--phase_a', dest="phase_a",
                        type=str, default=defaults.SOURCE_DATASET_PHASE,
                        help=('train,test,predict,export'))

    parser.add_argument('--dataset_b', dest="dataset_b",
                        type=str, default=defaults.TARGET_DATASET,
                        help=('train,test,predict,export'))
    parser.add_argument('--phase_b', dest="phase_b",
                        type=str, default=defaults.TARGET_DATASET_PHASE,
                        help=('train,test,predict,export'))

    parser.add_argument('--dataset_path_a', dest='dataset_path_a',
                              type=str, default=defaults.DATA_PATH_A,
                              help=('training dataset in the TFRecords format'
                                    ' (default: %s)'
                                    % (defaults.DATA_PATH_A)))
    parser.add_argument('--dataset_path_b', dest='dataset_path_b',
                        type=str, default=defaults.DATA_PATH_B,
                        help=('training dataset in the TFRecords format'
                              ' (default: %s)'
                              % (defaults.DATA_PATH_B)))
    parser.add_argument('--steps-per-checkpoint', dest="steps_per_checkpoint",
                              type=int, default=defaults.STEPS_PER_CHECKPOINT,
                              #metavar=defaults.STEPS_PER_CHECKPOINT,
                              help=('steps between saving the model'
                                    ' (default: %s)'
                                    % (defaults.STEPS_PER_CHECKPOINT)))

    # Shared model arguments
    parser.add_argument('--max-samples', dest='max_samples',
                        type=int, default=defaults.MAX_SAMPLES_PER_TFRECORD,
                        metavar=defaults.MAX_SAMPLES_PER_TFRECORD,
                        help=('max samples number per tfrecord dataset (default: %s)'
                              % defaults.MAX_SAMPLES_PER_TFRECORD))

    parser.add_argument('--train-num', dest="train_num",
                        type=int, default=defaults.TRAIN_NUM,
                        metavar=defaults.TRAIN_NUM,
                        help=('batch size (default: %s)'
                              % (defaults.TRAIN_NUM)))
    parser.add_argument('--batch-size', dest="batch_size",
                              type=int, default=defaults.BATCH_SIZE,
                              metavar=defaults.BATCH_SIZE,
                              help=('batch size (default: %s)'
                                    % (defaults.BATCH_SIZE)))
    parser.add_argument('--num-epoch', dest="num_epoch",
                              type=int, default=defaults.NUM_EPOCH,
                              metavar=defaults.NUM_EPOCH,
                              help=('number of training epochs (default: %s)'
                                    % (defaults.NUM_EPOCH)))
    parser.add_argument('--no-resume', dest='load_model', action='store_false',
                              help=('create a new model even if checkpoints already exist'))


    parameters = parser.parse_args(args)
    return parameters


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)
    if parameters.phase == "test":
        parameters.log_path = "test_" + parameters.log_path
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    # manualSeed = 6810  # fix seed
    manualSeed = np.random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    tf.set_random_seed(manualSeed)
    # torch.manual_seed(opt.manualSeed)
    logging.info("random_seed:{}".format(manualSeed))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:

        # if parameters.full_ascii:
        #     DataGen.setFullAsciiCharmap()

        model = Model(
            sess,
            parameters,
            phase=parameters.phase,
            visualize=parameters.visualize,
            output_dir=parameters.output_dir,
            batch_size=parameters.batch_size,
            initial_learning_rate=parameters.initial_learning_rate,
            steps_per_checkpoint=parameters.steps_per_checkpoint,
            model_dir=parameters.model_dir,
            target_embedding_size=parameters.target_embedding_size,
            attn_num_hidden=parameters.attn_num_hidden,
            attn_num_layers=parameters.attn_num_layers,
            clip_gradients=parameters.clip_gradients,
            max_gradient_norm=parameters.max_gradient_norm,
            load_model=parameters.load_model,
            gpu_id=parameters.gpu_id,
            use_gru=parameters.use_gru,
            use_distance=parameters.use_distance,
            max_image_width=parameters.max_width,
            max_image_height=parameters.max_height,
            max_prediction_length=parameters.max_prediction,
            channels=parameters.channels,
        )

        if parameters.phase == 'train':
            print("parameters.train_num", parameters.train_num)
            model.train(
                data_path_a=parameters.dataset_path_a,
                data_path_b=parameters.dataset_path_b,
                num_epoch=parameters.num_epoch,
                train_num=parameters.train_num
            )
            logging.info("random_seed:{}".format(manualSeed))
            print("training is finished.......")
        elif parameters.phase == 'test':
            # word_acc, char_acc = model.test(
            #     data_path=parameters.test_dataset_path
            # )
            # word_acc, char_acc = model.test(parameters.dataset_path_b, print_info=False)
            # logging.info(
            #     ' sample - word_acc: {:6.2%} char_acc: {:6.2%}'.format(
            #         word_acc,
            #         char_acc))
            logging.info("baseline.........")
            word_acc, char_acc = model.test('/home/data/OCR/IAM/words/standard_split_ok/test_tfrecords',
                                            selected_num=3e4,
                                            print_info=False)
            logging.info(
                ' iam-words -standard-split-ok- word_acc: {:6.2%} char_acc: {:6.2%}'.format(
                    word_acc,
                    char_acc))
            word_acc, char_acc = model.test('/home/data/OCR/IAM/words/test_tfrecords',
                                            selected_num=3e4,
                                            print_info=False)
            logging.info(
                ' iam-words - word_acc: {:6.2%} char_acc: {:6.2%}'.format(
                    word_acc,
                    char_acc))

            word_acc, char_acc = model.test('/home/data/OCR/evaluation_data/icdar13/test_tfrecords',
                                            print_info=False)
            logging.info(
                ' icdar13 - word_acc: {:6.2%} char_acc: {:6.2%}'.format(
                    word_acc,
                    char_acc))

            word_acc, char_acc = model.test('/home/data/OCR/evaluation_data/icdar03/test_tfrecords',
                                            print_info=False)
            logging.info(
                ' icdar03 - word_acc: {:6.2%} char_acc: {:6.2%}'.format(
                    word_acc,
                    char_acc))

            word_acc, char_acc = model.test('/home/data/OCR/evaluation_data/svt/test_tfrecords',
                                            print_info=False)
            logging.info(
                ' svt - word_acc: {:6.2%} char_acc: {:6.2%}'.format(
                    word_acc,
                    char_acc))

            word_acc, char_acc = model.test('/home/data/OCR/evaluation_data/iiit5k/test_tfrecords',
                                            print_info=False)
            logging.info(
                ' iiit5k - word_acc: {:6.2%} char_acc: {:6.2%}'.format(
                    word_acc,
                    char_acc))
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
