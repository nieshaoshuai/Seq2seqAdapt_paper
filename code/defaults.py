"""
Default parameters
"""
import os


def pair_data_path(dataset, dataset_phase):
    DATASET = dataset
    DATASET_PHASE = dataset_phase
    if DATASET == 'sample':
        DATA_HOME = '/home/data/OCR/'
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + 'sample.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, 'sample_tfrecords')

    elif DATASET == 'mjsynth':
        DATA_HOME = '/home/data/OCR/'
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_label.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'iiit5k':
        DATA_HOME = '/home/data/OCR/evaluation_data'
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'svt':
        DATA_HOME = '/home/data/OCR/evaluation_data'
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'icdar03':
        DATA_HOME = '/home/data/OCR/evaluation_data'
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'icdar13':
        DATA_HOME = '/home/data/OCR/evaluation_data'
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')
    elif DATASET == 'IAM/words':
        DATA_HOME = '/home/data/OCR/'
        annotation_path = os.path.join(DATA_HOME, DATASET,
                                       "largeWriterIndependentTextLineRecognitionTask")
        if DATASET_PHASE == "train":
            ANNOTATION_PATH = os.path.join(annotation_path, "trainset_label.txt")
        elif DATASET_PHASE == "valid1":
            ANNOTATION_PATH = os.path.join(annotation_path, "validationset1_label.txt")
        elif DATASET_PHASE == "valid2":
            ANNOTATION_PATH = os.path.join(annotation_path, "validationset2_label.txt")
        elif DATASET_PHASE == "test":
            ANNOTATION_PATH = os.path.join(annotation_path, "testset_label.txt")
        else:
            ANNOTATION_PATH = annotation_path
            print(ANNOTATION_PATH + DATASET_PHASE + " is not existed")
            raise NotImplementedError
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')
    else:
        raise NotImplementedError

    return DATA_HOME, ANNOTATION_PATH, OUTPUT_PATH


class Config:
    GPU_ID = 0
    VISUALIZE = False

    # DATASET = 'svt'
    # DATASET_PHASE = 'test'  #'train', 'valid', 'test'
    # pair_data_path(DATASET, DATASET_PHASE)

    FORCE_UPPERCASE = True  # TARGET_VOCAB_SIZE = 26 + 10 + 3  # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z
    SAVE_FILENAME = True
    FULL_ASCII = False

    CHANNELS = 1  # number of color channels from source image (1 = grayscale, 3 = rgb)

    # SOURCE_DATASET = 'mjsynth'
    # SOURCE_DATASET_PHASE = 'valid'
    # TARGET_DATASET = 'mjsynth'
    # TARGET_DATASET_PHASE = 'test'

    SOURCE_DATASET = 'IAM/words'
    SOURCE_DATASET_PHASE = 'train'
    TARGET_DATASET = 'IAM/words'
    TARGET_DATASET_PHASE = 'test'
    _, _, DATA_PATH_A = pair_data_path(SOURCE_DATASET, SOURCE_DATASET_PHASE)
    # DATA_PATH_A = ["/home/data/OCR/mjsynth/train_tfrecords",
    #                "/home/data/OCR/mjsynth/valid_tfrecords",
    #                "/home/data/OCR/mjsynth/test_tfrecords"]
    _, _, DATA_PATH_B = pair_data_path(TARGET_DATASET, TARGET_DATASET_PHASE)
    # _, _, TEST_DATA_PATH = pair_data_path(TARGET_DATASET, TARGET_DATASET_PHASE)

    MODEL_DIR = './checkpoints'
    LOG_PATH = 'aocr_new.log'
    OUTPUT_DIR = './results'
    STEPS_PER_CHECKPOINT = 500
    EXPORT_FORMAT = 'savedmodel'
    EXPORT_PATH = 'exported'

    MAX_SAMPLES_PER_TFRECORD = 10000
    TRAIN_NUM = 6e4
    # Optimization
    NUM_EPOCH = 30
    BATCH_SIZE = 32

    INITIAL_LEARNING_RATE = 0.9

    # Network parameters
    CLIP_GRADIENTS = True  # whether to perform gradient clipping
    MAX_GRADIENT_NORM = 5.0  # Clip gradients to this norm
    TARGET_EMBEDDING_SIZE = 10  # embedding dimension for each target
    ATTN_NUM_HIDDEN = 128  # number of hidden units in attention decoder cell
    ATTN_NUM_LAYERS = 2  # number of layers in attention decoder cell
    # (Encoder number of hidden units will be ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS)

    LOAD_MODEL = True
    OLD_MODEL_VERSION = False

    MAX_WIDTH = 256
    MAX_HEIGHT = 32
    MAX_PREDICTION = 25

    USE_DISTANCE = True

    # Dataset generation
    LOG_STEP = 500
    # LOG_STEP = 500
