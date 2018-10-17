"""
Default parameters
"""
import os


class Config:
    GPU_ID = 0

    DATA_HOME = '/home/data/OCR/'

    DATASET = 'sample'
    DATASET_PHASE = 'test'  # 'train', 'valid1', 'test'

    if DATASET == 'sample':

        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, 'sample.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, 'sample_tfrecords')

    elif DATASET == 'mjsynth':
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_label.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'iiit5k':

        DATA_HOME = os.path.join(DATA_HOME, "evaluation_data")
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET,
                                       DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'svt':
        DATA_HOME = os.path.join(DATA_HOME, "evaluation_data")
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET,
                                       DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'icdar03':
        DATA_HOME = os.path.join(DATA_HOME, "evaluation_data")
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET,
                                       DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'icdar13':
        DATA_HOME = os.path.join(DATA_HOME, "evaluation_data")
        ANNOTATION_PATH = os.path.join(DATA_HOME, DATASET,
                                       DATASET_PHASE + '.txt')
        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'IAM/words':
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
            print(ANNOTATION_PATH + DATASET_PHASE + "is not existed")
            raise NotImplementedError

        OUTPUT_PATH = os.path.join(DATA_HOME, DATASET, DATASET_PHASE + '_tfrecords')

    elif DATASET == 'IAM/words/standard_split_ok':
        annotation_path = os.path.join(DATA_HOME, DATASET)
        DATASET = "IAM/words/"
        if DATASET_PHASE == "train":
            ANNOTATION_PATH = os.path.join(annotation_path, "train_label.txt")
        elif DATASET_PHASE == "valid":
            ANNOTATION_PATH = os.path.join(annotation_path, "valid_label.txt")
        elif DATASET_PHASE == "test":
            ANNOTATION_PATH = os.path.join(annotation_path, "eval_label.txt")
        else:
            ANNOTATION_PATH = annotation_path
            print(ANNOTATION_PATH + DATASET_PHASE + "is not existed")
            raise NotImplementedError
        OUTPUT_PATH = os.path.join(annotation_path, DATASET_PHASE + '_tfrecords')
    else:
        print("please check your dataset name is in (mjsynth, iiit5k, svt, icdar03, icdar13,IAM)")
        raise NotImplementedError

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    FORCE_UPPERCASE = True
    SAVE_FILENAME = True
    FULL_ASCII = False
    TARGET_VOCAB_SIZE = 26 + 10 + 3  # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z
    CHANNELS = 1  # number of color channels from source image (1 = grayscale, 3 = rgb)

    MAX_SAMPLES_PER_TFRECORD = 10000

    LOG_STEP = 500
    LOG_PATH = 'aocr.log'
