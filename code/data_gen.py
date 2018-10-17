import numpy as np
import tensorflow as tf
import sys

import os
import glob
import random

from bucketdata import BucketData
from PIL import Image
from six import BytesIO as IO


def load_char_list(char_list_path):
    charlist = []
    with open(char_list_path) as f:
        for l in f:
            charlist.append(l.strip())  # .decode('utf-8'))
    vocab_size = len(charlist)
    print("vocab_size is:", vocab_size)
    return charlist


class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32

    # char_list = load_char_list('/home/data/OCR/IAM/words/label_dictionary.txt')
    char_list = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    CHARMAP = ['', '', ''] + char_list
    print(CHARMAP)

    def __init__(self,
                 sess,
                 annotation_fn,
                 buckets,
                 epochs=1000,
                 max_width=None,
                 selected_num=10000):
        """
        :param annotation_fn:.
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.sess = sess
        self.epochs = epochs
        self.max_width = max_width

        self.bucket_specs = buckets  # (encoder_size, decoder_size)
        self.bucket_data = BucketData()

        if isinstance(annotation_fn, list):
            dataset_list = []
            for cur_annotaion_fn in annotation_fn:
                dataset_list.extend(glob.glob(os.path.join(cur_annotaion_fn, '*.tfrecords')))
        elif annotation_fn.endswith('.tfrecords'):
            dataset_list = [annotation_fn]
            print("annotation_fn.endswith('.tfrecords')")
            self.data_num = 1e4
        else:
            dataset_list = glob.glob(os.path.join(annotation_fn, '*.tfrecords'))

        dataset_list.sort()  #
        if selected_num < len(dataset_list):  # selected one of the specific tfrecords
            print("selected_num < len(dataset_list)")
            selected_id = selected_num
            selected_dataset = [os.path.join(annotation_fn, str(int(selected_id)) + '.tfrecords')]
            self.data_num = 1e4
        else:
            print("selected_num > len(dataset_list)")
            if len(dataset_list) > 1 and selected_num >= 10000:
                if int(selected_num / 10000) > len(dataset_list):
                    selected_dataset = dataset_list
                    self.data_num = len(dataset_list) * 1e4
                    print("the total number of dataset is less than 10000*", len(dataset_list))
                else:
                    selected_dataset = dataset_list[:int(selected_num / 10000)]
                    self.data_num = selected_num
                    print("selected_train_num", selected_num)
            else:
                selected_dataset = dataset_list
                self.data_num = len(dataset_list) * 1e4
                print("the total number of dataset is less than 10000")
            # random.shuffle(selected_dataset)
        print(selected_dataset)
        dataset = tf.data.TFRecordDataset(selected_dataset)

        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        print("begin self.clear")
        self.bucket_data = BucketData()
        print("end self.clear")

    def gen(self, batch_size):

        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        images, labels, comments = iterator.get_next()
        print("begining gen data--------")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # self.sess = sess
            i = 0
            while True:
                try:
                    raw_images, raw_labels, raw_comments = sess.run([images, labels, comments])
                    for img, lex, comment in zip(raw_images, raw_labels, raw_comments):
                        word = self.convert_lex(lex)
                        bucket_size = self.bucket_data.append(img, word, lex, comment)
                        if bucket_size >= batch_size:
                            bucket = self.bucket_data.flush_out(
                                self.bucket_specs,
                                go_shift=1)
                            yield bucket
                except tf.errors.OutOfRangeError:
                    break
            self.clear()

    def convert_lex(self, lex):
        if sys.version_info >= (3,):
            lex = lex.decode('iso-8859-1')

        assert len(lex) < self.bucket_specs[-1][1]
        word_label = np.array(
            [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
            dtype=np.int32)

        return word_label

    @staticmethod
    def _parse_record(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'comment': tf.FixedLenFeature([], tf.string, default_value=''),
            })
        return features['image'], features['label'], features['comment']
