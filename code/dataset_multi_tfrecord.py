import tensorflow as tf
import logging

from six import b
import os


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(max_samples_per_tfrecord,
             data_home, dataset,
             annotations_path,
             output_path,
             log_step=5000,
             force_uppercase=False,
             save_filename=False):
    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    # max_samples_per_tfrecord = parameters.max_samples_per_tfrecord



    longest_label = ''
    count = 0
    with open(annotations_path, 'r') as f:
        lines = f.readlines()
        if len(lines) > max_samples_per_tfrecord:
            num_tfrecords = int(len(lines)/max_samples_per_tfrecord) + 1

        cur_tfrecord = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(output_path, str(cur_tfrecord) + '.tfrecords'))

        for idx, line in enumerate(lines):
            line = line.rstrip('\n')
            try:
                (img_path, label) = line.split(None, 1)
            except ValueError:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue

            if idx >= max_samples_per_tfrecord * (cur_tfrecord + 1):
                cur_tfrecord += 1
                writer.close()
                writer = tf.python_io.TFRecordWriter(os.path.join(output_path, str(cur_tfrecord) + '.tfrecords'))

            # writer = tf.python_io.TFRecordWriter(output_path)

            img_path = os.path.join(data_home, dataset, img_path)
            try:
                # To check if image is in correct format
                # with tf.Graph().as_default():
                #     image_contents = tf.read_file(img_path)
                #     image = tf.image.decode_jpeg(image_contents)
                #     init_op = tf.initialize_all_tables()
                #     with tf.Session() as sess:
                #         sess.run(init_op)
                #         tmp = sess.run(image)
                #         print(idx, img_path, tmp.shape)

                with open(img_path, 'rb') as img_file:
                    img = img_file.read()

                    if force_uppercase:
                        label = label.upper()

                    if len(label) > len(longest_label):
                        longest_label = label

                    feature = {}
                    feature['image'] = _bytes_feature(img)
                    feature['label'] = _bytes_feature(b(label))
                    if save_filename:
                        feature['comment'] = _bytes_feature(b(img_path))

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    writer.write(example.SerializeToString())

                    if idx % log_step == 0:
                        logging.info('Processed %s pairs.', idx+1)
            except:
                count += 1
                print(count, idx, img_path)
        logging.info('Dataset is ready: %i pairs.', idx - count + 1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)
        print("bad case number:",count)


    writer.close()
