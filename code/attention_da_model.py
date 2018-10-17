"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division

import logging
import math
import os
import sys
import time

import distance
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

import losses
from cnn import cnn_encoder
from data_gen import DataGen
from densenet import DenseNet
from seq2seq_model import Seq2SeqModel
from utils import set_indices, keep_indices
from visualizations import visualize_attention


class Model(object):
    def __init__(self,
                 sess,
                 parameters,
                 phase,
                 visualize,
                 output_dir,
                 batch_size,
                 initial_learning_rate,
                 steps_per_checkpoint,
                 model_dir,
                 target_embedding_size,
                 attn_num_hidden,
                 attn_num_layers,
                 clip_gradients,
                 max_gradient_norm,
                 session,
                 load_model,
                 gpu_id,
                 use_gru,
                 use_distance=False,
                 max_image_width=160,
                 max_image_height=60,
                 max_prediction_length=15,
                 channels=1,
                 reg_val=0):

        self.sess = sess

        self.parameters = parameters
        self.use_distance = use_distance

        # We need resized width, not the actual width
        self.max_original_width = max_image_width
        self.max_width = int(
            math.ceil(1. * max_image_width / max_image_height * DataGen.IMAGE_HEIGHT))

        self.encoder_size = int(math.ceil(1. * self.max_width / 4))
        self.decoder_size = max_prediction_length + 2
        self.buckets = [(self.encoder_size, self.decoder_size)]

        if gpu_id >= 0:
            device_id = '/gpu:' + str(gpu_id)
        else:
            device_id = '/cpu:0'
        self.device_id = device_id

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if phase == 'train':
            self.forward_only = False
        else:
            self.forward_only = True
            batch_size = 1

        logging.info('phase: %s' % phase)
        logging.info('model_dir: %s' % (model_dir))
        logging.info('load_model: %s' % (load_model))
        logging.info('output_dir: %s' % (output_dir))
        logging.info('steps_per_checkpoint: %d' % (steps_per_checkpoint))
        logging.info('batch_size: %d' % (batch_size))
        logging.info('learning_rate: %d' % initial_learning_rate)
        logging.info('reg_val: %d' % (reg_val))
        logging.info('max_gradient_norm: %f' % max_gradient_norm)
        logging.info('clip_gradients: %s' % clip_gradients)
        logging.info('max_image_width %f' % max_image_width)
        logging.info('max_prediction_length %f' % max_prediction_length)
        logging.info('channels: %d' % (channels))
        logging.info('target_embedding_size: %f' % target_embedding_size)
        logging.info('attn_num_hidden: %d' % attn_num_hidden)
        logging.info('attn_num_layers: %d' % attn_num_layers)
        logging.info('visualize: %s' % visualize)

        if use_gru:
            logging.info('using GRU in the decoder.')

        self.reg_val = reg_val
        self.sess = session
        self.steps_per_checkpoint = steps_per_checkpoint
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.global_step = tf.Variable(0, trainable=False)

        self.phase = phase
        self.visualize = visualize
        self.load_model = load_model
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients
        self.max_gradient_norm = max_gradient_norm
        self.channels = channels

        self.target_embedding_size = target_embedding_size
        self.attn_num_layers = attn_num_layers
        self.attn_num_hidden = attn_num_hidden
        self.use_gru = use_gru

        self.checkpoint_path_a = os.path.join(self.model_dir,
                                              parameters.dataset_a)  # transferred weights
        self.checkpoint_path_b = os.path.join(self.model_dir, parameters.dataset_b)

        self.summaries = []

        # with tf.device(self.device_id):
        self.height = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.int32)
        self.height_float = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float64)

        ####################   INPUT ##################
        ##--------------------- source data definition -------------------------#
        self.img_pl_a = tf.placeholder(tf.string, name='input_image_as_bytes_a')
        self.img_data_a = tf.cond(
            tf.less(tf.rank(self.img_pl_a), 1),
            lambda: tf.expand_dims(self.img_pl_a, 0),
            lambda: self.img_pl_a
        )

        self.img_pl_b = tf.placeholder(tf.string, name='input_image_as_bytes_b')
        self.img_data_b = tf.cond(
            tf.less(tf.rank(self.img_pl_b), 1),
            lambda: tf.expand_dims(self.img_pl_b, 0),
            lambda: self.img_pl_b
        )

        self.img_data_a = tf.map_fn(self._prepare_image, self.img_data_a, dtype=tf.float32)
        self.img_data_b = tf.map_fn(self._prepare_image, self.img_data_b, dtype=tf.float32)
        ##--------------------- source data definition -------------------------#
        num_images_a = tf.shape(self.img_data_a)[0]
        # TODO: create a mask depending on the image/batch size
        self.encoder_masks_a = []
        self.decoder_inputs_a = []
        self.target_weights_a = []
        # self.encoder_masks  [tf.ones([num_images, 1])] * (self.encoder_size + 1) ?
        # self.decoder_inputs  [(self.decoder_size+1) * tf.zeros([num_images, ])]  ?
        # self.tager_weights   [self.decoder_size * tf.ones([num_images,])]  + [0]
        for i in xrange(self.encoder_size + 1):
            self.encoder_masks_a.append(
                tf.tile([[1.]], [num_images_a, 1])
            )
        for i in xrange(self.decoder_size + 1):
            self.decoder_inputs_a.append(
                tf.tile([0], [num_images_a])
            )
            if i < self.decoder_size:
                self.target_weights_a.append(tf.tile([1.], [num_images_a]))
            else:
                self.target_weights_a.append(tf.tile([0.], [num_images_a]))

        # --------------------- target data definition -------------------- #
        num_images_b = tf.shape(self.img_data_b)[0]
        self.encoder_masks_b = []
        self.decoder_inputs_b = []
        self.target_weights_b = []
        for i in xrange(self.encoder_size + 1):
            self.encoder_masks_b.append(
                tf.tile([[1.]], [num_images_b, 1])
            )
        for i in xrange(self.decoder_size + 1):
            self.decoder_inputs_b.append(
                tf.tile([0], [num_images_b])
            )
            if i < self.decoder_size:
                self.target_weights_b.append(tf.tile([1.], [num_images_b]))
            else:
                self.target_weights_b.append(tf.tile([0.], [num_images_b]))

        self.criteria_function = losses.correlation_loss

        self.loss_op, self.trainable_variables = self._bulid_da_model(self.img_data_a,
                                                                      self.img_data_b,
                                                                      self.target_weights_a,
                                                                      self.target_weights_b,
                                                                      self.encoder_masks_a,
                                                                      self.encoder_masks_b,
                                                                      self.decoder_inputs_a,
                                                                      self.decoder_inputs_b)

        if not self.forward_only:
            self._optimize(self.trainable_variables)
        self.saver_all = tf.train.Saver(tf.all_variables())
        self._initialize_model(self.parameters)

        # self._initialize_model(parameters)

    def _bulid_da_model(self, img_data_a, img_data_b,
                        target_weights_a, target_weights_b,
                        encoder_masks_a, encoder_masks_b,
                        decoder_inputs_a, decoder_inputs_b,
                        name_scope="towers"):
        """
        build attention sequence domain adaptation model
        """
        # with tf.device(self.device_id):
        self.attention_decoder_model_a, self.conv_output_a = self.build_attention_model(
            img_data_a,
            target_weights_a,
            encoder_masks_a,
            decoder_inputs_a,
            forward_only=self.forward_only,
            reuse=False,
            name_scope=name_scope)

        self.attention_decoder_model_b, self.conv_output_b = self.build_attention_model(
            img_data_b,
            target_weights_b,
            encoder_masks_b,
            decoder_inputs_b,
            forward_only=True,
            reuse=True,
            name_scope=name_scope)

        self.prediction, self.probability = self.decoder(self.attention_decoder_model_b.output)
        # self.similarity_losses = self.sequence_content_loss()
        # self.similarity_losses = self.instance_char_losses()
        print("Starting batch char losses")
        self.similarity_losses = self.instance_char_losses_with_confidence(prob_threshold=0.5)


        if self.reg_val > 0:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            logging.info('Adding %s regularization losses', len(reg_losses))
            logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
            loss_op = self.reg_val * tf.reduce_sum(
                reg_losses) + self.attention_decoder_model_a.loss
        else:
            loss_op = self.attention_decoder_model_a.loss

        loss_op = loss_op + self.similarity_losses
        trainable_variables = tf.trainable_variables()
        return loss_op, trainable_variables

    def instance_char_losses(self):
        """
        compare if source and target char are same
        """
        attention_contents_a = self.attention_decoder_model_a.attention_contents
        attention_contents_b = self.attention_decoder_model_b.attention_contents
        similarity_losses = 0.0
        for cur_time in range(len(self.attention_decoder_model_a.output)):
            cur_prediction_a = tf.argmax(self.attention_decoder_model_a.output[cur_time],
                                         axis=1)  # (batch_size, 1)
            # print("cur_prediction_a", cur_prediction_a.shape)
            cur_prediction_b = tf.argmax(self.attention_decoder_model_b.output[cur_time],
                                         axis=1)  # (batch_size, 1)
            # print("cur_prediction_b", cur_prediction_b.shape)
            cur_a_is_equal_b = tf.equal(cur_prediction_a, cur_prediction_b)  # (batch_size,)
            # print(cur_a_is_equal_b.shape)

            indices = []

            # 1 + 1 = 2  ==> and

            indices = tf.map_fn(lambda idx:
                                tf.cond(tf.equal(
                                    tf.add(tf.cast(cur_a_is_equal_b[idx], tf.uint8),
                                           tf.cast(tf.not_equal(cur_prediction_a[idx], 2),
                                                   tf.uint8)), 2),
                                    lambda: set_indices(idx, indices),
                                    lambda: keep_indices(idx, indices)),
                                tf.range(tf.shape(cur_a_is_equal_b)[0]))

            indices = tf.convert_to_tensor(indices)
            valid_attention_contents_a = tf.nn.embedding_lookup(attention_contents_a[cur_time],
                                                                indices)
            valid_attention_contents_b = tf.nn.embedding_lookup(attention_contents_b[cur_time],
                                                                indices)

            similarity_losses = tf.cond(
                tf.greater_equal(tf.shape(valid_attention_contents_a)[0], 1),
                lambda: tf.add(similarity_losses,
                               self.criteria_function(valid_attention_contents_a,
                                                      valid_attention_contents_b, 0.1)),
                lambda: tf.add(similarity_losses, 0.0))
        return similarity_losses

    def instance_char_losses_with_confidence(self, prob_threshold):
        """
        with confidence
        """
        attention_contents_a = self.attention_decoder_model_a.attention_contents
        attention_contents_b = self.attention_decoder_model_b.attention_contents
        similarity_losses = 0.0
        for cur_time in range(len(self.attention_decoder_model_a.output)):
            cur_prediction_prob_a = tf.reduce_max(self.attention_decoder_model_a.output[cur_time],
                                                  axis=1)  # (batch_size, 1)
            cur_prediction_a = tf.argmax(self.attention_decoder_model_a.output[cur_time],
                                         axis=1)  # (batch_size, )

            print("cur_prediction_a", cur_prediction_a.shape)
            cur_prediction_prob_b = tf.reduce_max(self.attention_decoder_model_b.output[cur_time],
                                                  axis=1)  # (batch_size, 1)
            cur_prediction_b = tf.argmax(self.attention_decoder_model_b.output[cur_time],
                                         axis=1)  # (batch_size, )
            print("cur_prediction_b", cur_prediction_b.shape)
            is_valid_char_a = tf.greater_equal(cur_prediction_prob_a,
                                               prob_threshold)  # (batch_size,)
            is_valid_char_b = tf.greater_equal(cur_prediction_prob_b,
                                               prob_threshold)  # (batch_size,)
            print(is_valid_char_a.shape)

            valid_char_indices_a = []
            valid_char_indices_b = []
            # 1 + 1 = 2  ==> and
            valid_char_indices_a = tf.map_fn(lambda idx:
                                             tf.cond(tf.equal(
                                                 tf.add(tf.cast(is_valid_char_a[idx], tf.uint8),
                                                        tf.cast(
                                                            tf.not_equal(cur_prediction_a[idx], 2),
                                                            tf.uint8)), 2),
                                                 lambda: set_indices(idx, valid_char_indices_a),
                                                 lambda: keep_indices(idx, valid_char_indices_a)),
                                             tf.range(tf.shape(is_valid_char_a)[0]))

            valid_char_indices_b = tf.map_fn(lambda idx:
                                             tf.cond(tf.equal(
                                                 tf.add(tf.cast(is_valid_char_b[idx], tf.uint8),
                                                        tf.cast(
                                                            tf.not_equal(cur_prediction_b[idx], 2),
                                                            tf.uint8)), 2),
                                                 lambda: set_indices(idx, valid_char_indices_b),
                                                 lambda: keep_indices(idx, valid_char_indices_b)),
                                             tf.range(tf.shape(is_valid_char_b)[0]))

            valid_char_indices_a = tf.convert_to_tensor(valid_char_indices_a)
            valid_char_indices_b = tf.convert_to_tensor(valid_char_indices_b)
            valid_attention_contents_a = tf.nn.embedding_lookup(attention_contents_a[cur_time],
                                                                valid_char_indices_a)
            valid_attention_contents_b = tf.nn.embedding_lookup(attention_contents_b[cur_time],
                                                                valid_char_indices_b)

            similarity_losses = tf.cond(
                tf.greater_equal(tf.shape(valid_attention_contents_a)[0], 1),
                lambda: tf.add(similarity_losses,
                               self.criteria_function(valid_attention_contents_a,
                                                      valid_attention_contents_b, 0.1)),
                lambda: tf.add(similarity_losses, 0.0))
        return similarity_losses

    def sequence_content_loss(self):
        """
        with confidence
        """
        attention_contents_a = self.attention_decoder_model_a.attention_contents
        attention_contents_b = self.attention_decoder_model_b.attention_contents
        # print(attention_contents_a)
        similarity_losses = 0.0
        for cur_time in range(len(attention_contents_a)):
            # print(attention_contents_a[cur_time][0].shape)
            # print(attention_contents_b[cur_time][0].shape)
            similarity_losses = tf.add(similarity_losses,
                                       self.criteria_function(attention_contents_a[cur_time][0],
                                                              attention_contents_b[cur_time][0],
                                                              0.1))
        return similarity_losses

    def sequence_attention_convout_loss(self):
        """
        attention_weights * convout
        """
        attention_weight_a = self.attention_decoder_model_a.attention_weights_history
        attention_weight_b = self.attention_decoder_model_b.attention_weights_history
        similarity_losses = 0.0
        for cur_attention_weight_a, cur_attention_weight_b in zip(attention_weight_a,
                                                                  attention_weight_b):
            attention_out_a = tf.reduce_sum(
                tf.expand_dims(cur_attention_weight_a, axis=-1) * self.conv_output_a, axis=[1])
            # print(attention_out_a.shape)
            attention_out_b = tf.reduce_sum(
                tf.expand_dims(cur_attention_weight_b, axis=-1) * self.conv_output_b, axis=[1])
            similarity_losses = tf.add(similarity_losses,
                                       self.criteria_function(attention_out_a, attention_out_b,
                                                              0.01))

        return similarity_losses

    def cnn_encoder_similarity_loss(self):
        """
        the cnn out
        """
        contents_a = tf.reduce_sum(self.conv_output_a, axis=[1])
        contents_b = tf.reduce_sum(self.conv_output_b, axis=[1])
        print(contents_a.shape)
        print(contents_b.shape)
        # print(attention_contents_a)
        similarity_losses = 0.0
        similarity_losses = self.criteria_function(contents_a,
                                                   contents_b,
                                                   0.1)
        return similarity_losses

    # def batch_char_losses(self):
    #     """
    #     """
    #
    #     def set_indices(idx, indices):
    #         indices.append(idx)
    #         return indices
    #
    #     def keep_indices(idx, indices):
    #         return indices
    #
    #     # 1 + 1 = 2  ==> and
    #
    #     # indices = tf.map_fn(lambda idx:
    #     #                     tf.cond(tf.equal(
    #     #                         tf.add(tf.cast(cur_a_is_equal_b[idx], tf.uint8),
    #     #                                tf.cast(tf.not_equal(cur_prediction_a[idx], 2),
    #     #                                        tf.uint8)), 2),
    #     #                         lambda: set_indices(idx, indices),
    #     #                         lambda: keep_indices(idx, indices)),
    #     #                     tf.range(tf.shape(cur_a_is_equal_b)[0]))
    #     attention_contents_a = self.attention_decoder_model_a.attention_contents
    #     attention_contents_b = self.attention_decoder_model_b.attention_contents
    #     similarity_losses = 0.0
    #     chars_features_repo_a = []
    #     chars_features_repo_b = []
    #
    #     # chars_features_repo_a = dict().fromkeys(range(len(DataGen.CHARMAP)))
    #     # chars_features_repo_b = dict().fromkeys(range(len(DataGen.CHARMAP)))
    #     for cur_key in range(len(DataGen.CHARMAP)):
    #         chars_features_repo_a.append([])
    #         chars_features_repo_b.append([])
    #     print(attention_contents_a)
    #     def list_append(old_list, new_ele):
    #         old_list.append(new_ele)
    #         return old_list
    #
    #     def compute_batch_indices(batch_size, beam_size):
    #         """Computes the i'th coordinate that contains the batch index for gathers.
    #         Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    #         batch the beam item is in. This will create the i of the i,j coordinate
    #         needed for the gather.
    #         Args:
    #           batch_size: Batch size
    #           beam_size: Size of the beam.
    #         Returns:
    #           batch_pos: [batch_size, beam_size] tensor of ids
    #         """
    #         batch_pos = tf.range(batch_size * beam_size) // beam_size
    #         batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    #         return batch_pos
    #
    #
    #     for cur_time in range(len(self.attention_decoder_model_a.output)):
    #         cur_time_predictions_a = tf.cast(
    #             tf.argmax(self.attention_decoder_model_a.output[cur_time],
    #                       axis=1), tf.int32)
    #         cur_time_predictions_b = tf.cast(
    #             tf.argmax(self.attention_decoder_model_b.output[cur_time],
    #                       axis=1), tf.int32)
    #         for idx in range(self.batch_size):
    #             print(cur_time_predictions_a[idx])
    #             print(cur_time_predictions_b[idx])
    #             print(attention_contents_a[idx])
    #             cur_char_feature_repo = tf.gather(chars_features_repo_a, cur_time_predictions_a[idx])
    #             cur_char_feature_repo
    #             print(tf.gather(chars_features_repo_a, cur_time_predictions_a[idx]))
    #             print(tf.gather(chars_features_repo_a, cur_time_predictions_a[idx]).append(attention_contents_a[idx]))
    #             print(chars_features_repo_a[cur_time_predictions_a[idx]])
    #             chars_features_repo_a[idx].append(cur_time_predictions_a[idx])
    #             chars_features_repo_b[idx].append(cur_time_predictions_a[idx])
    #
    #         print(cur_time_predictions_b.shape)
    #         print(self.batch_size)
    #         # a = tf.map_fn(
    #         #     lambda idx: list_append([],
    #         #                             attention_contents_b[idx]),
    #         #     range(self.batch_size))

    def decoder(self, model_output):
        # ---------------Decoder-----------------#
        table = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value="",
            checkpoint=True)

        insert = table.insert(
            tf.constant(list(range(len(DataGen.CHARMAP))), dtype=tf.int64),
            tf.constant(DataGen.CHARMAP))

        with tf.control_dependencies([insert]):
            num_feed = []
            prb_feed = []

            for l in xrange(len(model_output)):
                guess = tf.argmax(model_output[l], axis=1)
                proba = tf.reduce_max(
                    tf.nn.softmax(model_output[l]), axis=1)
                num_feed.append(guess)
                prb_feed.append(proba)

            # Join the predictions into a single output string.
            trans_output = tf.transpose(num_feed)
            trans_output = tf.map_fn(
                lambda m: tf.foldr(
                    lambda a, x: tf.cond(
                        tf.equal(x, DataGen.EOS_ID),
                        lambda: '',
                        lambda: table.lookup(x) + a),
                    m, initializer=''),
                trans_output,
                dtype=tf.string
            )

            # Calculate the total probability of the output string.
            trans_outprb = tf.transpose(prb_feed)
            trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))
            trans_outprb = tf.map_fn(
                lambda m: tf.foldr(
                    lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                    m,
                    initializer=tf.cast(1, tf.float64)
                ),
                trans_outprb,
                dtype=tf.float64
            )

            prediction = tf.cond(
                tf.equal(tf.shape(trans_output)[0], 1),
                lambda: trans_output[0],
                lambda: trans_output,
            )
            probability = tf.cond(
                tf.equal(tf.shape(trans_outprb)[0], 1),
                lambda: trans_outprb[0],
                lambda: trans_outprb,
            )

            prediction = tf.identity(prediction, name='prediction')
            probability = tf.identity(probability, name='probability')
            return prediction, probability

    def build_attention_model(self,
                              img_data,
                              target_weights, encoder_masks, decoder_inputs,
                              forward_only=True,
                              reuse=False, name_scope="tower"):
        with tf.variable_scope(name_scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            # conv_output = cnn_encoder(img_data, not forward_only, name="cnn_encoder", reuse=reuse)
            # with tf.variable_scope("cnn_encoder"):
            conv_output = DenseNet(img_input=img_data,
                                   dense_blocks=3,
                                   dense_layers=[6, 12, 16],
                                   growth_rate=24,
                                   dropout_rate=None,
                                   bottleneck=True,
                                   compression=0.5,
                                   depth=32,
                                   is_training=not self.forward_only)
            perm_conv_output = tf.transpose(conv_output,
                                            perm=[1, 0, 2])  # Time_width, batch_size, channel

            attention_decoder_model = Seq2SeqModel(
                encoder_inputs_tensor=perm_conv_output,
                encoder_masks=encoder_masks,
                decoder_inputs=decoder_inputs,
                target_weights=target_weights,
                target_vocab_size=len(DataGen.CHARMAP),
                buckets=self.buckets,
                target_embedding_size=self.target_embedding_size,
                attn_num_layers=self.attn_num_layers,
                attn_num_hidden=self.attn_num_hidden,
                forward_only=forward_only,
                use_gru=self.use_gru)
            # predictions, probability = self.decoder(attention_decoder_model.output)
            return attention_decoder_model, conv_output

    def _optimize(self, params):
        self.updates = []
        self.summaries_by_bucket = []
        # params = tf.trainable_variables()
        for var in params:
            print(var.name)

        # params_cnn_encoder = [var for var in params if 'cnn_encoder' in var.name]
        params_cnn_encoder = [var for var in params if 'dense' in var.name]
        params_cnn_encoder_trainable = params_cnn_encoder  # finetune all convolutional layer
        for var in params_cnn_encoder_trainable:
            print(var.name)
        # params_cnn_encoder_trainable = params
        opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

        gradients, params_cnn_encoder_trainable = zip(
            *opt.compute_gradients(self.loss_op, params_cnn_encoder_trainable))
        if self.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Add summaries for loss, variables, gradients, gradient norms and total gradient norm.
        self.summaries.append(tf.summary.scalar("loss", self.loss_op))
        self.summaries.append(tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients)))
        all_summaries = tf.summary.merge(self.summaries)
        self.summaries_by_bucket.append(all_summaries)
        # update op - apply gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.updates.append(opt.apply_gradients(zip(gradients, params_cnn_encoder_trainable),
                                                    global_step=self.global_step))

    def train(self, data_path_a, data_path_b, num_epoch, train_num):
        logging.info('num_epoch: %d' % num_epoch)
        s_gen_a = DataGen(self.sess, data_path_a, self.buckets,
                          selected_num=train_num,
                          epochs=num_epoch + 1,
                          max_width=self.max_original_width)
        s_gen_b = DataGen(self.sess, data_path_b, self.buckets,
                          selected_num=train_num,
                          epochs=num_epoch, max_width=self.max_original_width)

        step_time = 0.0
        loss = 0.0
        current_step = 0
        writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

        logging.info('Starting the training process.')
        best_acc = 0.0
        if s_gen_a.data_num > s_gen_b.data_num:
            for batch_a, batch_b in zip(s_gen_a.gen(self.batch_size), s_gen_b.gen(self.batch_size)):
                current_step += 1
                start_time = time.time()
                result = self.step_da(batch_a, batch_b, self.forward_only)
                loss += result['loss'] / self.steps_per_checkpoint
                curr_step_time = (time.time() - start_time)
                step_time += curr_step_time / self.steps_per_checkpoint

                writer.add_summary(result['summaries'], current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % self.steps_per_checkpoint == 0:
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    # Print statistics for the previous epoch.
                    logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                                 % (self.sess.run(self.global_step), step_time, loss, perplexity))
                    word_acc, char_acc = self.test(data_path_b, print_info=False)
                    if char_acc > best_acc:
                        best_acc = char_acc
                        # Save checkpoint and reset timer and loss.
                        logging.info("Saving the model at step %d." % current_step)
                        logging.info("word_acc:{},char_acc:{}".format(word_acc, char_acc))
                        self.save(self.checkpoint_path_b, self.global_step)
                    step_time, loss = 0.0, 0.0
        else:
            for batch_b, batch_a in zip(s_gen_b.gen(self.batch_size),
                                        s_gen_a.gen(self.batch_size)):
                current_step += 1
                start_time = time.time()
                result = self.step_da(batch_a, batch_b, self.forward_only)
                loss += result['loss'] / self.steps_per_checkpoint
                curr_step_time = (time.time() - start_time)
                step_time += curr_step_time / self.steps_per_checkpoint

                writer.add_summary(result['summaries'], current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % self.steps_per_checkpoint == 0:
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    # Print statistics for the previous epoch.
                    logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                                 % (
                                     self.sess.run(self.global_step), step_time, loss, perplexity))
                    word_acc, char_acc = self.test(data_path_b, print_info=False)
                    if char_acc > best_acc:
                        best_acc = char_acc
                        # Save checkpoint and reset timer and loss.
                        logging.info("Saving the model at step %d." % current_step)
                        logging.info("word_acc:{},char_acc:{}".format(word_acc, char_acc))
                        self.save(self.checkpoint_path_b, self.global_step)
                    step_time, loss = 0.0, 0.0

        print("finish training")
        print("loss")

        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                     % (self.sess.run(self.global_step), step_time, loss, perplexity))

        word_acc, char_acc = self.test(data_path_b, print_info=False)
        if char_acc > best_acc:
            # Save checkpoint and reset timer and loss.
            logging.info("Saving the model at step %d." % current_step)
            logging.info("word_acc:{},char_acc:{}".format(word_acc, char_acc))
            best_acc = char_acc
            self.save(self.checkpoint_path_b, self.global_step)
        print("word_acc:", word_acc, "char_acc:", char_acc)

    def step_da(self, batch_a, batch_b, forward_only):
        img_data_a = batch_a['data']
        decoder_inputs_a = batch_a['decoder_inputs']
        target_weights_a = batch_a['target_weights']

        img_data_b = batch_b['data']
        decoder_inputs_b = batch_b['decoder_inputs']
        target_weights_b = batch_b['target_weights']

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_pl_a.name] = img_data_a
        input_feed[self.img_pl_b.name] = img_data_b

        for l in xrange(self.decoder_size):
            input_feed[self.decoder_inputs_a[l].name] = decoder_inputs_a[l]
            input_feed[self.target_weights_a[l].name] = target_weights_a[l]
            input_feed[self.decoder_inputs_b[l].name] = decoder_inputs_b[l]
            input_feed[self.target_weights_b[l].name] = target_weights_b[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target_a = self.decoder_inputs_a[self.decoder_size].name
        input_feed[last_target_a] = np.zeros([self.batch_size], dtype=np.int32)

        last_target_b = self.decoder_inputs_b[self.decoder_size].name
        input_feed[last_target_b] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [
            self.attention_decoder_model_a.loss,  # Loss for this batch.
        ]

        if not forward_only:
            output_feed += [self.summaries_by_bucket[0],
                            self.updates[0]]
        else:
            output_feed += [self.prediction]
            output_feed += [self.probability]
            if self.visualize:
                output_feed += self.attention_decoder_model_a.attentions

        outputs = self.sess.run(output_feed, input_feed)

        res = {
            'loss': outputs[0],
        }

        if not forward_only:
            res['summaries'] = outputs[1]
        else:
            res['prediction'] = outputs[1]
            res['probability'] = outputs[2]
            if self.visualize:
                res['attentions'] = outputs[3:]

        return res

    def save(self, checkpoint_dir, step):
        model_name = "model.ckpt"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # Save checkpoint and reset timer and loss.
        # logging.info("Saving the model at step %d." % step)
        self.saver_all.save(self.sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
            logging.info("Loading model successfully")
            return True
        else:
            logging.info("Failed to loading model from ".format(checkpoint_dir))
            return False

    def _initialize_model(self, parameters):
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        if self.load_model:
            if self.phase == 'test' and parameters.is_da:
                print("parameters.is_da", parameters.is_da)
                self.load(self.checkpoint_path_b)
            else:
                print("parameters.is_da", parameters.is_da)
                self.load(self.checkpoint_path_a)
        else:
            logging.info("Created model with fresh parameters.")

    def predict(self, image_file_data):
        input_feed = {}
        input_feed[self.img_pl_a.name] = image_file_data

        output_feed = [self.prediction, self.probability]

        # [prediction, probability] = self.decoder(image_file_data)
        # output_feed = [self.prediction, self.probability]
        outputs = self.sess.run(output_feed, input_feed)

        text = outputs[0]
        probability = outputs[1]
        if sys.version_info >= (3,):
            text = text.decode('iso-8859-1')

        return (text, probability)

    def test(self, data_path, selected_num=1e4, print_info=True, error_file="error_file.txt"):
        current_step = 0
        num_correct_character = 0.0
        num_correct_word = 0.0
        num_total = 0.0
        word_acc = 0.0
        char_acc = 0.0

        error_file_fp = open(error_file, "w")
        s_gen_test = DataGen(self.sess, data_path, self.buckets, epochs=1,
                             selected_num=selected_num,
                             max_width=self.max_original_width)
        print("step test begins")
        for cur_test_batch in s_gen_test.gen(1):
            # print("step test begins")
            current_step += 1
            # Get a batch (one image) and make a step.
            start_time = time.time()
            result_test = self.step_test(cur_test_batch, True)
            curr_step_time = (time.time() - start_time)
            # print("step test ends,test time is {}".format(curr_step_time))
            num_total += 1

            output_test = result_test['prediction']
            ground_test = cur_test_batch['labels'][0]
            comment = cur_test_batch['comments'][0]
            if sys.version_info >= (3,):
                output_test = output_test.decode('iso-8859-1')
                ground_test = ground_test.decode('iso-8859-1')
                comment = comment.decode('iso-8859-1')

            probability = result_test['probability']

            if self.use_distance:
                incorrect_character = distance.levenshtein(output_test, ground_test)
                if len(ground_test) == 0:
                    if len(output_test) == 0:
                        incorrect_character = 0
                    else:
                        incorrect_character = 1
                else:
                    incorrect_character = float(incorrect_character) / len(ground_test)
                incorrect_character = min(1, incorrect_character)
            else:
                incorrect_character = 0 if output_test == ground_test else 1

            correct_word = 1 if output_test == ground_test else 0
            num_correct_word += correct_word

            num_correct_character += 1. - incorrect_character
            # print(" num_correct_character", num_correct_character)

            if self.visualize:
                # Attention visualization.
                threshold = 0.5
                normalize = True
                binarize = True
                attns = np.array([[a.tolist() for a in step_attn] for step_attn in
                                  result_test['attentions']]).transpose(
                    [1, 0, 2])
                visualize_attention(cur_test_batch['data'],
                                    'out',
                                    attns,
                                    output_test,
                                    self.max_width,
                                    DataGen.IMAGE_HEIGHT,
                                    threshold=threshold,
                                    normalize=normalize,
                                    binarize=binarize,
                                    ground=ground_test,
                                    flag=None)

            step_accuracy = "{:>4.0%}".format(1. - incorrect_character)
            correctness = step_accuracy + (
                " ({} vs {}) {}".format(output_test, ground_test,
                                        comment) if incorrect_character else " (" + ground_test + ")")
            if incorrect_character:
                error_file_fp.write(comment + "\t" + ground_test + "\t" + output_test + "\n")
            word_acc = num_correct_word / num_total
            char_acc = num_correct_character / num_total
            if print_info:
                logging.info(
                    'Step {:.0f} ({:.3f}s).Word_Acc: {:6.2%} Char_Acc: {:6.2%}, loss: {:f}, perplexity: {:0<7.6}, probability: {:6.2%} {}'.format(
                        current_step,
                        curr_step_time,
                        word_acc,
                        char_acc,
                        result_test['loss'],
                        math.exp(result_test['loss']) if result_test['loss'] < 300 else float(
                            'inf'),
                        probability,
                        correctness))
        error_file_fp.close()
        return word_acc, char_acc

    # step, read one batch, generate gradients
    def step_test(self, batch, forward_only=True):
        img_data = batch['data']
        decoder_inputs = batch['decoder_inputs']
        target_weights = batch['target_weights']

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_pl_b.name] = img_data

        for l in xrange(self.decoder_size):
            input_feed[self.decoder_inputs_b[l].name] = decoder_inputs[l]
            input_feed[self.target_weights_b[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs_b[self.decoder_size].name
        input_feed[last_target] = np.zeros([self.test_batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [
            self.attention_decoder_model_b.loss,  # Loss for this batch.
        ]

        output_feed += [self.prediction]
        output_feed += [self.probability]
        if self.visualize:
            output_feed += self.attention_decoder_model_b.attentions

        outputs = self.sess.run(output_feed, input_feed)

        res = {'loss': outputs[0], }

        res['prediction'] = outputs[1]
        res['probability'] = outputs[2]
        if self.visualize:
            res['attentions'] = outputs[3:]

        return res

    def _prepare_image(self, image):
        """
        Resize the image to a maximum height of `self.height` and maximum
        width of `self.width` while maintaining the aspect ratio. Pad the
        resized image to a fixed size of ``[self.height, self.width]``.
        """
        img = tf.image.decode_png(image, channels=self.channels)
        dims = tf.shape(img)
        self.width = self.max_width

        max_width = tf.to_int32(tf.ceil(tf.truediv(dims[1], dims[0]) * self.height_float))
        max_height = tf.to_int32(tf.ceil(tf.truediv(self.width, max_width) * self.height_float))

        resized = tf.cond(
            tf.greater_equal(self.width, max_width),
            lambda: tf.cond(
                tf.less_equal(dims[0], self.height),
                lambda: tf.to_float(img),
                lambda: tf.image.resize_images(img, [self.height, max_width],
                                               method=tf.image.ResizeMethod.BICUBIC),
            ),
            lambda: tf.image.resize_images(img, [max_height, self.width],
                                           method=tf.image.ResizeMethod.BICUBIC)
        )

        padded = tf.image.pad_to_bounding_box(resized, 0, 0, self.height, self.width)
        return padded
