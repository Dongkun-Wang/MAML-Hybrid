# -*- coding: utf-8 -*-
# @File : meta_model.py
# @Time : 2021/11/6 16:52
# @Brief: 实现模型分类的网络，与网络结构无关，重点在训练过程

from tensorflow.keras import layers, models, losses
import tensorflow as tf


class MAML:
    def __init__(self, input_shape, num_classes):
        """
        MAML模型类，需要两个模型，一个是作为真实更新的权重θ，另一个是用来做θ'的更新
        :param input_shape: 模型输入shape
        :param num_classes: 分类数目
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.meta_model = self.make_meta_model()

    def make_meta_model(self):
        """
        建立meta模型
        :return: meta model
        """
        x = main_input = layers.Input(shape=self.input_shape, name='main_input')
        for _ in range(4):
            x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu", )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = layers.Flatten()(x)
        main_output = layers.Dense(self.num_classes, activation='softmax', name='main_output')(x)
        # create model
        meta_model = models.Model(inputs=main_input, outputs=main_output)
        return meta_model

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
        """
        MAML一个batch的训练过程
        :param train_data: 训练数据，以batch为一个单位
        :param inner_optimizer: support set对应的优化器
        :param inner_step: 内部更新几个step
        :param outer_optimizer: query set对应的优化器，如果对象不存在则不更新梯度
        :return: batch query loss
        """
        batch_acc = []
        batch_loss = []

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()
        batch_support_image, batch_support_label, batch_query_image, batch_query_label = next(train_data)

        with tf.GradientTape() as outer_tape:
            for support_image, support_label, query_image, query_label \
                    in zip(batch_support_image, batch_support_label, batch_query_image, batch_query_label):
                # 每个task都需要载入最原始的weights进行计算
                self.meta_model.set_weights(meta_weights)
                # inner loop with support-set
                for _ in range(inner_step):
                    with tf.GradientTape() as inner_tape:
                        inner_output = self.meta_model(support_image, training=True)
                        inner_loss = losses.sparse_categorical_crossentropy(support_label, inner_output)
                        inner_loss = tf.reduce_mean(inner_loss)
                    # inner loop optimization
                    inner_grads = inner_tape.gradient(inner_loss, self.meta_model.trainable_variables)
                    inner_optimizer.apply_gradients(zip(inner_grads, self.meta_model.trainable_variables))

                # outer loop with query-set
                outer_output = self.meta_model(query_image, training=True)
                outer_loss = losses.sparse_categorical_crossentropy(query_label, outer_output)
                outer_loss = tf.reduce_mean(outer_loss)
                batch_loss.append(outer_loss)
                acc = tf.cast(tf.argmax(outer_output, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)
            mean_batch_acc = tf.reduce_mean(batch_acc)
            mean_batch_loss = tf.reduce_mean(batch_loss, axis=0)

        # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            outer_grads = outer_tape.gradient(mean_batch_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(outer_grads, self.meta_model.trainable_variables))
        return mean_batch_loss, mean_batch_acc
