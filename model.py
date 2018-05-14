#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @author: guohainan
# @Create: 2018-05-13
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class WebVisitPredictModel(object):

    def __init__(self, time_step=50, input_dim=1, output_dim=1):
        self.time_step = time_step
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_units = 32

        # tensorflow input
        self.x = None
        self.y = None
        self.x_index = None
        # tensorflow output
        self.loss = None
        self.train_op = None
        self.y_prediction = None

        self._build_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        with tf.variable_scope("input"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.time_step, self.input_dim], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="y")
            self.x_index = tf.placeholder(tf.float32, shape=[None, 1], name="x_index")
        with tf.variable_scope("lstm"):
            cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_units, state_is_tuple=True, name="lstm_cell")
            out, state = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
            out = tf.transpose(out, [1, 0, 2])
            lstm_out = tf.gather(out, int(out.get_shape()[0]) - 1, name="lstm_out")
        lstm_prediction = tf.layers.dense(lstm_out, self.output_dim, activation=tf.tanh, name="lstm_prediction")

        with tf.variable_scope("vocation_dense"):
            l_1 = tf.layers.dense(self.x_index, 32, activation=tf.nn.relu, name="x_index_relu")
            l_1 = tf.layers.dense(l_1, 1, activation=tf.nn.tanh, name="x_index_tanh")
            self.y_prediction = l_1 * 10 * lstm_prediction

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_prediction), axis=1))
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    ts = 50
    m = WebVisitPredictModel(input_dim=1, time_step=ts)
    x_data = np.arange(0, 10000, 0.01)
    y_sin = np.sin(x_data)
    for i in range(len(x_data) / ts / 5):
        idx = i * 50 * ts
        # 模拟节假日 请求量整体幅度变大情况
        y_sin[idx: idx + 25 * ts] = y_sin[idx: idx + 25 * ts] * 3
        idx += 25 * ts
        # 模拟业务整体低谷
        y_sin[idx: idx + 25 * ts] = y_sin[idx: idx + 25 * ts] * 0.5
    plt.ion()
    plt.show()
    for i in range(2000):
        x_batch = []
        y_batch = []
        x_index = []
        for k in range(2000):
            start_idx = i * 500 + k
            end_idx = start_idx + ts
            x = y_sin[start_idx: end_idx].reshape(ts, 1)
            x_batch.append(x)
            y_batch.append([y_sin[end_idx]])
            x_index.append([end_idx])
        _, loss, y_pre = m.sess.run([m.train_op, m.loss, m.y_prediction], feed_dict={
            m.x: np.array(x_batch),
            m.x_index: x_index,
            m.y: np.array(y_batch)
        })
        if i % 100 == 0:
            print("loss: ", loss)
        plt.cla()
        plt.plot(x_index, np.array(y_batch).reshape([-1, 1]), label="real")
        plt.plot(x_index, y_pre.reshape([-1, 1]), label="prediction")
        # plt.ylim(-3, 3)
        plt.legend()
        plt.draw()
        plt.pause(0.1)
