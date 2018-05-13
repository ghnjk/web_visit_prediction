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
        self.lstm_hidden_units = 24

        # tensorflow input
        self.x = None
        self.y = None
        self.x_index = None
        # tensorflow output
        self.loss = None
        self.train_op = None
        self.train_dense_only = None
        self.y_prediction = None
        self.y_prediction_2 = None
        self.update_net = None

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
        lstm_prediction = tf.layers.dense(lstm_out, 2, activation=tf.tanh, name="lstm_prediction")
        stop_lstm_p = tf.stop_gradient(lstm_prediction)

        self.y_prediction = tf.layers.dense(lstm_prediction, self.output_dim, name="prediction")

        # with tf.variable_scope("dense_1"):
        #     l_1 = tf.layers.dense(self.x_index, 10, activation=tf.nn.relu, name="x_index_relu")
        #     l_1 = tf.layers.dense(l_1, 2, activation=tf.nn.tanh, name="x_index_tanh")
        #     self.y_prediction = tf.concat((lstm_prediction, l_1), axis=1)
        #     self.y_prediction = tf.layers.dense(self.y_prediction, self.output_dim, name="prediction")
        with tf.variable_scope("dense_2"):
            l_1 = tf.layers.dense(self.x_index, 10, activation=tf.nn.relu, name="x_index_relu")
            l_1 = tf.layers.dense(l_1, 2, activation=tf.nn.tanh, name="x_index_tanh")
            self.y_prediction_2= tf.concat((stop_lstm_p, l_1), axis=1)
            self.y_prediction_2 = tf.layers.dense(self.y_prediction_2, self.output_dim, name="prediction")

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_prediction), axis=1))
            self.loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_prediction_2), axis=1))
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
            self.train_dense_only = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(self.loss_2)
        # var_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dense_1")
        # var_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dense_2")
        # with tf.variable_scope("update"):
        #     self.update_net = [
        #         tf.assign(dst, src) for dst, src in zip(var_2, var_1)
        #     ]

if __name__ == '__main__':
    time_step = 20
    m = WebVisitPredictModel(input_dim=1, time_step=time_step)
    x_data = np.arange(0, 10000, 0.01)
    y_sin = np.sin(x_data)
    for i in range(len(x_data) / time_step / 5):
        idx = i * 50 * time_step
        y_sin[idx: idx + 25 * time_step] = y_sin[idx: idx + 25 * time_step] * 3
    plt.ion()
    plt.show()
    for i in range(500):
        x_batch = []
        y_batch = []
        x_index = []
        for k in range(2000):
            start_idx = i * 500 + k
            end_idx = start_idx + time_step
            x = y_sin[start_idx: end_idx].reshape(time_step, 1)
            tmp_2 = np.arange(start_idx, end_idx).reshape(time_step, 1)
            # x = np.concatenate((tmp_1, tmp_2), axis=1)
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
        plt.plot(x_index, np.array(y_batch).reshape([-1, 1]))
        plt.plot(x_index, y_pre.reshape([-1, 1]))
        plt.ylim(-3, 3)
        # plt.xlim(0, 6000)
        plt.draw()
        plt.pause(0.1)
    # m.sess.run(m.update_net)
    print("update network")
    plt.pause(1)
    for i in range(1000):
        x_batch = []
        y_batch = []
        x_index = []
        for k in range(2000):
            start_idx = i * 500 + k
            end_idx = start_idx + time_step
            x = y_sin[start_idx: end_idx].reshape(time_step, 1)
            tmp_2 = np.arange(start_idx, end_idx).reshape(time_step, 1)
            # x = np.concatenate((tmp_1, tmp_2), axis=1)
            x_batch.append(x)
            y_batch.append([y_sin[end_idx]])
            x_index.append([end_idx])
        _, loss, y_pre = m.sess.run([m.train_dense_only, m.loss_2, m.y_prediction_2], feed_dict={
            m.x: np.array(x_batch),
            m.x_index: x_index,
            m.y: np.array(y_batch)
        })
        if i % 100 == 0:
            print("loss: ", loss)
        plt.cla()
        plt.plot(x_index, np.array(y_batch).reshape([-1, 1]))
        plt.plot(x_index, y_pre.reshape([-1, 1]))
        plt.ylim(-3, 3)
        # plt.xlim(0, 6000)
        plt.draw()
        plt.pause(0.1)