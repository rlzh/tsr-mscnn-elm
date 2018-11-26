import numpy as np
import tensorflow as tf
import os
import ms_os_elm_architecture


class MS_OS_ELM(object):

    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes, dropout,
                 activation='sigmoid', loss='mean_squared_error', name=None):

        if name == None:
            self.name = 'model'
        else:
            self.name = name

        self.__sess = tf.Session()
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes
        self.__dropout = dropout

        if activation == 'sigmoid':
            self.__activation = tf.nn.sigmoid
        elif activation == 'linear' or activation == None:
            self.__activation = tf.identity
        elif activation == 'tanh':
            self.__activation = tf.tanh
        else:
            raise ValueError(
                'an unknown activation function \'%s\' was given.' % (
                    activation)
            )
        self.__loss = loss
        if loss == 'mean_squared_error':
            self.__lossfun = tf.losses.mean_squared_error
        elif loss == 'mean_absolute_error':
            self.__lossfun = tf.keras.losses.mean_absolute_error
        elif loss == 'categorical_crossentropy':
            self.__lossfun = tf.keras.losses.categorical_crossentropy
        elif loss == 'binary_crossentropy':
            self.__lossfun = tf.keras.losses.binary_crossentropy
        else:
            raise ValueError(
                'an unknown loss function \'%s\' was given. ' % loss
            )

        self.__is_finished_init_train = tf.get_variable(
            'is_finished_init_train',
            shape=[],
            dtype=bool,
            initializer=tf.constant_initializer(False),
        )
        self.__x = tf.placeholder(
            tf.float32,
            shape=(None, self.__n_input_nodes, self.__n_input_nodes, 1),
            name='x'
        )
        self.__t = tf.placeholder(
            tf.float32,
            shape=(None, self.__n_output_nodes),
            name='t'
        )

        # Multi-Scale Arch
        self.__logits, regularizers = ms_os_elm_architecture.MultiScaleCNNArch(
            self.__x,
            self.__dropout
        )
        # one_hot_y = tf.one_hot(self.__t, self.__n_output_nodes)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        #     labels=one_hot_y,
        #     logits=logits
        # )
        # self.__loss_op = tf.reduce_mean(cross_entropy) + 1e-5 * regularizers
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
        # self.__train_op = optimizer.minimize(self.__loss_op)

        self.__n_elm_input_nodes = self.__logits.shape[1]
        self.__elm_x = tf.placeholder(
            tf.float32,
            shape=(None, self.__n_elm_input_nodes),
            name='elm_x'
        )
        self.__alpha = tf.get_variable(
            'alpha',
            shape=[self.__n_elm_input_nodes, self.__n_hidden_nodes],
            initializer=tf.random_uniform_initializer(-1, 1),
            trainable=False,
        )
        self.__bias = tf.get_variable(
            'bias',
            shape=[self.__n_hidden_nodes],
            initializer=tf.random_uniform_initializer(-1, 1),
            trainable=False,
        )
        self.__beta = tf.get_variable(
            'beta',
            shape=[self.__n_hidden_nodes, self.__n_output_nodes],
            initializer=tf.zeros_initializer(),
            trainable=False,
        )
        self.__p = tf.get_variable(
            'p',
            shape=[self.__n_hidden_nodes, self.__n_hidden_nodes],
            initializer=tf.zeros_initializer(),
            trainable=False,
        )

        # Finish initial training phase
        self.__finish_init_train = tf.assign(
            self.__is_finished_init_train, True)

        # Predict
        self.__predict = tf.matmul(
            self.__activation(
                tf.matmul(self.__elm_x, self.__alpha) + self.__bias
            ),
            self.__beta)

        # Loss
        self.__loss = self.__lossfun(self.__t, self.__predict)

        # Accuracy
        self.__accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.__predict, 1), tf.argmax(self.__t, 1)), tf.float32))

        # Initial training phase
        self.__init_train = self.__build_init_train_graph()

        # Sequential training phase
        self.__seq_train = self.__build_seq_train_graph()

        # Saver
        self.__saver = tf.train.Saver()

        # Initialize variables
        self.__sess.run(tf.global_variables_initializer())

        tf.get_variable_scope().reuse_variables()

    def predict(self, x):
        l = self.__sess.run(self.__logits, feed_dict={self.__x: x})
        return self.__sess.run(self.__predict, feed_dict={self.__elm_x: l})

    def evaluate(self, x, t, metrics=['loss']):
        met = []
        for m in metrics:
            if m == 'loss':
                met.append(self.__loss)
            elif m == 'accuracy':
                met.append(self.__accuracy)
            else:
                return ValueError(
                    'an unknown metric \'%s\' was given.' % m
                )
        l = self.__sess.run(self.__logits, feed_dict={self.__x: x})
        ret = self.__sess.run(met, feed_dict={self.__elm_x: l, self.__t: t})
        return ret

    def init_train(self, x, t):
        if self.__sess.run(self.__is_finished_init_train):
            raise Exception(
                'the initial training phase has already finished. '
                'please call \'seq_train\' method for further training.'
            )
        if len(x) < self.__n_hidden_nodes:
            raise ValueError(
                'in the initial training phase, the number of training samples '
                'must be greater than the number of hidden nodes. '
                'But this time len(x) = %d, while n_hidden_nodes = %d' % (
                    len(x), self.__n_hidden_nodes)
            )
        l = self.__sess.run(
            self.__logits,
            feed_dict={self.__x: x}
        )
        self.__sess.run(
            self.__init_train,
            feed_dict={self.__elm_x: l, self.__t: t}
        )
        self.__sess.run(self.__finish_init_train)

    def seq_train(self, x, t):
        if self.__sess.run(self.__is_finished_init_train) == False:
            raise Exception(
                'you have not gone through the initial training phase yet. '
                'please first initialize the model\'s weights by \'init_train\' '
                'method before calling \'seq_train\' method.'
            )
        l = self.__sess.run(
            self.__logits,
            feed_dict={self.__x: x}
        )
        self.__sess.run(
            self.__seq_train,
            feed_dict={self.__elm_x: l, self.__t: t}
        )

    def __build_init_train_graph(self):
        H = self.__activation(
            tf.matmul(self.__elm_x, self.__alpha) + self.__bias
        )
        HT = tf.transpose(H)
        HTH = tf.matmul(HT, H)
        fo = tf.matrix_inverse(HTH)
        p = tf.assign(self.__p, fo)
        pHT = tf.matmul(p, HT)
        pHTt = tf.matmul(pHT, self.__t)
        init_train = tf.assign(self.__beta, pHTt)
        return init_train

    def __build_seq_train_graph(self):
        H = self.__activation(
            tf.matmul(self.__elm_x, self.__alpha) + self.__bias
        )
        HT = tf.transpose(H)
        HTH = tf.matmul(HT, H)
        batch_size = tf.shape(self.__elm_x)[0]
        I = tf.eye(batch_size)
        Hp = tf.matmul(H, self.__p)
        HpHT = tf.matmul(Hp, HT)
        temp = tf.matrix_inverse(I + HpHT)
        pHT = tf.matmul(self.__p, HT)
        p = tf.assign(self.__p, self.__p - tf.matmul(tf.matmul(pHT, temp), Hp))
        pHT = tf.matmul(p, HT)
        Hbeta = tf.matmul(H, self.__beta)
        seq_train = self.__beta.assign(
            self.__beta + tf.matmul(pHT, self.__t - Hbeta))
        return seq_train

    def save(self, filepath):
        self.__saver.save(self.__sess, filepath)

    def restore(self, filepath):
        self.__saver.restore(self.__sess, filepath)

    def initialize_variables(self):
        for var in [self.__alpha, self.__bias, self.__beta, self.__p, self.__is_finished_init_train]:
            self.__sess.run(var.initializer)

    def __del__(self):
        self.__sess.close()
        tf.keras.backend.clear_session()

    @property
    def input_shape(self):
        return (None, self.__n_input_nodes, self.__n_input_nodes, 1)

    @property
    def output_shape(self):
        return (self.__n_output_nodes,)

    @property
    def n_input_nodes(self):
        return self.__n_input_nodes

    @property
    def n_hidden_nodes(self):
        return self.__n_hidden_nodes

    @property
    def n_output_nodes(self):
        return self.__n_output_nodes

    @property
    def t_predict(self):
        return self.__predict
