import os
import helpers
import sys
import pickle
import tensorflow as tf
import cv2
import sklearn as skl
import numpy as np
import preproc
from os_elm import OS_ELM
from keras.utils import to_categorical
import tqdm


def softmax(a):
    c = np.max(a, axis=-1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    return exp_a / sum_exp_a


def main():
    # Load preprocessed data
    training_file = 'traffic-signs-data/train_preproc_clahe_data.p'
    validation_file = 'traffic-signs-data/valid_preproc_clahe_data.p'
    testing_file = 'traffic-signs-data/test_preproc_clahe_data.p'
    if not os.path.isfile(training_file) or not os.path.isfile(validation_file) or not os.path.isfile(testing_file):
        print("ERROR: Run preproc.py to create ", " ",
              training_file, " ", validation_file, " ", testing_file)
    else:
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        X_train, t_train = train['features'], train['labels']

        X_valid, t_valid = valid['features'], valid['labels']

        X_test, t_test = test['features'], test['labels']

        n_train = X_train.shape[0]
        n_validation = X_valid.shape[0]
        n_test = X_test.shape[0]
        image_shape = X_train.shape[1:]
        n_classes = max(t_test) - min(t_test) + 1
        print("Number of training examples =", n_train)
        print("Number of validation examples =", n_validation)
        print("Number of testing examples =", n_test)
        print("Image data shape =", image_shape)
        print("max y_valid = ", max(t_valid))
        print("min y_valid = ", min(t_valid))
        print("Number of classes =", n_classes)

        X_train = np.reshape(
            X_train,
            (X_train.shape[0], X_train.shape[1] * X_train.shape[1])
        )
        X_valid = np.reshape(
            X_valid,
            (X_valid.shape[0], X_valid.shape[1] * X_valid.shape[1])
        )
        X_test = np.reshape(
            X_test,
            (X_test.shape[0], X_test.shape[1] * X_test.shape[1])
        )
        t_train = to_categorical(t_train, num_classes=n_classes)
        t_test = to_categorical(t_test, num_classes=n_classes)
        t_valid = to_categorical(t_valid, num_classes=n_classes)
        t_train = t_train.astype(np.float32)
        t_test = t_test.astype(np.float32)
        t_valid = t_valid.astype(np.float32)

        n_input_nodes = image_shape[0]
        n_hidden_nodes = 1024
        n_output_nodes = n_classes

        os_elm = OS_ELM(
            # the number of input nodes.
            n_input_nodes=n_input_nodes*n_input_nodes,
            # the number of hidden nodes.
            n_hidden_nodes=n_hidden_nodes,
            # the number of output nodes.
            n_output_nodes=n_output_nodes,
            # loss function.
            # the default value is 'mean_squared_error'.
            # for the other functions, we support
            # 'mean_absolute_error', 'categorical_crossentropy', and 'binary_crossentropy'.
            loss='mean_squared_error',
            # activation function applied to the hidden nodes.
            # the default value is 'sigmoid'.
            # for the other functions, we support 'linear' and 'tanh'.
            # NOTE: OS-ELM can apply an activation function only to the hidden nodes.
            activation='sigmoid',
        )

        BATCH_SIZE = 64
        border = int(1.5 * n_hidden_nodes)
        x_train_init = X_train[:border]
        x_train_seq = X_train[border:]
        t_train_init = t_train[:border]
        t_train_seq = t_train[border:]

        pbar = tqdm.tqdm(total=len(X_train), desc='initial training phase')
        os_elm.init_train(x_train_init, t_train_init)
        pbar.update(n=len(x_train_init))

        pbar.set_description('sequential training phase')
        for i in range(0, len(x_train_seq), BATCH_SIZE):
            x_batch = x_train_seq[i:i+BATCH_SIZE]
            t_batch = t_train_seq[i:i+BATCH_SIZE]
            os_elm.seq_train(x_batch, t_batch)
            pbar.update(n=len(x_batch))
        pbar.close()

        X_valid, t_valid = skl.utils.shuffle(X_valid, t_valid)
        [accuracy] = os_elm.evaluate(
            X_valid, t_valid, metrics=['accuracy'])
        print('val_accuracy: %f' % (accuracy))


if __name__ == '__main__':
    main()
