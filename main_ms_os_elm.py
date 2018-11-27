# Load pickled data
import os
import helpers
import pickle
import tensorflow as tf
import cv2
import sklearn as skl
import numpy as np
import preproc
from ms_os_elm import MS_OS_ELM
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

        t_train = to_categorical(t_train, num_classes=n_classes)
        t_valid = to_categorical(t_valid, num_classes=n_classes)
        t_test = to_categorical(t_test, num_classes=n_classes)
        t_train = t_train.astype(np.float32)
        t_valid = t_valid.astype(np.float32)
        t_test = t_test.astype(np.float32)

        N_HIDDEN_NODES_BASE = 512
        n_hidden_nodes_mult = [1, 2, 3, 4, 5]
        BATCH_SIZE = 64

        LOSS_FUNCS = [
            'mean_squared_error',  'mean_absolute_error',
            'categorical_crossentropy',  'binary_crossentropy'
        ]
        highest_accuracy = 0
        optimal = ()
        n_input_nodes = image_shape[0]
        n_output_nodes = n_classes

        for m in n_hidden_nodes_mult:

            n_hidden_nodes = N_HIDDEN_NODES_BASE * (2**m)
            border = int(1.5 * n_hidden_nodes)
            for loss_func in LOSS_FUNCS:
                os_elm = MS_OS_ELM(
                    # the number of input nodes.
                    n_input_nodes=n_input_nodes,
                    # the number of hidden nodes.
                    n_hidden_nodes=n_hidden_nodes,
                    # the number of output nodes.
                    n_output_nodes=n_output_nodes,
                    dropout=0.5,
                    # loss function.
                    loss=loss_func,
                    # activation function applied to the hidden nodes.
                    # NOTE: OS-ELM can apply an activation function only to the hidden nodes.
                    activation='linear',
                    name='%d_%s' % (n_hidden_nodes, loss_func)
                )
                print('training config - hidden nodes=%d, loss func="%s"' %
                      (n_hidden_nodes, loss_func))
                x_train_init = X_train[:border]
                x_train_seq = X_train[border:]
                t_train_init = t_train[:border]
                t_train_seq = t_train[border:]

                os_elm.init_train(x_train_init, t_train_init)
                accuracy = 0
                pbar = tqdm.tqdm(total=len(X_train))
                pbar.update(n=len(x_train_init))
                x_train_seq, t_train_seq = skl.utils.shuffle(
                    x_train_seq, t_train_seq
                )
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

                if accuracy > highest_accuracy:
                    print('best accuracy yet! saving model...')
                    highest_accuracy = accuracy
                    optimal = os_elm.name
                    SAVE_PATH = './elm_checkpoint/model.ckpt'
                    os_elm.save(SAVE_PATH)
                    # initialize weights of os_elm
                    os_elm.initialize_variables()
                    print('restoring model parameters...')
                    os_elm.restore(SAVE_PATH)
                    [accuracy] = os_elm.evaluate(
                        X_test, t_test, metrics=['accuracy'])
                    print('test_accuracy: %f' % (accuracy))

                del os_elm

        print('optimal accuracy config was ' + str(optimal))


if __name__ == '__main__':
    main()
