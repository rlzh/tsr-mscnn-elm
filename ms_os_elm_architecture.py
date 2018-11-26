import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2


def MultiScaleCNNArch(x, dropout):
    """
    My first attempt to implement multiscale CNNs by using the paper below.
    See "Traffic Sign Recognition with MultiScale Convolutional Neural Networks" by Sermanet, 2011.
    """
    mu = 0
    sigma = 0.1

    feature_size1 = 64
    feature_size2 = 32
    feature_size3 = 16
    # **** Layer 1 ****
    # Convolutional. Input = 32x32x1. Output = 28x28xfeature_size1.
    conv1_W = tf.Variable(
        tf.truncated_normal(
            shape=(5, 5, 1, feature_size1),
            mean=mu,
            stddev=sigma
        )
    )
    conv1_b = tf.Variable(tf.zeros(feature_size1))
    conv1 = tf.nn.conv2d(
        x,
        conv1_W,
        strides=[1, 1, 1, 1],
        padding='VALID'
    )
    conv1 += conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28xfeature_size1. Output = 14x14xfeature_size1.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # **** Layer 2 ****
    # Convolutional. Input = 14x14xfeature_size1. Output = 10x10xfeature_size2.
    conv2_W = tf.Variable(
        tf.truncated_normal(
            shape=(5, 5, feature_size1, feature_size2),
            mean=mu,
            stddev=sigma
        )
    )
    conv2_b = tf.Variable(tf.zeros(feature_size2))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[
                         1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10xfeature_size2. Output = 5x5xfeature_size2.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # **** Layer 3 ****
    # From Layer 2: Input = 5x5xfeature_size2. Output = 3x3xfeature_size3
    conv32_W = tf.Variable(tf.truncated_normal(
        shape=(3, 3, feature_size2, feature_size3)))
    conv32_b = tf.Variable(tf.zeros(feature_size3))
    conv32 = tf.nn.conv2d(conv2, conv32_W, strides=[
                          1, 1, 1, 1], padding='VALID') + conv32_b
    # Activation. Output = 1x288
    conv32_active = tf.nn.relu(conv32)
    # Flattening
    conv32_active_flat = tf.contrib.layers.flatten(conv32_active)
    # From Layer 2: Input = 5x5x32. Output = 800
    conv2_flat = tf.contrib.layers.flatten(conv2)
    # From Layer 1: Input = 14x14x32. Output = 1x6272.
    conv1_flat = tf.contrib.layers.flatten(conv1)
    # Combine from Layer 1 and from Layer 2. Output = 1x7360
    concat = tf.concat([conv32_active_flat, conv2_flat, conv1_flat], 1)
    regularizers = (
        tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) +
        tf.nn.l2_loss(conv32_W)
    )
    return concat, regularizers


def MultiScaleCNNArchV2(x, dropout):
    """
    See "Traffic Sign Recognition with MultiScale Convolutional Neural Networks" by Sermanet, 2011.
    See also https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad.
    I re-implemented an architecture similar to the ones described by Yadav and Sermanet.
    """
    mu = 0
    sigma = 0.05

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    layer1 = tf.nn.conv2d(x, conv1_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv1_b
    layer1 = tf.nn.relu(layer1)  # activation
    # Layer 2: Convolutional. Input = 32x32x32. Output = 32x32x32.
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 32, 32), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    layer2 = tf.nn.conv2d(layer1, conv2_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv2_b
    layer2 = tf.nn.relu(layer2)  # activation
    # Layer 3: Max Pooling. Input = 32x32x32. Output = 16x16x32.
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # Layer 4: Dropout: Input = 16x16x32. Output = 16x16x32.
    layer4 = tf.nn.dropout(layer3, dropout)

    # Layer 5: Convolutional. Input = 16x16x32. Output = 16x16x64.
    conv5_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    conv5_b = tf.Variable(tf.zeros(64))
    layer5 = tf.nn.conv2d(layer4, conv5_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv5_b
    layer5 = tf.nn.relu(layer5)
    # Layer 6: Convolutional. Input = 16x16x64. Output = 16x16x64.
    conv6_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 64, 64), mean=mu, stddev=sigma))
    conv6_b = tf.Variable(tf.zeros(64))
    layer6 = tf.nn.conv2d(layer5, conv6_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv6_b
    layer6 = tf.nn.relu(layer6)
    # Layer 7: Max Pooling. Input = 16x16x64. Output = 8x8x64.
    layer7 = tf.nn.max_pool(layer6, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # Layer 8: Dropout. Input = 8x8x64. Output = 8x8x64.
    layer8 = tf.nn.dropout(layer7, dropout)

    # Layer 9: Convolutional. Input = 8x8x64. Output = 8x8x128.
    conv9_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 64, 128), mean=mu, stddev=sigma))
    conv9_b = tf.Variable(tf.zeros(128))
    layer9 = tf.nn.conv2d(layer8, conv9_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv9_b
    layer9 = tf.nn.relu(layer9)
    # Layer 10: Convolutional. Input = 8x8x128. Output = 8x8x128.
    conv10_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 128, 128), mean=mu, stddev=sigma))
    conv10_b = tf.Variable(tf.zeros(128))
    layer10 = tf.nn.conv2d(layer9, conv10_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv10_b
    layer10 = tf.nn.relu(layer10)
    # Layer 11: Max Pooling: Input = 8x8x128. Output = 4x4x128.
    layer11 = tf.nn.max_pool(layer10, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # Layer 12: Dropout. Input = 4x4x128. Output = 4x4x128.
    layer12 = tf.nn.dropout(layer11, dropout)

    # Layer 13: Combining layers 4, 8, and 12 into one flattened layer.
    # Input = 16x16x32, 8x8x64, 4x4x128. Output = 1x14336
    flat_layer4 = tf.contrib.layers.flatten(layer4)
    flat_layer8 = tf.contrib.layers.flatten(layer8)
    flat_layer12 = tf.contrib.layers.flatten(layer12)
    layer13 = tf.concat([flat_layer4, flat_layer8, flat_layer12], 1)
    regularizers = (
        tf.nn.l2_loss(conv1_W)
        + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(conv5_W)
        + tf.nn.l2_loss(conv6_W) + tf.nn.l2_loss(conv9_W)
        + tf.nn.l2_loss(conv10_W)
    )
    return layer13, regularizers


def MultiScaleCNNArchV2Small(x, dropout):
    """
    Smaller version of above architecture. Half of the conv layers to train faster!
    """
    mu = 0
    sigma = 0.05

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    layer1 = tf.nn.conv2d(x, conv1_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv1_b
    layer1 = tf.nn.relu(layer1)  # activation
    # Layer 3: Max Pooling. Input = 32x32x32. Output = 16x16x32.
    layer3 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # Layer 4: Dropout: Input = 16x16x32. Output = 16x16x32.
    layer4 = tf.nn.dropout(layer3, dropout)

    # Layer 5: Convolutional. Input = 16x16x32. Output = 16x16x64.
    conv5_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    conv5_b = tf.Variable(tf.zeros(64))
    layer5 = tf.nn.conv2d(layer4, conv5_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv5_b
    layer5 = tf.nn.relu(layer5)
    # Layer 7: Max Pooling. Input = 16x16x64. Output = 8x8x64.
    layer7 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # Layer 8: Dropout. Input = 8x8x64. Output = 8x8x64.
    layer8 = tf.nn.dropout(layer7, dropout)

    # Layer 9: Convolutional. Input = 8x8x64. Output = 8x8x128.
    conv9_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 64, 128), mean=mu, stddev=sigma))
    conv9_b = tf.Variable(tf.zeros(128))
    layer9 = tf.nn.conv2d(layer8, conv9_W, strides=[
        1, 1, 1, 1], padding='SAME') + conv9_b
    layer9 = tf.nn.relu(layer9)
    # Layer 11: Max Pooling: Input = 8x8x128. Output = 4x4x128.
    layer11 = tf.nn.max_pool(layer9, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    # Layer 12: Dropout. Input = 4x4x128. Output = 4x4x128.
    layer12 = tf.nn.dropout(layer11, dropout)

    # Layer 13: Combining layers 4, 8, and 12 into one flattened layer.
    # Input = 16x16x32, 8x8x64, 4x4x128. Output = 1x14336
    flat_layer4 = tf.contrib.layers.flatten(layer4)
    flat_layer8 = tf.contrib.layers.flatten(layer8)
    flat_layer12 = tf.contrib.layers.flatten(layer12)
    layer13 = tf.concat([flat_layer4, flat_layer8, flat_layer12], 1)

    regularizers = (
        tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv5_W) +
        tf.nn.l2_loss(conv9_W)
    )

    return layer13, regularizers
