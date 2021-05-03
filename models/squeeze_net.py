import tensorflow as tf
from tensorflow.keras import layers

# based on https://arxiv.org/abs/1602.07360


def fire_block(x, channel):
    assert channel % 8 == 0
    inputs = x
    x = layers.Conv3D(filters=channel / 8, kernel_size=1, activation="relu")(x)
    x = layers.BatchNormalization(x)
    x1 = layers.Conv3D(filters=channel / 2, kernel_size=1, activation="relu")(x)
    x1 = layers.BatchNormalization(x1)
    x2 = layers.Conv3D(filters=channel / 2, kernel_size=3, padding="same", activation="relu")(x)
    x2 = layers.BatchNormalization(x2)
    x = tf.concat(x1, x2)
    # TODO add skip connection
    return x


def get_model(input_shape=(None, None, None, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(filters=96, kernel_size=1, activation="relu")(inputs)
    x = layers.MaxPool3D(strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = fire_block(x, 128)
    x = fire_block(x, 128)
    x = fire_block(x, 256)
    x = layers.MaxPool3D(strides=2)(x)
    x = fire_block(x, 256)
    x = fire_block(x, 384)
    x = fire_block(x, 384)
    x = fire_block(x, 512)
    x = layers.MaxPool3D(strides=2)(x)
    x = fire_block(x, 512)
    x = layers.Conv3D(filters=1, kernel_size=1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Softmax()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="SqueezeNet")

