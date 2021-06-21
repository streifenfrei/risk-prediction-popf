import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def residual_block(x, filters, strides=1, regularizer=None):
    skip = layers.Conv3D(filters=filters, kernel_size=3, strides=strides, activation="relu", padding="same",
                         kernel_regularizer=regularizer)(x)
    skip = layers.BatchNormalization()(skip)

    x = layers.Conv3D(filters=filters, kernel_size=3, strides=strides, activation="relu", padding="same",
                      kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)

    x = tf.add(skip, x)
    x = keras.activations.relu(x)
    x = layers.BatchNormalization()(x)
    return x


def get_model(ct_shape=(None, None, None, 1),
              vector_shape=None,
              first_conv_channel=16,
              dropout=0.3,
              regularizer=None):
    ct = layers.Input(shape=ct_shape)
    inputs = [ct]
    x = layers.Conv3D(filters=first_conv_channel, kernel_size=3, strides=2, activation="relu",
                      input_shape=ct_shape)(ct)
    x = layers.BatchNormalization()(x)

    x = residual_block(x, first_conv_channel)
    x = residual_block(x, first_conv_channel)
    x = residual_block(x, first_conv_channel, strides=2)
    x = residual_block(x, first_conv_channel)
    x = residual_block(x, 2*first_conv_channel, strides=2)
    x = residual_block(x, 2*first_conv_channel)
    x = residual_block(x, 4*first_conv_channel)
    x = residual_block(x, 4*first_conv_channel)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(rate=dropout)(x)
    x = layers.Dense(units=1, activation="sigmoid", kernel_regularizer=regularizer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="ResNet")
