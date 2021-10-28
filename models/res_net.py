import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def residual_block(x, filters, strides=1, regularizer=None, volumetric=True, name=None):
    convolutional_layer = layers.Conv3D if volumetric else layers.Conv2D

    skip = convolutional_layer(filters=filters, kernel_size=3, strides=strides, activation="relu", padding="same",
                               kernel_regularizer=regularizer)(x)
    skip = layers.BatchNormalization()(skip)
    x = residual_block_internal(x, filters, strides, regularizer, volumetric)
    x = tf.add(skip, x)
    x = keras.activations.relu(x)
    x = layers.BatchNormalization(name=name)(x)
    return x


def residual_block_internal(x, filters, strides=1, regularizer=None, volumetric=True):
    convolutional_layer = layers.Conv3D if volumetric else layers.Conv2D
    x = convolutional_layer(filters=filters, kernel_size=3, strides=strides, activation="relu", padding="same",
                            kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = convolutional_layer(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizer)(x)
    return layers.BatchNormalization()(x)


def get_model(ct_shape=None,
              vector_shape=None,
              first_conv_channel=16,
              dropout=0.3,
              dense_regularizer=None,
              conv_regularizer=None,
              volumetric=True):
    if ct_shape is None:
        ct_shape = (None, None, None, 1) if volumetric else (None, None, 1)
    convolutional_layer = layers.Conv3D if volumetric else layers.Conv2D
    global_pooling_layer = layers.GlobalAveragePooling3D if volumetric else layers.GlobalAveragePooling2D
    ct = layers.Input(shape=ct_shape)
    inputs = [ct]
    x = convolutional_layer(filters=first_conv_channel, kernel_size=3, strides=2, activation="relu",
                            input_shape=ct_shape, kernel_regularizer=conv_regularizer)(ct)
    x = layers.BatchNormalization()(x)

    x = residual_block(x, first_conv_channel, regularizer=conv_regularizer, volumetric=volumetric)
    x = residual_block(x, first_conv_channel, regularizer=conv_regularizer, volumetric=volumetric)
    x = residual_block(x, first_conv_channel, strides=2, regularizer=conv_regularizer, volumetric=volumetric)
    x = residual_block(x, first_conv_channel, regularizer=conv_regularizer, volumetric=volumetric)
    x = residual_block(x, 2 * first_conv_channel, strides=2, regularizer=conv_regularizer, volumetric=volumetric)
    x = residual_block(x, 2 * first_conv_channel, regularizer=conv_regularizer, volumetric=volumetric)
    x = residual_block(x, 4 * first_conv_channel, regularizer=conv_regularizer, volumetric=volumetric)
    features = residual_block(x, 4 * first_conv_channel, regularizer=conv_regularizer, volumetric=volumetric, name="features")

    x = global_pooling_layer()(features)
    x = layers.Dropout(rate=dropout)(x)
    x = layers.Dense(units=2, activation="sigmoid", kernel_regularizer=dense_regularizer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="ResNet")
