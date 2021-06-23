import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# based on https://www.nature.com/articles/s41598-021-85671-y


def get_model(ct_shape=None,
              vector_shape=None,
              first_conv_channel=32,
              dropout=0.3,
              volumetric=True):
    if ct_shape is None:
        ct_shape = (None, None, None, 1) if volumetric else (None, None, 1)
    convolutional_layer = layers.Conv3D if volumetric else layers.Conv2D
    max_pool_layer = layers.MaxPool3D if volumetric else layers.MaxPool2D

    ct = layers.Input(shape=ct_shape)
    inputs = [ct]
    x = convolutional_layer(filters=first_conv_channel, kernel_size=5, input_shape=ct_shape, padding="same")(ct)
    x = layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25))(x)
    x = max_pool_layer(pool_size=4, strides=4)(x)

    x = convolutional_layer(filters=2 * first_conv_channel, kernel_size=3, padding="same")(x)
    x = layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25))(x)
    x = max_pool_layer(pool_size=4, strides=4)(x)

    x = convolutional_layer(filters=4 * first_conv_channel, kernel_size=3, padding="same")(x)
    x = layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25))(x)
    x = max_pool_layer(pool_size=4, strides=4)(x)

    x = layers.Flatten()(x)
    if vector_shape is not None:
        vector = layers.Input(shape=vector_shape)
        inputs.append(vector)
        x = tf.concat([x, vector], axis=1)
    x = layers.Dense(4 * first_conv_channel)(x)
    x = layers.Dense(8 * first_conv_channel)(x)
    x = layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25))(x)
    x = layers.Dropout(rate=dropout)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="Lombardo")
