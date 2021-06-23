from tensorflow import keras
from tensorflow.keras import layers


def get_model(ct_shape=None,
              vector_shape=None,
              first_conv_channel=64,
              dropout=0.3,
              volumetric=True):
    if ct_shape is None:
        ct_shape = (None, None, None, 1) if volumetric else (None, None, 1)
    convolutional_layer = layers.Conv3D if volumetric else layers.Conv2D
    max_pool_layer = layers.MaxPool3D if volumetric else layers.MaxPool2D
    global_pooling_layer = layers.GlobalAveragePooling3D if volumetric else layers.GlobalAveragePooling2D

    model = keras.Sequential(name="SimpleNet")
    model.add(convolutional_layer(filters=first_conv_channel, kernel_size=3, padding="same", activation="relu",
                            input_shape=ct_shape))
    model.add(max_pool_layer())
    model.add(layers.BatchNormalization())

    model.add(convolutional_layer(filters=first_conv_channel, kernel_size=3, padding="same", activation="relu"))
    model.add(max_pool_layer())
    model.add(layers.BatchNormalization())

    model.add(convolutional_layer(filters=2*first_conv_channel, kernel_size=3, padding="same", activation="relu"))
    model.add(max_pool_layer())
    model.add(layers.BatchNormalization())

    model.add(convolutional_layer(filters=4*first_conv_channel, kernel_size=3, padding="same", activation="relu"))
    model.add(max_pool_layer())
    model.add(layers.BatchNormalization())

    model.add(global_pooling_layer())
    model.add(layers.Dense(units=8*first_conv_channel, activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(units=1, activation="sigmoid"))
    return model
