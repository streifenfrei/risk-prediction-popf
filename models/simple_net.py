from tensorflow import keras
from tensorflow.keras import layers


def get_model(input_shape=(None, None, None, 1), first_conv_channel=64, dropout=0.3):
    model = keras.Sequential(name="SimpleNet")
    model.add(layers.Conv3D(filters=first_conv_channel, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=first_conv_channel, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=2*first_conv_channel, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=4*first_conv_channel, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling3D())
    model.add(layers.Dense(units=8*first_conv_channel, activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(units=1, activation="sigmoid"))
    return model
