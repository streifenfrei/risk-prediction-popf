from tensorflow import keras
from tensorflow.keras import layers

# based on https://arxiv.org/abs/2007.13224


def get_model(input_shape=(None, None, None, 1)):
    model = keras.Sequential(name="SimpleNet")
    model.add(layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPool3D())
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling3D())
    model.add(layers.Dense(units=512, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(units=1, activation="sigmoid"))
    return model
