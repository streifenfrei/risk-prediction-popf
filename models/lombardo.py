from tensorflow import keras
from tensorflow.keras import layers

# based on https://www.nature.com/articles/s41598-021-85671-y


def get_model(input_shape=(None, None, None, 1), first_conv_channel=32, dropout=0.3):
    model = keras.Sequential(name="Lombardo")

    model.add(layers.Conv3D(filters=first_conv_channel, kernel_size=5, input_shape=input_shape, padding="same"))
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.MaxPooling3D(pool_size=4, strides=4))

    model.add(layers.Conv3D(filters=2*first_conv_channel, kernel_size=3, padding="same"))
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.MaxPooling3D(pool_size=4, strides=4))

    model.add(layers.Conv3D(filters=4*first_conv_channel, kernel_size=3, padding="same"))
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.MaxPooling3D(pool_size=4, strides=4))

    model.add(layers.Flatten())
    model.add(layers.Dense(4*first_conv_channel))
    model.add(layers.Dense(8*first_conv_channel))
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dropout(rate=dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

