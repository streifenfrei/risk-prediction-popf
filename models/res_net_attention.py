import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.python.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.python.ops import math_ops


def conv_block(x, filters, regularizer, volumetric, split, strides=1, padding="valid", activation=None):
    if volumetric and split:
        x = layers.Conv3D(filters=filters, kernel_size=(3, 3, 1), strides=(strides, strides, 1), padding="same",
                          kernel_regularizer=regularizer)(x)
        x = layers.Conv3D(filters=filters, kernel_size=(1, 1, 3), strides=(1, 1, strides), padding=padding,
                          kernel_regularizer=regularizer, activation=activation)(x)
    else:
        conv_layer = layers.Conv3D if volumetric else layers.Conv2D
        x = conv_layer(filters=filters, kernel_size=3, strides=strides, padding=padding,
                       kernel_regularizer=regularizer, activation=activation)(x)
    return x


def residual_block_internal_splitconv(x, filters, strides=1, regularizer=None, volumetric=True, split=False):
    x = conv_block(x, filters=filters, strides=strides, activation="relu", padding="same",
                   regularizer=regularizer, volumetric=volumetric, split=split)
    x = layers.BatchNormalization()(x)
    x = conv_block(x, filters=filters, padding="same", regularizer=regularizer, volumetric=volumetric, split=split)
    return layers.BatchNormalization()(x)


def squeeze_excitation_block(x, filters, ratio, name="SE", strides=1, regularizer=None, volumetric=True, split=False):
    global_pooling_layer = layers.GlobalAveragePooling3D if volumetric else layers.GlobalAveragePooling2D
    with tf.name_scope(name):
        skip = conv_block(x, filters=filters, strides=strides, activation="relu", padding="same",
                          regularizer=regularizer,
                          volumetric=volumetric, split=split)
        skip = layers.BatchNormalization()(skip)
        residual = residual_block_internal_splitconv(x, filters=filters, strides=strides,
                                                     regularizer=regularizer, volumetric=volumetric, split=split)
        x = global_pooling_layer()(residual)
        x = layers.Dense(units=filters / ratio, kernel_regularizer=regularizer)(x)
        x = layers.ReLU()(x)
        x = layers.Dense(units=filters, activation="sigmoid", kernel_regularizer=regularizer)(x)
        shape = [-1, 1, 1, 1, filters] if volumetric else [-1, 1, 1, filters]
        x = tf.reshape(x, shape)
        x = residual * x
        x = tf.add(skip, x)
    return x


def get_model(ct_shape=None,
              vector_shape=None,
              first_conv_channel=16,
              dropout=0.3,
              ratio=2,
              split=True,
              dense_regularizer=None,
              conv_regularizer=None,
              volumetric=True,
              alpha=3,
              sigmoid_alpha=100,
              sigmoid_beta=0.5,
              p_class_weight=0.5):
    if ct_shape is None:
        ct_shape = (None, None, None, 1) if volumetric else (None, None, 1)
    convolutional_layer = layers.Conv3D if volumetric else layers.Conv2D
    global_pooling_layer = layers.GlobalAveragePooling3D if volumetric else layers.GlobalAveragePooling2D
    ct = layers.Input(shape=ct_shape)
    inputs = [ct]
    x = convolutional_layer(filters=first_conv_channel, kernel_size=3, strides=2, activation="relu",
                            input_shape=ct_shape, kernel_regularizer=conv_regularizer)(ct)
    x = layers.BatchNormalization()(x)

    x = squeeze_excitation_block(x, first_conv_channel, ratio, regularizer=conv_regularizer, volumetric=volumetric,
                                 split=split)
    x = squeeze_excitation_block(x, first_conv_channel, ratio, regularizer=conv_regularizer, volumetric=volumetric,
                                 split=split)
    x = squeeze_excitation_block(x, first_conv_channel, ratio, strides=2, regularizer=conv_regularizer,
                                 volumetric=volumetric, split=split)
    x = squeeze_excitation_block(x, first_conv_channel, ratio, regularizer=conv_regularizer, volumetric=volumetric)
    features = squeeze_excitation_block(x, first_conv_channel, ratio, regularizer=conv_regularizer,
                                        volumetric=volumetric, split=split, name="attention_map")
    prediction = global_pooling_layer()(features)
    prediction = layers.Dropout(rate=dropout)(prediction)
    prediction_layer = layers.Dense(units=2, activation="softmax",
                                    kernel_regularizer=dense_regularizer, name="label")
    prediction = prediction_layer(prediction)

    total_loss_metric = tf.keras.metrics.Mean(name="loss")
    label_loss_metric = tf.keras.metrics.Mean(name="label_loss")
    attention_loss_metric = tf.keras.metrics.Mean(name="attention_loss")
    auc_metric = tf.keras.metrics.AUC(name="auc")

    class AttentionResnet(tf.keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_weight(shape=prediction_layer.weights[0].shape, name="attention_conv", trainable=False)

        @property
        def metrics(self):
            return [total_loss_metric, label_loss_metric, attention_loss_metric, auc_metric]

        # see https://arxiv.org/abs/2005.02690
        def _create_attention_map(self, feats):
            attention_weights = self.non_trainable_weights[-1].assign(prediction_layer.weights[0])
            weights_shape = [1, 1, 1, attention_weights.shape[0], 2] if volumetric else \
                [1, 1, attention_weights.shape[0], 2]
            attention_weights = tf.reshape(attention_weights, weights_shape)
            attention = tf.nn.conv3d(feats, attention_weights, strides=[1, 1, 1, 1, 1], padding="SAME") if \
                volumetric else tf.nn.conv2d(feats, attention_weights, strides=1, padding="SAME")
            attention = tf.nn.relu(attention)
            # tf does not seem to have builtin bilinear upsampling for 3D data (only nearest neighbour)
            attention = UpSampling3D(size=[4, 4, 4])(attention) if \
                volumetric else UpSampling2D(size=[4, 4])
            positive_map, negative_map = tf.split(attention, num_or_size_splits=2, axis=-1)
            p_max = tf.reduce_max(positive_map)
            p_min = tf.reduce_min(positive_map)
            positive_map = (positive_map - p_min) / (p_max - p_min + 10e-7)
            n_max = tf.reduce_max(negative_map)
            n_min = tf.reduce_min(negative_map)
            negative_map = (negative_map - n_min) / (n_max - n_min + 10e-7)
            attention = tf.concat([positive_map, negative_map], axis=-1)
            attention = tf.keras.backend.sigmoid(attention)
            return tf.sigmoid(sigmoid_alpha * (attention - sigmoid_beta))

        ALPHA = 500

        def calc_loss(self, _x, tlabel, seg):
            label, feats = self(_x, training=True)
            attention_map = self._create_attention_map(feats)
            label = tf.squeeze(label)
            label_loss = tfa.losses.SigmoidFocalCrossEntropy()(label, tlabel)
            attention_map = tf.where(seg == 1., 0., attention_map)
            attention_loss = alpha * tf.keras.backend.mean(attention_map, axis=range(1, 5))
            return label, label_loss, attention_loss

        def train_step(self, data):
            _x, (tlabel, seg) = data
            seg = tf.concat([seg, seg], axis=-1)
            weight = p_class_weight * tlabel[:, 0] + (1 - p_class_weight) * tlabel[:, 1]
            with tf.GradientTape() as tape:
                label, label_loss, attention_loss = self.calc_loss(_x, tlabel, seg)
                total_loss = weight * (label_loss + attention_loss)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            total_loss_metric.update_state(total_loss)
            label_loss_metric.update_state(label_loss)
            attention_loss_metric.update_state(attention_loss)
            auc_metric.update_state(tlabel, label)
            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            _x, (tlabel, seg) = data
            seg = tf.concat([seg, seg], axis=-1)
            label, label_loss, attention_loss = self.calc_loss(_x, tlabel, seg)
            weight = p_class_weight * tlabel[:, 0] + (1 - p_class_weight) * tlabel[:, 1]
            total_loss = weight * (label_loss + attention_loss)
            total_loss_metric.update_state(total_loss)
            label_loss_metric.update_state(label_loss)
            attention_loss_metric.update_state(attention_loss)
            auc_metric.update_state(tlabel, label)
            return {m.name: m.result() for m in self.metrics}

    return AttentionResnet(inputs=inputs, outputs=[prediction, features], name="AttentionResNet")
