"""Fixed neural-network architectures used in the wind-only evaluation.

This module is deliberately limited to the five architectures evaluated in
the accompanying manuscript.  It has no data-loading or experimental logic.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GRU,
    Input,
    Layer,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


LOOK_BACK = 48
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TARGET_COL = "ActivePower_kW"


def build_lstm(shape, seed):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=shape),
        LSTM(128, return_sequences=True), Dropout(0.05),
        LSTM(64), Dropout(0.05), Dense(1),
    ])
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return model


def build_gru(shape, seed):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=shape),
        GRU(128, return_sequences=True), Dropout(0.05),
        GRU(64), Dropout(0.05), Dense(1),
    ])
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return model


def build_mlp(dim, seed):
    tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(dim,)),
        Dense(128, activation="relu"), Dropout(0.05),
        Dense(64, activation="relu"), Dropout(0.05), Dense(1),
    ])
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return model


class SeriesDecomposition(Layer):
    def __init__(self, kernel_size=25, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def call(self, x):
        x_pad = tf.concat([
            tf.repeat(x[:, :1, :], self.pad, axis=1), x,
            tf.repeat(x[:, -1:, :], self.pad, axis=1),
        ], axis=1)
        trend = tf.nn.avg_pool1d(x_pad, ksize=self.kernel_size, strides=1, padding="VALID")
        return x - trend, trend


def build_dlinear(shape, seed):
    tf.random.set_seed(seed)
    inputs = Input(shape=shape)
    seasonal, trend = SeriesDecomposition(kernel_size=25)(inputs)
    seasonal_out = Dense(1, use_bias=True)(Flatten()(seasonal))
    trend_out = Dense(1, use_bias=True)(Flatten()(trend))
    model = Model(inputs, Add()([seasonal_out, trend_out]))
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return model


class PatchChannelEmbed(Layer):
    def __init__(self, patch_len=16, stride=8, d_model=64, n_channels=8, **kwargs):
        super().__init__(**kwargs)
        self.patch_len, self.stride = patch_len, stride
        self.d_model, self.n_channels = d_model, n_channels
        self.proj = Dense(d_model)

    def call(self, x):
        batch = tf.shape(x)[0]
        steps = tf.shape(x)[1]
        channels = self.n_channels
        transposed = tf.reshape(tf.transpose(x, [0, 2, 1]), [batch * channels, steps, 1])
        patches = tf.image.extract_patches(
            images=tf.expand_dims(transposed, 1),
            sizes=[1, 1, self.patch_len, 1], strides=[1, 1, self.stride, 1],
            rates=[1, 1, 1, 1], padding="VALID",
        )
        return self.proj(tf.squeeze(patches, 1))


class RecombineChannels(Layer):
    def __init__(self, n_channels, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_channels, self.d_model = n_channels, d_model

    def call(self, x):
        batch = tf.shape(x)[0] // self.n_channels
        return tf.reshape(x, [batch, self.n_channels * self.d_model])


def transformer_block(x, d_model, n_heads, ff_dim):
    attention = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attention]))
    feed_forward = Dense(d_model)(Dense(ff_dim, activation="relu")(x))
    return LayerNormalization(epsilon=1e-6)(Add()([x, feed_forward]))


def build_patchtst(shape, seed, patch_len=16, stride=8, d_model=64, n_heads=4,
                   ff_dim=128, n_layers=2):
    tf.random.set_seed(seed)
    _, n_features = shape
    inputs = Input(shape=shape)
    x = PatchChannelEmbed(patch_len, stride, d_model, n_channels=n_features)(inputs)
    for _ in range(n_layers):
        x = transformer_block(x, d_model, n_heads, ff_dim)
    x = GlobalAveragePooling1D()(x)
    x = RecombineChannels(n_features, d_model)(x)
    outputs = Dense(1)(Dense(64, activation="relu")(x))
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return model


ARCHITECTURES = {
    "MLP": build_mlp,
    "LSTM": build_lstm,
    "GRU": build_gru,
    "DLinear": build_dlinear,
    "PatchTST": build_patchtst,
}
