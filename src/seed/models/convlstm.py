"""Paper ConvLSTM architectures with 1x1 output heads."""

from __future__ import annotations


def _keras():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as error:
        raise ImportError("install the 'ml' extra to build ConvLSTM models") from error
    return tf, keras


def build_sequence_to_map(shape, filters=(64, 64, 64), kernel_size=(7, 7)):
    """Build stacked sequence-to-map ConvLSTM."""
    _, keras = _keras()
    layers = keras.layers
    inputs = layers.Input(shape=shape, name="input_sequence")
    x = inputs
    for index, channels in enumerate(filters):
        x = layers.ConvLSTM2D(
            channels,
            kernel_size,
            padding="same",
            activation="tanh",
            return_sequences=True,
            name=f"convlstm_{index + 1}",
        )(x)
    x = layers.Lambda(lambda value: value[:, -1], name="take_last")(x)
    outputs = layers.Conv2D(1, 1, padding="same", activation="linear", dtype="float32", name="pred")(x)
    return keras.Model(inputs, outputs, name="SequenceToMapConvLSTM")


def build_encoder_decoder(
    shape,
    horizon,
    encoder_filters=(64, 64),
    decoder_filters=(64, 64),
    kernel_size=(7, 7),
):
    """Build encoder-decoder ConvLSTM endpoint model."""
    tf, keras = _keras()
    layers = keras.layers
    if decoder_filters[0] != encoder_filters[-1]:
        raise ValueError("first decoder width must equal final encoder width")
    inputs = layers.Input(shape=shape, name="input_sequence")
    x = inputs
    state = None
    for index, channels in enumerate(encoder_filters):
        last = index == len(encoder_filters) - 1
        x, hidden, cell = layers.ConvLSTM2D(
            channels,
            kernel_size,
            padding="same",
            activation="tanh",
            return_sequences=not last,
            return_state=True,
            name=f"enc_convlstm_{index + 1}",
        )(x)
        state = [hidden, cell]
    y = layers.Lambda(
        lambda value: tf.tile(tf.expand_dims(value, 1), [1, horizon, 1, 1, 1]),
        name="tile_to_horizon",
    )(x)
    for index, channels in enumerate(decoder_filters):
        decoder = layers.ConvLSTM2D(
            channels,
            kernel_size,
            padding="same",
            activation="tanh",
            return_sequences=True,
            name=f"dec_convlstm_{index + 1}",
        )
        y = decoder(y, initial_state=state) if index == 0 else decoder(y)
    y = layers.Lambda(lambda value: value[:, -1], name="take_last")(y)
    outputs = layers.Conv2D(1, 1, padding="same", activation="linear", dtype="float32", name="pred")(y)
    return keras.Model(inputs, outputs, name="EncoderDecoderConvLSTM")


def build_autoregressive(shape, horizon, filters=(64, 64), kernel_size=(7, 7)):
    """Build autoregressive ConvLSTM endpoint model."""
    _, keras = _keras()
    layers = keras.layers
    height, width = shape[1], shape[2]
    inputs = layers.Input(shape=shape, name="input_sequence")
    x = inputs
    state = None
    for index, channels in enumerate(filters):
        last = index == len(filters) - 1
        x, hidden, cell = layers.ConvLSTM2D(
            channels,
            kernel_size,
            padding="same",
            activation="tanh",
            return_sequences=not last,
            return_state=True,
            name=f"enc_convlstm_{index + 1}",
        )(x)
        state = [hidden, cell]
    current = layers.Conv2D(1, 1, padding="same", activation="linear", name="init_pred")(x)
    cell = layers.ConvLSTM2D(
        filters[-1],
        kernel_size,
        padding="same",
        activation="tanh",
        return_state=True,
        name="ar_dec_cell",
    )
    output_head = layers.Conv2D(1, 1, padding="same", activation="linear", dtype="float32", name="pred")
    for step in range(horizon):
        sequence = layers.Reshape((1, height, width, 1), name=f"ar_expand_{step + 1}")(current)
        decoded, hidden, memory = cell(sequence, initial_state=state)
        state = [hidden, memory]
        current = output_head(decoded)
    return keras.Model(inputs, current, name="AutoregressiveConvLSTM")
