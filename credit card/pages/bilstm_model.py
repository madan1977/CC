import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Layer, Permute, Multiply, Lambda, Softmax, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential

class AttentionLayer(Layer):
    """
    Custom attention layer for temporal sequence data
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, time_steps, 1)
        e = K.squeeze(e, axis=-1)              # (batch_size, time_steps)
        alpha = K.softmax(e)                   # (batch_size, time_steps)
        alpha = K.expand_dims(alpha, axis=-1)  # (batch_size, time_steps, 1)
        context = x * alpha                    # (batch_size, time_steps, features)
        context = K.sum(context, axis=1)       # (batch_size, features)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_bilstm_model(
    input_shape, output_dim, loss_fn='binary_crossentropy',
    lstm_units_1=128, lstm_units_2=64, dense_units=128,
    dropout_1=0.4, dropout_2=0.3, dropout_dense=0.3, learning_rate=0.0005, optimizer_name="adam"
):
    from tensorflow.keras.optimizers import Adam, RMSprop

    if optimizer_name.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_units_1, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_1))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(lstm_units_2, return_sequences=True)))
    model.add(Dropout(dropout_2))
    model.add(AttentionLayer())

    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_dense))

    if output_dim == 1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    return model
