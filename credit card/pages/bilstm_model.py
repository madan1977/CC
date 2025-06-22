import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Layer, Permute, Multiply, Lambda, Softmax,BatchNormalization
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
        # x shape: (batch_size, time_steps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, time_steps, 1)
        e = K.squeeze(e, axis=-1)              # (batch_size, time_steps)
        alpha = K.softmax(e)                   # (batch_size, time_steps)
        alpha = K.expand_dims(alpha, axis=-1)  # (batch_size, time_steps, 1)
        context = x * alpha                    # (batch_size, time_steps, features)
        context = K.sum(context, axis=1)       # (batch_size, features)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_bilstm_model(input_shape, output_dim, loss_fn='binary_crossentropy'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from pages.bilstm_model import AttentionLayer

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(AttentionLayer())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    if output_dim == 1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss=loss_fn, optimizer=Adam(0.001), metrics=['accuracy'])
    return model
