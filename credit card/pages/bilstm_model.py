import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Layer, Permute, Multiply, Lambda, Softmax
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

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

def build_bilstm_model(input_shape, output_dim):
    """
    BiLSTM with Attention for temporal sequence classification.
    
    Args:
        input_shape (tuple): (sequence_length, num_features)
        output_dim (int): number of output classes (e.g., 2 for fraud/non-fraud)

    Returns:
        Keras Model
    """
    inputs = Input(shape=input_shape)

    # BiLSTM layers
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)

    # Attention mechanism
    attention_output = AttentionLayer()(x)

    # Dense layers
    x = Dense(64, activation='relu')(attention_output)
    x = Dropout(0.3)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
