from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model

def create_lstm_encoder(input_dim, hidden_dim, output_dim, num_lstm_layers=1, bidirectional=False, return_sequences=False, dropout_rate=0.0):
    
    inputs = Input(shape=(None, input_dim))
    
    x = inputs
    for i in range(num_lstm_layers):
        return_seq = True if i < num_lstm_layers - 1 else return_sequences
        lstm_layer = LSTM(units=hidden_dim, activation='tanh', dropout=dropout_rate, return_sequences=return_seq)
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        x = lstm_layer(x)

    outputs = Dense(output_dim, activation='relu')(x)

    model = Model(inputs, outputs, name='lstm_encoder')
    return model