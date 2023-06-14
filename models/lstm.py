from tensorflow.keras.layers import Layer, LSTM, Bidirectional


class CustomLSTMLayer(Layer):
    def __init__(self, hidden_dim, bidirectional=False, dropout_rate=0.0, return_sequences=True, **kwargs):
        self.bidirectional = bidirectional
        self.lstm_layer = LSTM(units=hidden_dim, activation='tanh', dropout=dropout_rate, return_sequences=return_sequences)
        self.bidirectional_lstm_layer = Bidirectional(self.lstm_layer)
        super(CustomLSTMLayer, self).__init__(**kwargs)

    def call(self, inputs):
        if self.bidirectional:
            return self.bidirectional_lstm_layer(inputs)
        else:
            return self.lstm_layer(inputs)
        
    def get_config(self):
        config = super(CustomLSTMLayer, self).get_config()
        config.update({'hidden_dim': self.lstm_layer.units, 'bidirectional': self.bidirectional,
                       'dropout_rate': self.lstm_layer.dropout, 'return_sequences': self.lstm_layer.return_sequences})
        return config

