import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Bidirectional, BatchNormalization, Activation, LayerNormalization, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Dropout, Subtract, Multiply, Concatenate, Lambda, concatenate, TimeDistributed, RepeatVector, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import activations
from tcn import TCN
from keras_gcn import GraphConv
from feature_extraction import ADJ_MATRIX


''' Encoder '''

def create_lstm_encoder(input_dim, hidden_dim, output_dim, num_lstm_layers=1, bidirectional=False, dropout_rate=0.0, use_conv1d=False, conv1d_filters=[], kernel_size=7):
    inputs = Input(shape=(None, input_dim))
    
    x = inputs
    for _ in range(num_lstm_layers - 1):
        lstm_layer = LSTM(units=hidden_dim, activation='tanh', dropout=dropout_rate, return_sequences=True)
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        x = lstm_layer(x)
    
    lstm_layer = LSTM(units=hidden_dim, activation='tanh', dropout=dropout_rate, return_sequences=use_conv1d)
    if bidirectional:
        lstm_layer = Bidirectional(lstm_layer)
    x = lstm_layer(x)
    
    if use_conv1d:
        for filters in conv1d_filters:
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
            x = MaxPooling1D()(x)
        x = GlobalAveragePooling1D()(x)
        
    outputs = Dense(output_dim, activation='relu')(x)

    model = Model(inputs, outputs)
    return model


def create_gcn_lstm_encoder(num_joints, num_features, num_filters, hidden_dim, output_dim, num_lstm_layers=1, bidirectional=False, dropout_rate=0.0):
    
    inputs = Input(shape=(None, num_joints, num_features)) 
    
    # GCN層
    gcn_output = TimeDistributed(CustomGraphConv(ADJ_MATRIX, num_filters, activation='relu'))(inputs)
    
    x = TimeDistributed(Flatten())(gcn_output)
    for _ in range(num_lstm_layers - 1):
        lstm_layer = LSTM(units=hidden_dim, activation='tanh', dropout=dropout_rate, return_sequences=True)
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)
        x = lstm_layer(x)
    
    lstm_layer = LSTM(units=hidden_dim, activation='tanh', dropout=dropout_rate, return_sequences=False)
    if bidirectional:
        lstm_layer = Bidirectional(lstm_layer)
    x = lstm_layer(x)
    
    outputs = Dense(output_dim, activation='relu')(x)

    model = Model(inputs, outputs)
    return model

def gcn_module(num_filters):
    return TimeDistributed(CustomGraphConv(ADJ_MATRIX, num_filters, activation='relu'))

class CustomGraphConv(GraphConv):
    def __init__(self, adj_matrix, *args, **kwargs):
        super(CustomGraphConv, self).__init__(*args, **kwargs)
        self.adj_matrix = np.array(adj_matrix)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        batch_adj_matrix = tf.tile(tf.expand_dims(self.adj_matrix, axis=0), [batch_size, 1, 1])
        return super(CustomGraphConv, self).call([inputs, batch_adj_matrix])

    def get_config(self):
        config = super(CustomGraphConv, self).get_config()
        config['activation'] = activations.serialize(self.activation)
        config.update({'adj_matrix': self.adj_matrix.tolist()})
        return config
        
    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.units,)
    

def create_tcn_encoder(input_dim, output_dim, nb_filters=64, kernel_size=3, nb_stacks=1, dilations=[1, 2, 4, 8, 16, 32], dropout_rate=0.0):
    inputs = Input(shape=(None, input_dim))
    
    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=dropout_rate,
        return_sequences=False,
        activation='relu',
        kernel_initializer='he_normal',
        use_batch_norm=False,
        use_layer_norm=False,
        use_weight_norm=False,
        go_backwards=False,
        return_state=False
    )(inputs)
    
    outputs = Dense(output_dim, activation='relu')(x)

    model = Model(inputs, outputs)
    return model
    

def create_inception_model(input_dim, output_dim, num_blocks, filter_sizes, bottleneck_size=32):
    inputs = Input(shape=(None, input_dim)) 

    x = inputs
    for _ in range(num_blocks):
        x = _inception_module(x, filter_sizes)
    
    gap_layer = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim, activation='relu')(gap_layer)
    
    # モデルの構築
    model = Model(inputs=inputs, outputs=outputs)
    return model

def _inception_module(inputs, filter_sizes, bottleneck_size=32):
    # ボトルネック層（オプション）
    x = Conv1D(filters=bottleneck_size, kernel_size=1, padding='same', activation='relu')(inputs)

    # 異なるカーネルサイズを持つ畳み込み層
    conv_list = []
    for filter_size in filter_sizes:
        conv_list.append(Conv1D(filters=bottleneck_size, kernel_size=filter_size, padding='same', activation='relu')(x))

    # マックスプーリング層
    max_pool_1 = MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)
    conv_6 = Conv1D(filters=bottleneck_size, kernel_size=1, padding='same', activation='relu')(max_pool_1)

    # 結合
    concat = concatenate(conv_list + [conv_6], axis=2)
    return concat



''' Evaluation Model '''

def create_model(encoder, hidden_dims=[128,64], output_dim=3, activation='softmax', dropout_rate=0.1, trainable=False):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = Input(shape=encoder.input_shape[1:])
    x = encoder(inputs)
    for hidden_dim in hidden_dims:
        x = Dense(hidden_dim, activation='tanh')(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(output_dim, activation=activation)(x)

    model = Model(inputs, outputs, name='evaluation_model')
    return model
    

def create_reference_model(encoder, hidden_dims=[128,64], output_dim=3, activation='softmax', dropout_rate=0.1, diff='sub', trainable=False):
    for layer in encoder.layers:
        layer.trainable = trainable
    
    input_evaluation = Input((None, encoder.input_shape[-1]))
    input_reference = Input((None, encoder.input_shape[-1]))

    # extract embeddings
    features_evaluation = encoder(input_evaluation)
    features_reference = encoder(input_reference)
    
    #features_evaluation = Dropout(0.2)(features_evaluation)
    #features_evaluation = Dense(128)(features_evaluation)
    
    #features_reference = Dropout(0.2)(features_reference)
    #features_reference = Dense(128)(features_reference)
    
    # calculate difference between embeddings
    if diff == 'mul':
        features_diff = Multiply()([features_evaluation, features_reference])
    else:
        features_diff = Subtract()([features_evaluation, features_reference])

    # MLP
    #x = features_diff
    x = Concatenate()([features_evaluation, features_reference, features_diff])
    
    for hidden_dim in hidden_dims:
        x = Dense(hidden_dim, activation='tanh')(x)
        
    outputs = Dense(output_dim, activation=activation)(x)

    # モデルの定義
    reference_model = Model(inputs=[input_evaluation, input_reference], outputs=outputs, name="reference-guided")
    return reference_model


''' Contrastive Learning '''

def create_embedding_model(encoder, embedding_dim=64):
    x = encoder.output
    embedding_layer = Dense(embedding_dim, activation='relu')(x)
    return Model(encoder.input, embedding_layer, name='embedding_model')


def create_triplet_model(embedding_model):
    anchor_input = Input((None, embedding_model.input_shape[-1]))
    positive_input = Input((None, embedding_model.input_shape[-1]))
    negative_input = Input((None, embedding_model.input_shape[-1]))

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    outputs = Lambda(stack_embeddings)([anchor_embedding, positive_embedding, negative_embedding])

    return Model([anchor_input, positive_input, negative_input], outputs, name="triplet_model")


def stack_embeddings(embeddings):
    return tf.stack(embeddings, axis=1)
