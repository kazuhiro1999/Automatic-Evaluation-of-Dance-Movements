import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, LayerNormalization, GlobalAveragePooling1D, Dropout, Subtract, Multiply, Lambda
from tensorflow.keras.models import Model

from models.transformer import TransformerEncoder


''' Motion Encoder '''

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


def create_transformer_encoder(sequence_length, input_dim, hidden_dim, output_dim, dropout_rate, num_transformer_blocks=6, num_heads=8):
    inputs = Input(shape=(sequence_length, input_dim))
    x = Dense(hidden_dim)(inputs)
    x = TransformerEncoder(hidden_dim, sequence_length, num_transformer_blocks, num_heads, hidden_dim, dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs, name="transformer_encoder")
    return model


''' Evaluation Model '''

def create_model(encoder, input_dim, hidden_dims=[128,64], output_dim=3, activation='softmax', dropout_rate=0.1, trainable=False):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = Input(shape=(None, input_dim))
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
    x = features_diff
    
    for hidden_dim in hidden_dims:
        x = Dense(hidden_dim, activation='relu')(x)
        
    outputs = Dense(output_dim, activation=activation)(x)

    # モデルの定義
    reference_model = Model(inputs=[input_evaluation, input_reference], outputs=outputs, name="Reference-guided")
    return reference_model


''' Contrastive Learning Model '''

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