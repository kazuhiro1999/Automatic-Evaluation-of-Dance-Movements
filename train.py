import argparse
import numpy as np
import tensorflow as tf

from loader.dataloader import DataLoader
from generator import DataGenerator
from models.model_util import create_lstm_encoder, create_embedding_model, create_model, create_reference_model, create_triplet_model
from utils.loss import swap_error, triplet_loss
from utils.setting import determine_split, get_frame_splits, train_test_split

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    item = 'Dynamics'
    num_classes = 3
    path = ""
    method = 'default'
    learn_encoder = False
    learning_rate = 2*1e-4
    epochs = 50

    dataloader = DataLoader(path)

    ''' determine train test split '''
    data = dataloader.data
    scores = data[item][data['IsReference'] | data['Annotated']].to_list()
    filtered_scores = [score for score in scores if score >= 0]
    counts = np.bincount(filtered_scores)
    splits, label_transform = determine_split(counts, num_classes)
    train_keys, test_keys = train_test_split(data, item, splits)
    reference_keys = data['ID'][data['IsReference']].to_list()

    # split dance motion by beat
    frame_splits = get_frame_splits(bpm=161, fps=60, offset=9.9, n_beats=4, step=4, size=8)


    # create encoder
    encoder = create_lstm_encoder(input_dim=258, hidden_dim=512, output_dim=128, num_lstm_layers=1, bidirectional=False, dropout_rate=0.115)
    
    # learn encoder
    if learn_encoder:
        train_generator = DataGenerator(dataloader, train_keys, reference_keys, frame_splits, item, label_transform, method='triplet')
        valid_generator = DataGenerator(dataloader, test_keys, reference_keys, frame_splits, item, label_transform, method='triplet')

        embedding_model = create_embedding_model(encoder, embedding_dim=128)

        triplet_model = create_triplet_model(embedding_model)

        triplet_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True),
            loss=triplet_loss,
            metrics=[swap_error],
        )

        triplet_model.summary()
        history = triplet_model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=16, shuffle=True)

                
    # learn evaluation model
    if method == 'default':
    
        train_generator = DataGenerator(dataloader, train_keys, reference_keys, frame_splits, item, label_transform, method='default')
        valid_generator = DataGenerator(dataloader, test_keys, reference_keys, frame_splits, item, label_transform, method='default')

        model = create_model(encoder, input_dim=258, hidden_dims=[16,8], output_dim=num_classes, activation='softmax', dropout_rate=0.115, trainable=True)
    
    elif method == 'triplet':

        train_generator = DataGenerator(dataloader, train_keys, reference_keys, frame_splits, item, label_transform, method='default')
        valid_generator = DataGenerator(dataloader, test_keys, reference_keys, frame_splits, item, label_transform, method='default')

        model = create_model(encoder, input_dim=258, hidden_dims=[16,8], output_dim=num_classes, activation='softmax', dropout_rate=0.115, trainable=False)

    elif method == 'reference':

        train_generator = DataGenerator(dataloader, train_keys, reference_keys, frame_splits, item, label_transform, method='reference')
        valid_generator = DataGenerator(dataloader, test_keys, reference_keys, frame_splits, item, label_transform, method='reference')

        model = create_reference_model(encoder, hidden_dims=[128,64], output_dim=num_classes, activation='softmax', diff='sub', dropout_rate=0.115, trainable=False)

    else:
        pass


    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )
    model.summary()

    checkpoint_filepath = '/models/tmp/best_model'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_categorical_accuracy', mode='max', save_best_only=True)

    history = model.fit(train_generator, validation_data = valid_generator, epochs=epochs, batch_size=16, shuffle=True, callbacks=[model_checkpoint_callback])

    