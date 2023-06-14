import numpy as np
import tensorflow as tf
import itertools

from preprocess.feature_extraction import extract_features, data_augmentation


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, loader, keys, reference_keys, frame_splits, item, label_transform, method='default', batch_size=16, shuffle=True):
        self.loader = loader
        self.keys = keys
        self.reference_keys = reference_keys
        self.frame_splits = frame_splits
        self.item = item
        self.label_transform = label_transform
        self.method = method
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initialize()

    def __len__(self):
        return int(np.ceil(len(self.info) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_info = [self.info[i] for i in indexes]

        if self.method == 'default':
            X, Y = self.get_default_data(batch_info)
        elif self.method == 'triplet':
            X, Y = self.get_triplet_data(batch_info)
        elif self.method == 'reference':
            X, Y = self.get_reference_data(batch_info)
        else:
            raise ValueError('Unknown method: {}'.format(self.method))

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.info))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def initialize(self):
        if self.method in ['contrastive, triplet']:
            self.keys.extend(self.reference_keys)
        self.info = list(itertools.product(self.keys, self.frame_splits))
        self.on_epoch_end()
        
        
    def get_default_data(self, batch_info):
        X, Y = [], []
        for info in batch_info:
            key, (start_frame, end_frame) = info
            is_mirror = np.random.choice([True, False])
            
            # features
            keypoints3d_list = self.loader.load_keypoints3d(key, start_frame, end_frame)
            keypoints3d_list = data_augmentation(keypoints3d_list, alpha=0.05, is_mirror=is_mirror)
            features = extract_features(keypoints3d_list)
            
            # score
            score = self.loader.load_score(key, self.item)
            label = self.transform_label(score)
            
            X.append(features)
            Y.append(label)

        return np.array(X), np.array(Y)
    
    
    def get_triplet_data(self, batch_info):
        anchor_inputs = []
        positive_inputs = []
        negative_inputs = []
        for info in batch_info:
            anchor_key, (start_frame, end_frame) = info
            is_mirror = np.random.choice([True, False])
            if self.loader.data[self.loader.data['ID'] == anchor_key]['IsReference'].tolist()[0]:
                positive_key = np.random.choice([key for key in self.keys if self.loader.load_score(key, self.item) > 4])
                negative_key = np.random.choice([key for key in self.keys if self.loader.load_score(key, self.item) < 5])
            else:            
                positive_key = np.random.choice([key for key in self.keys if abs(self.loader.load_score(key, self.item) - self.loader.load_score(anchor_key, self.item)) < 2])
                negative_key = np.random.choice([key for key in self.keys if abs(self.loader.load_score(key, self.item) - self.loader.load_score(anchor_key, self.item)) > 1])

            # anchor
            keypoints3d_list = self.loader.load_keypoints3d(anchor_key, start_frame, end_frame)
            keypoints3d_list = data_augmentation(keypoints3d_list, alpha=0.05, is_mirror=is_mirror)
            features = extract_features(keypoints3d_list)
            anchor_inputs.append(features)

            # positives        
            keypoints3d_list = self.loader.load_keypoints3d(positive_key, start_frame, end_frame)
            keypoints3d_list = data_augmentation(keypoints3d_list, alpha=0.05, is_mirror=is_mirror)
            features = extract_features(keypoints3d_list)
            positive_inputs.append(features)

            # negatives        
            keypoints3d_list = self.loader.load_keypoints3d(negative_key, start_frame, end_frame)
            keypoints3d_list = data_augmentation(keypoints3d_list, alpha=0.05, is_mirror=is_mirror)
            features = extract_features(keypoints3d_list)
            negative_inputs.append(features)     

        return [np.array(anchor_inputs), np.array(positive_inputs), np.array(negative_inputs)], np.zeros((len(batch_info),))


    def get_reference_data(self, batch_info):
        anchor_inputs = []
        reference_inputs = []
        labels = []
        for info in batch_info:
            anchor_key, (start_frame, end_frame) = info
            reference_key = np.random.choice(self.reference_keys)
            is_mirror = np.random.choice([True, False])

            # anchor
            keypoints3d_list = self.loader.load_keypoints3d(anchor_key, start_frame, end_frame)
            keypoints3d_list = data_augmentation(keypoints3d_list, alpha=0.05, is_mirror=is_mirror)
            features = extract_features(keypoints3d_list)
            anchor_inputs.append(features)
            
            score = self.loader.load_score(anchor_key, self.item)
            label = self.transform_label(score)
            labels.append(label)

            # reference       
            keypoints3d_list = self.loader.load_keypoints3d(reference_key, start_frame, end_frame)
            keypoints3d_list = data_augmentation(keypoints3d_list, alpha=0.05, is_mirror=is_mirror)
            features = extract_features(keypoints3d_list)
            reference_inputs.append(features)

        return [np.array(anchor_inputs), np.array(reference_inputs)], np.array(labels)

    
    def transform_label(self, score):
        y = self.label_transform[score] 
        y = np.eye(max(self.label_transform)+1)[y]  
        return y