import numpy as np
import random
import itertools


def determine_split(counts, num_classes=3):
    target_count = sum(counts) / num_classes

    best_split = None
    best_split_indices = None
    min_e = 1e+5

    for split_indices in itertools.combinations(range(1,len(counts)), num_classes-1):
        ''' split pattern '''
        split = np.zeros(len(counts), dtype=np.int64)
        for cls, idx in enumerate(split_indices):
            split[idx:] = cls+1

        ''' calculate enargy '''
        class_counts = np.zeros(num_classes)
        for i, count in enumerate(counts):
            class_counts[split[i]] += count

        e = ((class_counts - target_count)**2).sum()

        ''' save best split '''
        if e < min_e:
            min_e = e
            best_split_indices = split_indices
            best_split = split

    return best_split_indices, best_split


def train_test_split(df, label, splits, test_split=0.2):
    train_keys = []
    test_keys = []
    
    # split by dancer
    splits = (0,) + splits + (10,)
    for i in range(len(splits)-1):
        dancer = df[(splits[i] < df[label]) &(df[label] < splits[i+1])]['Dancer'].unique().tolist()
        test_size = max(int(len(dancer) * test_split), 1)
        test = random.sample(dancer, test_size)
        
        train_keys.extend(df[df['Annotated'] & ~df['Dancer'].isin(test)]['ID'].tolist())
        test_keys.extend(df[df['Annotated'] & df['Dancer'].isin(test)]['ID'].tolist())
        
    return train_keys, test_keys


def get_frame_splits(bpm, fps=60, offset=0.0, n_beats=4, step=4, size=8):
    sec_per_beat = 60 / bpm
    frame_length = int((sec_per_beat) * n_beats * fps)
    frame_splits = []
    for beat_i in range(1, size*step, step):
        start_time = offset + (beat_i - 1) * sec_per_beat
        start_frame = int(start_time * fps)
        frame_splits.append([start_frame, start_frame + frame_length])
    return frame_splits