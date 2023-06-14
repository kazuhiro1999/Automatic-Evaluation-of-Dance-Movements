import tensorflow as tf


''' for triplet '''
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(distance_positive - distance_negative + alpha, 0.0)
    return tf.reduce_mean(loss)

def swap_error(y_true, y_pred):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    # error if distance detween anchor and positives is larger than distance detween anchor and negatives
    swap_error = tf.reduce_mean(tf.cast(distance_positive > distance_negative, dtype=tf.float32))
    
    return swap_error


''' for contrastive '''
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def contrastive_accuracy(y_true, y_pred):
    # flip label because labels are similarity (1:similar, 0:not similar)
    y_true_flipped = 1 - y_true
    # calculate binary accuracy
    return tf.keras.metrics.binary_accuracy(y_true_flipped, y_pred)
