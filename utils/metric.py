import tensorflow as tf
import tensorflow_addons as tfa


def swap_error(y_true, y_pred):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    # error if distance detween anchor and positives is larger than distance detween anchor and negatives
    swap_error = tf.reduce_mean(tf.cast(distance_positive > distance_negative, dtype=tf.float32))
    
    return swap_error


def contrastive_accuracy(y_true, y_pred):
    # flip label because labels are similarity (1:similar, 0:not similar)
    y_true_flipped = 1 - y_true
    # calculate binary accuracy
    return tf.keras.metrics.binary_accuracy(y_true_flipped, y_pred)


''' custom metrics '''
class CohenKappa(tfa.metrics.CohenKappa):
    def __init__(self, num_classes=9, name="cohen_kappa", weightage='quadratic', **kwargs):
        super(CohenKappa, self).__init__(num_classes=num_classes, weightage=weightage, name=name, **kwargs)

    def get_config(self):
        config = super(CohenKappa, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)

        # y_pred をスコアからラベルに変換
        class_values = tf.range(self.num_classes, dtype=tf.float32)
        predicted_scores = tf.reduce_sum(y_pred * class_values, axis=-1)
        y_pred = tf.round(predicted_scores)
        y_pred = tf.cast(y_pred, tf.int32)

        # 親クラスのupdate_stateを呼び出す
        super(CohenKappa, self).update_state(y_true, y_pred, sample_weight)