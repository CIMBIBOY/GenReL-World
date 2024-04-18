import tensorflow as tf

class RNNPolicy(tf.keras.Model):
    def __init__(self, num_actions):
        super(RNNPolicy, self).__init__()
        self.rnn_layer = tf.keras.layers.LSTM(128)
        self.dense_layer = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.rnn_layer(inputs)
        return self.dense_layer(x)