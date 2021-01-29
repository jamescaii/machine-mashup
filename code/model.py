import numpy as np
import tensorflow as tf
import model_funcs as transformer
from scipy import signal

class SongMasher(tf.keras.Model):
    def __init__(self, spectrogram_width, spectrogram_height, learning_rate):
        super(SongMasher, self).__init__()

        # Define hyperparameters
        self.batch_size = 100
        self.learning_rate = learning_rate
        self.embedding_size = 256
        self.width = spectrogram_width
        self.height = spectrogram_height

        # Define K, V, Q matrices
        self.K = tf.random.truncated_normal(shape=[self.width, self.embedding_size], stddev=0.01, dtype=tf.float32)
        self.V = tf.random.truncated_normal(shape=[self.width, self.embedding_size], stddev=0.01, dtype=tf.float32)
        self.Q = tf.random.truncated_normal(shape=[self.width, self.embedding_size], stddev=0.01, dtype=tf.float32)

        # Define encoder layers
        self.encoder = tf.keras.layers.LSTM(self.embedding_size, activation='tanh', return_state=True, return_sequences=True)
        self.encoder_attention = transformer.Transformer_Block(self.embedding_size)

        # Define decoder layer
        self.decoder = tf.keras.layers.LSTM(self.embedding_size, activation='tanh', return_state=True, return_sequences=True)
        self.decoder_dense = tf.keras.layers.Dense(self.height)

    @tf.function
    def call(self, originals):
        """
        :param originals1: spectrogram of one of the original two songs that will be used to generate a mashup
        :param originals2: spectrogram of the other original song that will be used to generate a mashup
        :return: the spectrogram of the mashup generated from the model
        """
        
        originals1, originals2 = originals
        # Generate encodings of the original two songs using attention
        all_states1, fin_state1, cell_state1 = self.encoder(originals1, initial_state=None)
        attention1 = self.encoder_attention(all_states1)
        all_states2, fin_state2, cell_state2 = self.encoder(originals2, initial_state=None)
        attention2 = self.encoder_attention(all_states2)
        # Generate the mashup by using the encodings found above with the original two songs
        orig_with_context = tf.concat([attention1, attention2, originals1, originals2], axis=-1)
        all_states3, fin_state3, cell_state3 = self.decoder(orig_with_context, [fin_state1, fin_state2])
        artif_mash = self.decoder_dense(all_states3)

        return artif_mash

    def cross_correlation(self, artif, real):
        """
        Calculates cross correltaion between two spectrograms.

        :param artif: artificial spectrogram generated from model
        :param real: real spectrogram downloaded from YouTube
        :return: cross correlation between the two spectrograms
        """

        return signal.correlate(artif, real, mode="valid")[0][0]

    def loss_function(self, artif, real):
        """
        Calculates the model loss after one forward pass.

        :param artif: artificial spectrogram generated from model
        :param real: real spectrogram downloaded from YouTube
        :return: mean squared error of the two spectrograms
        """

        return tf.reduce_mean(tf.square(artif - real))


    def __call__(self, *args, **kwargs):
        return super(SongMasher, self).__call__(*args, **kwargs)
