import numpy as np
import tensorflow as tf
import numpy as np

def Attention_Matrix(K, Q, use_mask=False):
	"""
	Calculate the attention matrix for the attention layer of the model.

	:param K: the matrix of key vectors
	:param Q: the matrix of query vectors
	:return: the attention matrix
	"""
	
	window_size_keys = K.get_shape()[1]
	key_embedding_size = K.get_shape()[2]
	# Calculate the score vector of each slice of the spectrogram
	score = tf.matmul(Q, tf.reshape(K, [-1, key_embedding_size, window_size_keys]))
	# Scale scores and apply softmax
	score = score / tf.sqrt(tf.cast(key_embedding_size, np.float32))
	weights = tf.nn.softmax(score)
	return weights


class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size):		
		super(Atten_Head, self).__init__()
		self.K = tf.random.truncated_normal(shape=[input_size, output_size], stddev=0.01, dtype=tf.float32)
		self.V = tf.random.truncated_normal(shape=[input_size, output_size], stddev=0.01, dtype=tf.float32)
		self.Q = tf.random.truncated_normal(shape=[input_size, output_size], stddev=0.01, dtype=tf.float32)
		
	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		"""
		Runs a single attention head.

		:param inputs_for_keys: tensor of [batch_size x spectrogram_width x input_size]
		:param inputs_for_values: tensor of [batch_size x spectrogram_width x input_size]
		:param inputs_for_queries: tensor of [batch_size x spectrogram_width x input_size]
		:return: tensor of [batch_size x spectrogram_width x output_size]
		"""

		K = tf.matmul(inputs_for_keys, self.K)
		V = tf.matmul(inputs_for_values, self.V)
		Q = tf.matmul(inputs_for_queries, self.Q)

		weights = Attention_Matrix(K, Q)
		Z = tf.matmul(weights, V)

		return Z


class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		Runs a feed forward network.

		:param inputs: input tensor [batch_size x spectrogram_width x embedding_size]
		:return: tensor [batch_size x spectrogram_width x embedding_size]
		"""

		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Transformer_Block, self).__init__()
		self.ff_layer = Feed_Forwards(emb_sz)
		self.self_atten = Atten_Head(emb_sz, emb_sz)
		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None):
		"""
		This functions calls a transformer block.

		:param inputs: tensor of [batch_size x spectrogram_width x embedding_size]
		:context: None
		:return: tensor [batch_size x spectrogram_width x embedding_size]
		"""

		atten_out = self.self_atten(inputs,inputs,inputs)
		atten_out+=inputs
		atten_normalized = self.layer_norm(atten_out)

		ff_out=self.ff_layer(atten_normalized)
		ff_out+=atten_normalized
		ff_norm = self.layer_norm(ff_out)

		return tf.nn.relu(ff_norm)
