from model import Model
import tensorflow as tf
import datetime
from utils import data_utils, prior_utils
import numpy as np
import config

class HeterogeneousSupervision(Model):
	def __init__(self, sequence_length, mention_length, num_classes, vocab_size, 
		embedding_size, position_size, pretrained_embedding, wpe, type_info, num_lfs, hparams):
		self.sequence_length = sequence_length
		self.mention_length = mention_length
		self.num_classes = num_classes
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.position_size = position_size
		self.pretrained_embedding = pretrained_embedding
		self.wpe = wpe

		self.num_lfs = num_lfs
		
		self.state_size = hparams.state_size
		self.hidden_layers = hparams.hidden_layers
		self.hidden_size = hparams.hidden_size
		self.wpe_dim = hparams.wpe_dim
		self.l2_reg_lambda = hparams.l2_reg_lambda
		self.lr = hparams.lr
		self.dense_keep_prob = hparams.dense_keep_prob
		self.rnn_keep_prob = hparams.rnn_keep_prob
		self.batch_size = hparams.batch_size
		self.num_epochs = hparams.num_epochs

		self.prior = tf.Variable(prior_utils.create_prior(type_info), trainable=False, dtype=tf.float32, name="prior")
		self.tune = tf.Variable(np.transpose(prior_utils.create_prior(type_info, hparams.alpha)), trainable=False, dtype=tf.float32, name="tune")
		self.global_step = tf.Variable(0, name="global_step", trainable=False)

		self.build()

	def add_placeholders(self):
		self.input_words = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_words")
		self.input_textlen = tf.placeholder(tf.int32, [None], name="input_textlen")
		self.input_mentions = tf.placeholder(tf.int32, [None, self.mention_length], name="input_mentions")
		self.input_mentionlen = tf.placeholder(tf.int32, [None], name="input_mentionlen")
		self.input_positions = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_positions")
		self.input_labels = tf.placeholder(tf.float32, [None, self.num_classes], name="input_labels")
		self.lfs = tf.placeholder(tf.int32, [None, self.num_lfs], name="labeling_funcs")
		self.phase = tf.placeholder(tf.bool, name="phase")
		self.dense_dropout = tf.placeholder(tf.float32, name="dense_dropout")
		self.rnn_dropout = tf.placeholder(tf.float32, name="rnn_dropout")
	
		tmp = [i for i in range(self.mention_length)]
		tmp[0] = self.mention_length
		interval = tf.Variable(tmp, trainable=False)
		interval_row = tf.expand_dims(interval, 0)
		upper = tf.expand_dims(self.input_mentionlen-1, 1)
		mask = tf.less(interval_row, upper)
		self.mention = tf.where(mask, self.input_mentions, tf.zeros_like(self.input_mentions)) 
		self.mentionlen = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
		self.mentionlen = tf.cast(tf.where(tf.not_equal(self.mentionlen, tf.zeros_like(self.mentionlen)), self.mentionlen, tf.ones_like(self.mentionlen)), tf.float32)
		self.mentionlen = tf.expand_dims(self.mentionlen, 1)
	
	def create_feed_dict(self, input_words, input_textlen, input_mentions, input_mentionlen, input_positions, input_lfs, input_labels=None, phase=False, dense_dropout=1., rnn_dropout=1.):
		feed_dict = {
				self.input_words: input_words,
				self.input_textlen: input_textlen,
				self.input_mentions: input_mentions,
				self.input_mentionlen: input_mentionlen,
				self.input_positions: input_positions,
				self.lfs: input_lfs,
				self.phase: phase,
				self.dense_dropout: dense_dropout,
				self.rnn_dropout: rnn_dropout,
		}
		if input_labels is not None:
			feed_dict[self.input_labels] = input_labels
		return feed_dict

	def add_embedding(self):
		with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
			W = tf.Variable(self.pretrained_embedding, trainable=False, dtype=tf.float32, name="W")
			self.embedded_words = tf.nn.embedding_lookup(W, self.input_words)
			self.embedded_mentions = tf.nn.embedding_lookup(W, self.input_mentions)
			self.mention_embedding = tf.divide(tf.reduce_sum(tf.nn.embedding_lookup(W, self.mention), axis=1), self.mentionlen)

		with tf.device('/cpu:0'), tf.name_scope("position_embedding"):
			W = tf.Variable(self.wpe, trainable=False, dtype=tf.float32, name="W")
			self.wpe_chars = tf.nn.embedding_lookup(W, self.input_positions)

		self.input_sentences = tf.concat([self.embedded_words, self.wpe_chars], 2)
	
	def add_hidden_layer(self, x, idx):
		dim = self.feature_dim if idx == 0 else self.hidden_size
		with tf.variable_scope("hidden_%d" % idx):
			W = tf.get_variable("W", shape=[dim, self.hidden_size],
					initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
			b = tf.get_variable("b", shape=[self.hidden_size],
					initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
			h = tf.nn.xw_plus_b(x, W, b)
			h_norm = tf.layers.batch_normalization(h, training=self.phase)
			h_drop = tf.nn.dropout(tf.nn.relu(h_norm), self.dense_dropout, seed=config.RANDOM_SEED)
		return h_drop

	def extract_last_relevant(self, outputs, seq_len):
		batch_size = tf.shape(outputs)[0]
		max_length = int(outputs.get_shape()[1])
		num_units = int(outputs.get_shape()[2])
		index = tf.range(0, batch_size) * max_length + (seq_len-1)
		flat = tf.reshape(outputs, [-1, num_units])
		relevant = tf.gather(flat, index)
		return relevant
	
	def add_prediction_op(self):
		self.add_embedding()

		with tf.name_scope("sentence_repr"):
			attention_w = tf.get_variable("attention_w", [self.state_size, 1])
			cell_forward = tf.contrib.rnn.LSTMCell(self.state_size)
			cell_backward = tf.contrib.rnn.LSTMCell(self.state_size)
			cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, input_keep_prob=self.dense_dropout, output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)
			cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, input_keep_prob=self.dense_dropout, output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

			outputs, states = tf.nn.bidirectional_dynamic_rnn(
					cell_forward, cell_backward, self.input_sentences, 
					sequence_length=self.input_textlen, dtype=tf.float32)
			outputs_added = tf.nn.tanh(tf.add(outputs[0], outputs[1]))
			alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(outputs_added, [-1, self.state_size]), attention_w), [-1, self.sequence_length]))
			alpha = tf.expand_dims(alpha, 1)
			self.sen_repr = tf.squeeze(tf.matmul(alpha, outputs_added)) 

		with tf.name_scope("mention_repr"):
			cell = tf.contrib.rnn.LSTMCell(self.state_size)
			cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dense_dropout, output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

			outputs, states = tf.nn.dynamic_rnn(
					cell, self.embedded_mentions,
					sequence_length=self.input_mentionlen, dtype=tf.float32)
			self.men_repr = self.extract_last_relevant(outputs, self.input_mentionlen)

		self.features = tf.concat([self.sen_repr, self.men_repr, self.mention_embedding], -1)
		self.feature_dim = self.state_size * 2 + self.embedding_size

		h_drop = tf.nn.dropout(tf.nn.relu(self.features), self.dense_dropout, seed=config.RANDOM_SEED)
		h_drop.set_shape([None, self.feature_dim])
		h_output = tf.layers.batch_normalization(h_drop, training=self.phase)
		for i in range(self.hidden_layers):
			h_output = self.add_hidden_layer(h_output, i)
		if self.hidden_layers == 0:
			self.hidden_size = self.feature_dim

		with tf.variable_scope("prob_output"):
			W = tf.get_variable("W", shape=[self.hidden_size, self.num_classes],
					initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
			b = tf.get_variable("b", shape=[self.num_classes],
					initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
			self.scores = tf.nn.xw_plus_b(h_output, W, b, name="scores")
			self.proba = tf.nn.softmax(self.scores, name="proba")
			self.adjusted_proba = tf.matmul(self.proba, self.tune)
			self.adjusted_proba = tf.clip_by_value(self.adjusted_proba, 1e-10, 1.0)
			#self.adjusted_proba = tf.nn.softmax(self.adjusted_proba)
			self.predictions = tf.argmax(self.adjusted_proba, 1, name="predictions")

		with tf.variable_scope("lf_output"):
			preds = tf.reshape(tf.range(self.num_classes), [1, 1, self.num_classes])
			preds = tf.tile(preds, [tf.shape(self.features)[0], self.num_lfs, 1])
			lfs = tf.tile(tf.expand_dims(self.lfs, -1), [1, 1, self.num_classes])
			rho = tf.cast(tf.equal(lfs, preds), tf.float32)

			W = tf.get_variable("W", shape=[self.feature_dim, self.num_lfs],
					initializer=tf.contrib.layers.xavier_initializer(seed=config.RANDOM_SEED))
			self.pscores = tf.nn.sigmoid(tf.matmul(self.features, W), name="pscores")
			proficient_scores = tf.tile(tf.expand_dims(self.pscores, 2), [1, 1, self.num_classes])
			self.phi1 = 0.9
			self.phi2 = 0.1
			losses = tf.reduce_sum(tf.log(proficient_scores \
					* tf.pow(self.phi1, rho) * tf.pow(self.phi2, 1-rho) \
					+ (1-proficient_scores) * tf.pow(self.phi2, rho) \
					* tf.pow(self.phi1, 1-rho)), 1)
			self.inferred_proba = tf.nn.softmax(losses, name="inferred_proba")
			self.inferred_labels = tf.argmax(losses, axis=-1, name="inferred_labels")
			self.distribution = tf.one_hot(self.inferred_labels, self.num_classes)

	def add_loss_op(self):
		#with tf.name_scope("ce_loss"):
		#	target = tf.argmax(tf.multiply(self.adjusted_proba, self.input_labels), axis=1)
		#	target_index = tf.one_hot(target, self.num_classes)
		#	losses = -tf.reduce_sum(target_index*tf.log(self.adjusted_proba), 1)
		#	self.ce_loss = tf.reduce_mean(losses)

		with tf.name_scope("hs_loss"):
			preds = tf.tile(tf.expand_dims(self.inferred_labels, 1), [1, self.num_lfs])
			preds = tf.cast(preds, tf.int32)
			rho = tf.cast(tf.equal(self.lfs, preds), tf.float32)
			losses = -tf.reduce_sum(tf.log(self.pscores \
					* tf.pow(self.phi1, rho) * tf.pow(self.phi2, 1-rho) \
					+ (1-self.pscores) * tf.pow(self.phi2, rho) \
					* tf.pow(self.phi1, 1-rho)), 1)
			self.hs_loss = tf.reduce_mean(losses)

		with tf.name_scope("ce_loss"):
			target = tf.argmax(tf.multiply(self.inferred_proba, self.input_labels), axis=1)
			target_index = tf.one_hot(target, self.num_classes)
			losses = -tf.reduce_sum(target_index*tf.log(self.inferred_proba), 1)
			self.ce_loss = tf.reduce_mean(losses)

		#with tf.name_scope("kl_loss"):
		#	losses = -tf.reduce_sum(self.distribution*tf.log(self.adjusted_proba), 1)
		#	self.kl_loss = tf.reduce_mean(losses)

		with tf.name_scope("loss"):
			self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda), weights_list=tf.trainable_variables())
			#self.loss = self.ce_loss + self.hs_loss + self.kl_loss + self.l2_loss
			self.loss = self.hs_loss + self.ce_loss + self.l2_loss

		with tf.name_scope("results"): 
			type_path = tf.nn.embedding_lookup(self.prior, self.predictions)
			matched_types = tf.reduce_sum(tf.multiply(type_path, self.input_labels), axis=-1)
			predicted_types = tf.reduce_sum(type_path, axis=-1)
			label_types = tf.reduce_sum(self.input_labels, axis=-1)

			partial_equal = tf.cast(tf.greater(matched_types, 0.0), tf.float32)
			exact_equal = tf.cast(tf.equal(matched_types, label_types), tf.float32)

			self.partial_accuracy = tf.reduce_mean(partial_equal, name="partial_accuracy")
			self.exact_accuracy = tf.reduce_mean(exact_equal, name="exact_accuracy")

	def add_training_op(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.grads_and_vars = optimizer.compute_gradients(self.loss)

		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(extra_update_ops):
			self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
	
	def train_on_batch(self, sess, input_words, input_textlen, input_mentions, input_mentionlen, input_positions, input_lfs, input_labels):
		feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions, input_lfs, input_labels, True, self.dense_keep_prob, self.rnn_keep_prob)
		_, step, loss, pacc, eacc = sess.run([self.train_op, self.global_step, self.loss, self.partial_accuracy, self.exact_accuracy], feed_dict=feed)
		time_str = datetime.datetime.now().isoformat()
		print("{}: step {}, loss {:g} pacc {:g} eacc {:g}".format(time_str, step, loss, pacc, eacc))

	def evaluation_on_dev(self, sess, dev):
		batches = data_utils.batch_iter(dev, self.batch_size, 1, shuffle=False)
		total_loss = 0.0
		total_pacc = 0.0
		total_eacc = 0.0
		total_len = 0
		for batch in batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch = zip(*batch)
			feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch)
			loss, pacc, eacc = sess.run([self.loss, self.partial_accuracy, self.exact_accuracy], feed_dict=feed)
			total_loss += loss * len(labels_batch)
			total_pacc += pacc * len(labels_batch)
			total_eacc += eacc * len(labels_batch)
			total_len += len(labels_batch)
		time_str = datetime.datetime.now().isoformat()
		print("{}: loss {:g} partial acc {:g} exact acc {:g}".format(time_str, total_loss/total_len, total_pacc/total_len, total_eacc/total_len))
		return total_loss/total_len, total_pacc/total_len, total_eacc/total_len
	
	def fit(self, sess, train, dev=None):
		train_batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)
		data_size = len(train)
		num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
		best_dev_pacc = 0.0
		best_dev_eacc = 0.0
		best_dev_loss = 1e10
		best_dev_epoch = 0
		for batch in train_batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch = zip(*batch)
			self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch)
			current_step = tf.train.global_step(sess, self.global_step)
			if (current_step % num_batches_per_epoch == 0) and (dev is not None):
				print("\nEvaluation:")
				print("previous best dev epoch {}, best dev loss {:g}\n with partial acc {:g} and exact acc {:g}".format(best_dev_epoch, best_dev_loss, best_dev_pacc, best_dev_eacc))
				loss, pacc, eacc = self.evaluation_on_dev(sess, dev)
				print("")
				if loss < best_dev_loss:
					best_dev_loss = loss
					best_dev_pacc = pacc
					best_dev_eacc = eacc
					best_dev_epoch = current_step // num_batches_per_epoch
				if current_step//num_batches_per_epoch - best_dev_epoch > 3:
					break
		return best_dev_epoch, best_dev_loss, best_dev_pacc, best_dev_eacc

	def predict(self, sess, test):
		batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
		all_predictions = []
		for batch in batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch = zip(*batch)
			feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch)
			batch_predictions = sess.run(self.predictions, feed_dict=feed)
			all_predictions = np.concatenate([all_predictions, batch_predictions])
		return all_predictions

	def evaluate(self, sess, train, test):
		train_batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)
		data_size = len(train)
		num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
		for batch in train_batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch = zip(*batch)
			self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, lfs_batch, labels_batch)
			current_step = tf.train.global_step(sess, self.global_step)
			if current_step % num_batches_per_epoch == 0:
				yield self.predict(sess, test)
