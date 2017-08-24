import tensorflow as tf
from optparse import OptionParser
from utils import embedding_utils, data_utils, pkl_utils
import config
import os
import numpy as np
import pandas as pd

def parse_args(parser):
	parser.add_option("-m", "--model", dest="model_name", type="string")
	parser.add_option("--input", dest="input_file", type="string", default="")
	parser.add_option("--output", dest="output_file", type="string")
	parser.add_option("-e", dest="embedding", default=False, action="store_true")
	options, args = parser.parse_args()
	return options, args

def get_types(model_name, input_file, output_file):
	checkpoint_file = os.path.join(config.CHECKPOINT_DIR, model_name)
	type2id, typeDict = pkl_utils._load(config.WIKI_TYPE)
	id2type = {type2id[x]:x for x in type2id.keys()}

	df = pd.read_csv(input_file, sep="\t", names=["r", "e1", "x1", "y1", "e2", "x2", "y2", "s"]) 
	n = df.shape[0]
	words1 = np.array(df.s)
	mentions1 = np.array(df.e1)
	positions1 = np.array([[x, y] for x, y in zip(df.x1, df.y1+1)])
	words2 = np.array(df.s)
	mentions2 = np.array(df.e2)
	positions2 = np.array([[x, y] for x, y in zip(df.x2, df.y2+1)])
	
	words = np.concatenate([words1, words2])
	mentions = np.concatenate([mentions1, mentions2])
	positions = np.concatenate([positions1, positions2])

	embedding = embedding_utils.Embedding.restore(checkpoint_file)

	textlen = np.array([embedding.len_transform1(x) for x in words])
	words = np.array([embedding.text_transform1(x) for x in words])
	mentionlen = np.array([embedding.len_transform2(x) for x in mentions])
	mentions = np.array([embedding.text_transform2(x) for x in mentions])
	positions = np.array([embedding.position_transform(x) for x in positions])
	labels = np.zeros(2*n)
	test_set = list(zip(words, textlen, mentions, mentionlen, positions, labels))

	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		input_words = graph.get_operation_by_name("input_words").outputs[0]
		input_textlen = graph.get_operation_by_name("input_textlen").outputs[0]
		input_mentions = graph.get_operation_by_name("input_mentions").outputs[0]
		input_mentionlen = graph.get_operation_by_name("input_mentionlen").outputs[0]
		input_positions = graph.get_operation_by_name("input_positions").outputs[0]
		phase = graph.get_operation_by_name("phase").outputs[0]
		dense_dropout = graph.get_operation_by_name("dense_dropout").outputs[0]
		rnn_dropout = graph.get_operation_by_name("rnn_dropout").outputs[0]

		pred_op = graph.get_operation_by_name("output/predictions").outputs[0]
		batches = data_utils.batch_iter(test_set, 512, 1, shuffle=False)
		all_predictions = []
		for batch in batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch = zip(*batch)
			feed = {
				input_words: words_batch,
				input_textlen: textlen_batch,
				input_mentions: mentions_batch,
				input_mentionlen: mentionlen_batch,
				input_positions: positions_batch,
				phase: False,
				dense_dropout: 1.0,
				rnn_dropout: 1.0
			}
			batch_predictions = sess.run(pred_op, feed_dict=feed)
			all_predictions = np.concatenate([all_predictions, batch_predictions])
	
	df["t1"] = all_predictions[:n]
	df["t2"] = all_predictions[n:]
	df["t1"] = df["t1"].map(id2type)
	df["t2"] = df["t2"].map(id2type)
	df.to_csv(output_file, sep="\t", header=False, index=False)

def get_embeddings(model_name, output_file):
	checkpoint_file = os.path.join(config.CHECKPOINT_DIR, model_name)
	graph = tf.Graph()
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		embedding_op = graph.get_tensor_by_name("output/W:0")
		type_embedding = sess.run(embedding_op)
		np.save(output_file, type_embedding)

def main(options):
	if options.input_file != "":
		get_types(options.model_name, options.input_file, options.output_file)
	if options.embedding:
		get_embeddings(options.model_name, options.output_file)

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
