from model_param_space import ModelParamSpace
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from optparse import OptionParser
from utils import logging_utils, data_utils, embedding_utils, pkl_utils
import numpy as np
from sklearn.model_selection import ShuffleSplit
import os
import config
import datetime
import tensorflow as tf
from nfetc import NFETC

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

class Task:
	def __init__(self, model_name, data_name, cv_runs, params_dict, logger):
		print("Loading data...")
		if data_name == "wiki":
			words_train, mentions_train, positions_train, labels_train = data_utils.load(config.WIKI_TRAIN_CLEAN)
			words, mentions, positions, labels = data_utils.load(config.WIKI_TEST_CLEAN)
			type2id, typeDict = pkl_utils._load(config.WIKI_TYPE)
			num_types = len(type2id)
			type_info = config.WIKI_TYPE
		else:
			words_train, mentions_train, positions_train, labels_train = data_utils.load(config.ONTONOTES_TRAIN_CLEAN)
			words, mentions, positions, labels = data_utils.load(config.ONTONOTES_TEST_CLEAN)
			type2id, typeDict = pkl_utils._load(config.ONTONOTES_TYPE)
			num_types = len(type2id)
			type_info = config.ONTONOTES_TYPE

		id2type = {type2id[x]:x for x in type2id.keys()}
		def type2vec(types):
			tmp = np.zeros(num_types)
			for t in types.split():
				tmp[type2id[t]] = 1.0
			return tmp
		labels_train = np.array([type2vec(t) for t in labels_train])
		labels = np.array([type2vec(t) for t in labels])

		self.embedding = embedding_utils.Embedding(config.EMBEDDING_DATA, list(words_train)+list(words), config.MAX_DOCUMENT_LENGTH, config.MENTION_SIZE)

		print("Preprocessing data...")
		textlen_train = np.array([self.embedding.len_transform1(x) for x in words_train])
		words_train = np.array([self.embedding.text_transform1(x) for x in words_train])
		mentionlen_train = np.array([self.embedding.len_transform2(x) for x in mentions_train])
		mentions_train = np.array([self.embedding.text_transform2(x) for x in mentions_train])
		positions_train = np.array([self.embedding.position_transform(x) for x in positions_train])

		textlen = np.array([self.embedding.len_transform1(x) for x in words])
		words = np.array([self.embedding.text_transform1(x) for x in words])
		mentionlen = np.array([self.embedding.len_transform2(x) for x in mentions])
		mentions = np.array([self.embedding.text_transform2(x) for x in mentions])
		positions = np.array([self.embedding.position_transform(x) for x in positions])

		ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=config.RANDOM_SEED)
		for test_index, valid_index in ss.split(np.zeros(len(labels)), labels):
			textlen_test, textlen_valid = textlen[test_index], textlen[valid_index]
			words_test, words_valid = words[test_index], words[valid_index]
			mentionlen_test, mentionlen_valid = mentionlen[test_index], mentionlen[valid_index]
			mentions_test, mentions_valid = mentions[test_index], mentions[valid_index]
			positions_test, positions_valid = positions[test_index], positions[valid_index]
			labels_test, labels_valid = labels[test_index], labels[valid_index]
		self.train_set = list(zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train))
		self.valid_set = list(zip(words_valid, textlen_valid, mentions_valid, mentionlen_valid, positions_valid, labels_valid))
		self.test_set = list(zip(words_test, textlen_test, mentions_test, mentionlen_test, positions_test, labels_test))

		self.model_name = model_name
		self.data_name = data_name
		self.cv_runs = cv_runs
		self.params_dict = params_dict
		self.hparams = AttrDict(params_dict)
		self.logger = logger

		self.num_types = num_types
		self.type_info = type_info

		self.model = self._get_model()
		self.saver = tf.train.Saver(tf.global_variables())
		checkpoint_dir = os.path.abspath(config.CHECKPOINT_DIR)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.checkpoint_prefix = os.path.join(checkpoint_dir, self.__str__())
	
	def __str__(self):
		return self.model_name

	def _get_model(self):
		np.random.seed(config.RANDOM_SEED)
		args = [
			config.MAX_DOCUMENT_LENGTH,
			config.MENTION_SIZE,
			self.num_types,
			self.embedding.vocab_size,
			self.embedding.embedding_dim,
			self.embedding.position_size,
			self.embedding.embedding,
			np.random.random_sample((self.embedding.position_size, self.hparams.wpe_dim)),
			self.type_info,
			self.hparams
			]
		if "nfetc" in self.model_name:
			return NFETC(*args)
		else:
			raise AttributeError("Invalid model name!")

	def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
		for k, v in sorted(d.items()):
			if isinstance(v, dict):
				self.logger.info("%s%s:" % (prefix, k))
				self.print_param_dict(v, prefix+incr_prefix, incr_prefix)
			else:
				self.logger.info("%s%s: %s" % (prefix, k, v))
	
	def create_session(self):
		session_conf = tf.ConfigProto(
				intra_op_parallelism_threads=8,
				allow_soft_placement=True,
				log_device_placement=False)
		return tf.Session(config=session_conf)

	def cv(self):
		self.logger.info("="*50)
		self.logger.info("Params")
		self._print_param_dict(self.params_dict)
		self.logger.info("Results")
		self.logger.info("\t\tRun\t\tStep\t\tLoss\t\tPAcc\t\t\tEAcc")

		cv_loss = []
		cv_pacc = []
		cv_eacc = []
		for i in range(self.cv_runs):
			sess = self.create_session()
			sess.run(tf.global_variables_initializer())
			step, loss, pacc, eacc = self.model.fit(sess, self.train_set, self.valid_set)
			self.logger.info("\t\t%d\t\t%d\t\t%.3f\t\t%.3f\t\t%.3f" % (i+1, step, loss, pacc, eacc))
			cv_loss.append(loss)
			cv_pacc.append(pacc)
			cv_eacc.append(eacc)
			sess.close()

		self.loss = np.mean(cv_loss)
		self.pacc = np.mean(cv_pacc)
		self.eacc = np.mean(cv_eacc)

		self.logger.info("CV Loss: %.3f" % self.loss)
		self.logger.info("CV Partial Accuracy: %.3f" % self.pacc)
		self.logger.info("CV Exact Accuracy: %.3f" % self.eacc)
		self.logger.info("-" * 50)

class TaskOptimizer:
	def __init__(self, model_name, data_name, cv_runs, max_evals, logger):
		self.model_name = model_name
		self.data_name = data_name
		self.cv_runs = cv_runs
		self.max_evals = max_evals
		self.logger = logger
		self.model_param_space = ModelParamSpace(self.model_name)

	def _obj(self, param_dict):
		param_dict = self.model_param_space._convert_into_param(param_dict)
		self.task = Task(self.model_name, self.data_name, self.cv_runs, param_dict, self.logger)
		self.task.cv()
		tf.reset_default_graph()
		ret = {
			"loss": self.task.loss,
			"attachments": {
				"pacc": self.task.pacc,
				"eacc": self.task.eacc,
			},
			"status": STATUS_OK
		}
		return ret

	def run(self):
		trials = Trials()
		best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
		best_params = space_eval(self.model_param_space._build_space(), best)
		best_params = self.model_param_space._convert_into_param(best_params)
		trial_loss = np.asarray(trials.losses(), dtype=float)
		best_ind = np.argmin(trial_loss)
		best_loss = trial_loss[best_ind]
		best_pacc = trials.trial_attachments(trials.trials[best_ind])["pacc"]
		best_eacc = trials.trial_attachments(trials.trials[best_ind])["eacc"]
		self.logger.info("-"*50)
		self.logger.info("Best Loss: %.3f\n with Parital Accuracy %.3f and Exact Accuracy %.3f" % (best_loss, best_pacc, best_eacc))
		self.logger.info("Best Param:")
		self._print_param_dict(best_params)
		self.logger.info("-"*50)

def parse_args(parser):
	parser.add_option("-m", "--model", type="string", dest="model_name")
	parser.add_option("-d", "--data", type="string", dest="data_name")
	parser.add_option("-e", "--eval", type="int", dest="max_evals", default=100)
	parser.add_option("-c", "--cv_runs", type="int", dest="cv_runs", default=3)
	options, args = parser.parse_args()
	return options, args

def main(options):
	time_str = datetime.datetime.now().isoformat()
	logname = "[Model@%s]_[Data@%s]_%s.log" % (options.model_name, options.data_name, time_str)
	logger = logging_utils._get_logger(config.LOG_DIR, logname)
	optimizer = TaskOptimizer(options.model_name, options.data_name, options.cv_runs, options.max_evals, logger)
	optimizer.run()

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
