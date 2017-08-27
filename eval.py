from optparse import OptionParser
from task import Task
import logging
from utils import logging_utils
from model_param_space import param_space_dict
import datetime
import config

def parse_args(parser):
	parser.add_option("-m", "--model", dest="model_name", type="string")
	parser.add_option("-d", "--data", dest="data_name", type="string")
	parser.add_option("-r", "--runs", dest="runs", type="int", default=5)
	parser.add_option("-e", "--epoch", dest="epoch", default=False, action="store_true")
	parser.add_option("-s", "--save", dest="save", default=False, action="store_true")
	parser.add_option("-f", "--full", dest="full", default=False, action="store_true")
	options, args = parser.parse_args()
	return options, args

def main(options):
	if options.epoch:
		time_str = datetime.datetime.now().isoformat()
		logname = "Eval_[Model@%s]_[Data@%s]_%s.log" % (options.model_name, options.data_name, time_str)
		logger = logging_utils._get_logger(config.LOG_DIR, logname)
	else:
		logger = logging.getLogger()
		logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
	params_dict = param_space_dict[options.model_name]
	task = Task(options.model_name, options.data_name, options.runs, params_dict, logger)
	if options.save:
		task.save()
	else:
		if options.epoch:
			task.refit()
		else:
			task.evaluate(options.full)
	

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
