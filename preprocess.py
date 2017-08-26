import pandas as pd
import numpy as np
from optparse import OptionParser
from utils import pkl_utils, eval_utils
import config
from collections import defaultdict
import re
import os

def path_count(types):
	cnt = 0
	for a in types:
		flag = 1
		for b in types:
			if len(a) >= len(b):
				continue
			if (a == b[:len(a)]) and (b[len(a)] == "/"):
				flag = 0
		cnt += flag
	return cnt

def create_type_dict(infile, outfile):
	df = pd.read_csv(infile, sep="\t", names=["p1", "p2", "text", "type", "f"])
	size = df.shape[0]
	typeSet = set()
	freq = defaultdict(int)
	for i in range(size):
		types = df["type"][i].split()
		out_type = []
		for a in types:
			flag = True
			for b in types:
				if len(a) >= len(b):
					continue
				if (a == b[:len(a)]) and (b[len(a)] == "/"):
					flag = False
			if flag:
				out_type.append(a)
		typeSet.update(out_type)
		for t in types:
			freq[t] += 1
	
	type2id = {y:x for x,y in enumerate(typeSet)}
	typeDict = defaultdict(list)
	for a in type2id.keys():
		for b in type2id.keys():
			if len(a) >= len(b):
				continue
			if (a == b[:len(a)]) and (b[len(a)] == "/"):
				typeDict[a].append(b)
	pkl_utils._save(outfile, (type2id, typeDict))

def clear_text(text):
	text = text.replace("-LRB-", "``")
	text = text.replace("-RRB-", "''")
	text = text.replace("-LSB-", "[")
	text = text.replace("-RSB-", "]")
	text = text.replace("-LCB-", "{")
	text = text.replace("-RCB-", "}")

	return text.strip()

def preprocess(data_name, if_clean=False):
	if data_name == "wiki":
		raw_all_file = config.WIKI_ALL
		raw_train_file = config.WIKI_TRAIN
		raw_valid_file = config.WIKI_VALID
		raw_test_file = config.WIKI_TEST
		clean_train_file = config.WIKI_TRAIN_CLEAN
		clean_test_file = config.WIKI_TEST_CLEAN
		type_file = config.WIKI_TYPE
	elif data_name == "wikim":
		raw_all_file = config.WIKIM_ALL
		raw_train_file = config.WIKIM_TRAIN
		raw_valid_file = config.WIKIM_VALID
		raw_test_file = config.WIKIM_TEST
		clean_train_file = config.WIKIM_TRAIN_CLEAN
		clean_test_file = config.WIKIM_TEST_CLEAN
		type_file = config.WIKIM_TYPE
	elif data_name == "ontonotes":
		raw_all_file = config.ONTONOTES_ALL
		raw_train_file = config.ONTONOTES_TRAIN
		raw_valid_file = config.ONTONOTES_VALID
		raw_test_file = config.ONTONOTES_TEST
		clean_train_file = config.ONTONOTES_TRAIN_CLEAN
		clean_test_file = config.ONTONOTES_TEST_CLEAN
		type_file = config.ONTONOTES_TYPE
	else:
		raise AttributeError("Invalid data name!")
		
	if not os.path.exists(type_file):
		create_type_dict(raw_all_file, type_file)
	type2id, typeDict = pkl_utils._load(type_file)

	df_train = pd.read_csv(raw_train_file, sep="\t", names=["p1", "p2", "text", "type", "f"])
	df_valid = pd.read_csv(raw_valid_file, sep="\t", names=["p1", "p2", "text", "type", "f"])
	df = pd.concat((df_train, df_valid), ignore_index=True)
	size = df.shape[0]
	outfile = open(clean_train_file, "w")
	for i in range(size):
		p1 = df["p1"][i]
		p2 = df["p2"][i]
		text = df["text"][i]
		types = df["type"][i].split()
		if (not path_count(types) == 1) and if_clean:
			continue

		text = clear_text(text)
		tokens = text.split()
		if p1 >= len(tokens):
			continue
		mention = " ".join(tokens[p1:p2])

		if p1 == 0:
			mention = "<PAD> " + mention
		else:
			mention = tokens[p1-1] + " " + mention
		if p2 >= len(tokens):
			mention = mention + " <PAD>"
		else:
			mention = mention + " " + tokens[p2]

		offset = max(0, p1-config.WINDOW_SIZE)
		text = " ".join(tokens[offset:min(len(tokens), p2+config.WINDOW_SIZE-1)])
		p1 -= offset
		p2 -= offset

		out_type = []
		for a in types:
			flag = True
			for b in types:
				if len(a) >= len(b):
					continue
				if (a == b[:len(a)]) and (b[len(a)] == "/"):
					flag = False
			if flag:
				out_type.append(a)

		if len(out_type) > 0:
			outfile.write("%d\t%d\t%s\t%s\t%s\n" % (p1, p2, text, mention, " ".join(out_type)))
	outfile.close()

	df = pd.read_csv(raw_test_file, sep="\t", names=["p1", "p2", "text", "type", "f"])
	size = df.shape[0]
	outfile = open(clean_test_file, "w")
	for i in range(size):
		p1 = df["p1"][i]
		p2 = df["p2"][i]
		text = df["text"][i]
		types = df["type"][i].split()

		text = clear_text(text)
		tokens = text.split()
		if p1 >= len(tokens):
			continue
		mention = " ".join(tokens[p1:p2])

		if p1 == 0:
			mention = "<PAD> " + mention
		else:
			mention = tokens[p1-1] + " " + mention
		if p2 >= len(tokens):
			mention = mention + " <PAD>"
		else:
			mention = mention + " " + tokens[p2]

		offset = max(0, p1-config.WINDOW_SIZE)
		text = " ".join(tokens[offset:min(len(tokens), p2+config.WINDOW_SIZE-1)])
		p1 -= offset
		p2 -= offset

		out_type = []
		for a in types:
			flag = True
			for b in types:
				if len(a) >= len(b):
					continue
				if (a == b[:len(a)]) and (b[len(a)] == "/"):
					flag = False
			if flag:
				out_type.append(a)

		outfile.write("%d\t%d\t%s\t%s\t%s\n" % (p1, p2, text, mention, " ".join(types)))
	outfile.close()

def parse_args(parser):
	parser.add_option("-d", "--data_name", type="string", dest="data_name")
	parser.add_option("-c", default=False, action="store_true", dest="if_clean")

	(options, args) = parser.parse_args()
	return options, args

def main(options):
	preprocess(options.data_name, options.if_clean)

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)

