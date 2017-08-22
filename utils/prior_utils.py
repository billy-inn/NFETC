from utils import pkl_utils
import sys
sys.path.append("../")
import config
import numpy as np

def create_prior(type_info, alpha=1.0):
	type2id, typeDict = pkl_utils._load(type_info)
	num_types = len(type2id)
	prior = np.zeros((num_types, num_types))
	for x in type2id.keys():
		tmp = np.zeros(num_types)
		tmp[type2id[x]] = 1.0
		for y in typeDict[x]:
			tmp[type2id[y]] = alpha
		prior[:,type2id[x]] = tmp
	return prior

