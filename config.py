# -------------------- PATH ---------------------

#ROOT_PATH = "/local/data2/pxu4/TypeClassification"
ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
WIKI_DATA_PATH = "%s/corpus/Wiki" % DATA_PATH
ONTONOTES_DATA_PATH = "%s/corpus/OntoNotes" % DATA_PATH

LOG_DIR = "%s/log" % ROOT_PATH
CHECKPOINT_DIR = "%s/checkpoint" % ROOT_PATH

# -------------------- DATA ----------------------

WIKI_ALL = "%s/all.txt" % WIKI_DATA_PATH 
WIKI_TRAIN = "%s/train.txt" % WIKI_DATA_PATH 
WIKI_VALID = "%s/dev.txt" % WIKI_DATA_PATH 
WIKI_TEST = "%s/test.txt" % WIKI_DATA_PATH 

WIKI_TYPE = "%s/type.pkl" % WIKI_DATA_PATH
WIKI_TRAIN_CLEAN = "%s/train_clean.tsv" % WIKI_DATA_PATH
WIKI_TEST_CLEAN = "%s/test_clean.tsv" % WIKI_DATA_PATH

ONTONOTES_ALL = "%s/all.txt" % ONTONOTES_DATA_PATH
ONTONOTES_TRAIN = "%s/train.txt" % ONTONOTES_DATA_PATH
ONTONOTES_VALID = "%s/dev.txt" % ONTONOTES_DATA_PATH
ONTONOTES_TEST = "%s/test.txt" % ONTONOTES_DATA_PATH

ONTONOTES_TYPE = "%s/type.pkl" % ONTONOTES_DATA_PATH
ONTONOTES_TRAIN_CLEAN = "%s/train_clean.tsv" % ONTONOTES_DATA_PATH
ONTONOTES_TEST_CLEAN = "%s/test_clean.tsv" % ONTONOTES_DATA_PATH

EMBEDDING_DATA = "%s/glove.840B.300d.txt" % DATA_PATH

# --------------------- PARAM -----------------------

MAX_DOCUMENT_LENGTH = 30

MENTION_SIZE = 15

WINDOW_SIZE = 10

RANDOM_SEED = 2017
