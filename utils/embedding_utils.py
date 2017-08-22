import numpy as np
import gensim

class Embedding:
	def __init__(self, f, corpus, max_document_length, mention_size):
		if ".txt" in f:
			model = gensim.models.KeyedVectors.load_word2vec_format(f, binary=False)
		else:
			model = gensim.models.KeyedVectors.load_word2vec_format(f, binary=True)

		wordSet = set(['"'])
		for sen in corpus:
			words = sen.split()
			for w in words:
				if w in model:
					wordSet.add(w)

		vocab_size = len(wordSet)
		print("%d unique tokens have been found!" % vocab_size)
		embedding_dim = model.syn0.shape[1]
		word2id = {"<PAD>":0}
		id2word = {0:"<PAD>"}
		word2id = {"<UNK>":1}
		id2word = {1:"<UNK>"}
		embedding = np.zeros((vocab_size+2, embedding_dim))

		np.random.seed(0)
		#embedding[0, :] = np.random.uniform(-1, 1, embedding_dim)
		embedding[1, :] = np.random.uniform(-1, 1, embedding_dim)
		for i, word in enumerate(wordSet):
			word2id[word] = i+2
			id2word[i+2] = word
			embedding[i+2, :] = model[word]

		self.vocab_size = vocab_size + 2
		self.embedding_dim = embedding_dim
		self.word2id = word2id
		self.id2word = id2word
		self.embedding = embedding
		self.max_document_length = max_document_length
		self.position_size = self.max_document_length * 2 + 1
		self.mention_size = mention_size
	
	def _text_transform(self, s, maxlen):
		if not isinstance(s, str):
			s = ""
		words = s.split()
		vec = []
		for w in words:
			if w == "''":
				w = '"'
			if w in self.word2id:
				vec.append(self.word2id[w])
			else:
				#vec.append(np.random.choice(self.vocab_size-1, 1)[0]+1)
				vec.append(1)
		for i in range(len(words), maxlen):
			vec.append(0)
		return vec[:maxlen]

	def _len_transform(self, s, maxlen):
		if not isinstance(s, str):
			s = ""
		length = len(s.split())
		return min(length, maxlen)

	def text_transform1(self, s):
		return self._text_transform(s, self.max_document_length)

	def len_transform1(self, s):
		return self._len_transform(s, self.max_document_length)

	def text_transform2(self, s):
		return self._text_transform(s, self.mention_size)

	def len_transform2(self, s):
		return self._len_transform(s, self.mention_size)

	def position_transform(self, s):
		x, y = s[0], s[1]
		y -= 1
		vec = []
		for i in range(self.max_document_length):
			if i < x:
				vec.append(i-x)
			elif i > y:
				vec.append(i-y)
			else:
				vec.append(0)
		vec = [p+self.max_document_length for p in vec]
		return vec
