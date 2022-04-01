import re
import spacy
import torch
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.keyedvectors import KeyedVectors as word2vec
from sklearn.feature_extraction.text import CountVectorizer


class Word2Vec(object):
    '''word2vec model (pre-trained weights) & BOW generator

    INTERFACE
    ---------
    self.__init__
      path [str] path of pre-trained word2vec model
      limit [int] maximum number of word-vectors
      stop_words [bool] remove built-in stop word (sklearn, gensim & spacy)

    self.corpus2bows => bag of words [torch.tensor, N * V]
      corpus [list] list of docs[str]

    self.weight => pre-trained weights [torch.tensor, V * E]
    self.V => # of words
    self.E => # of embedding layer dimensions
    self.inv_vocab => inversed vocabulary dict

    REFERENCE
    ---------
    Efficient Estimation of Word Representations in Vector Space [Mikolov et al. 2013]
    Distributed Representations of Words and Phrases and their Compositionality [Mikolov et al. 2013]
    Linguistic Regularities in Continuous Space Word Representations [Mikolov et al. 2013]
    '''
    def __init__(self, path, limit, stop_words=False, binary=True):
        if stop_words:
            # built-in stop word list
            sklearn_stopwords = CountVectorizer(stop_words='english').get_stop_words()
            gesim_stopwords = STOPWORDS
            spacy_stopwords = spacy.load('en_core_web_sm').Defaults.stop_words
            # stop words & meaningless special characters
            word_filter = lambda x: not (x.lower() in sklearn_stopwords or \
                                         x.lower() in gesim_stopwords or \
                                         x.lower() in spacy_stopwords or \
                                         re.search(r"[0-9#$%^&*()_+<>']", x.lower()) or \
                                         len(x) <= 2)
            model = word2vec.load_word2vec_format(path, binary=binary, limit=int(limit*1.5))
            keys = list(filter(word_filter, model.index_to_key))
            limit = model.key_to_index[keys[limit]] # update limit
            model = word2vec.load_word2vec_format(path, binary=binary, limit=limit)
            keys = list(filter(word_filter, model.index_to_key))
            # new vocabulary and word2vec weights
            key_to_index = dict(zip(keys, range(len(keys))))
            weight = model.vectors[[model.key_to_index[i] for i in keys], :]
        else:
            model = word2vec.load_word2vec_format(path, binary=binary, limit=limit)
            key_to_index, weight = model.key_to_index, model.vectors
        self._pretrained_weight = torch.FloatTensor(weight)
        self._bow_generator = CountVectorizer(vocabulary=key_to_index)
        self._inv_vocab = {v: k for k, v in self._bow_generator.vocabulary.items()}
        print(f'[Word2Vec]\nVocabulary Size: {self.V}\nEmbedding Size: {self.E}')

    def corpus2bows(self, corpus):
        bows = torch.FloatTensor(self._bow_generator.transform(corpus).toarray())
        return bows

    @property
    def weight(self):
        return self._pretrained_weight

    @property
    def V(self):
        return self._pretrained_weight.shape[0]

    @property
    def E(self):
        return self._pretrained_weight.shape[1]

    @property
    def inv_vocab(self):
        return self._inv_vocab