import os

from gensim import utils
from gensim.models import Word2Vec

import gensim


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield utils.simple_preprocess(line)


def get_word2vec_instance():
    # model = gensim.models.KeyedVectors.load_word2vec_format('cc.he.300' + '.vec', binary=False, encoding='utf8')
    # model = Word2Vec(sentences=MySentences("tokenized"))
    model = gensim.models.KeyedVectors.load('word2vec.model')
    # model = gensim.models.Word2Vec(sentences=MySentences("tokenized")).intersect_word2vec_format('cc.he.300.bin', binary=True)
    word_vectors = model.wv
    # -- this saves space, if you plan to use only, but not to train, the model:
    del model
    return word_vectors
