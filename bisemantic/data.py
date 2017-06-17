"""
Parse text and represent it as embedding matrices.
"""
import math
from itertools import cycle

import numpy as np
import spacy
from pandas import DataFrame
from toolz import partition_all

from bisemantic import logger, text_1, text_2, label


class TextPairEmbeddingGenerator(object):
    def __init__(self, data, maximum_tokens=None, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = math.ceil(len(self) / self.batch_size)
        self._labeled = label in self.data.columns
        if maximum_tokens is None:
            m1 = max(len(document) for document in parse_documents(self.data[text_1]))
            m2 = max(len(document) for document in parse_documents(self.data[text_2]))
            maximum_tokens = max(m1, m2)
        self.maximum_tokens = maximum_tokens

    def __len__(self):
        """
        :return: number of samples in the data
        :rtype: int
        """
        return len(self.data)

    def __repr__(self):
        return "%s: %d samples, batch size %d, maximum tokens %s" % (
            self.__class__.__name__, len(self), self.batch_size, self.maximum_tokens)

    def __call__(self):
        for batch_data in cycle(self._batches()):
            yield self._embed_batch(batch_data)

    def _batches(self):
        """
        Partition the data into consecutive data sets of the specified batch size.

        :return: batched data
        :rtype: DataFrame iterator
        """
        t1 = partition_all(self.batch_size, self.data[text_1])
        t2 = partition_all(self.batch_size, self.data[text_2])
        if self._labeled:
            l = partition_all(self.batch_size, self.data[label])
            batches = zip(t1, t2, l)
        else:
            batches = zip(t1, t2)
        for batch in batches:
            if self._labeled:
                columns = [text_1, text_2, label]
            else:
                columns = [text_1, text_2]
            yield DataFrame(dict(zip(columns, batch)), columns=columns)

    def _embed_batch(self, batch_data):
        batch = [self._embed_text_set(batch_data[text_1]), self._embed_text_set(batch_data[text_2])]
        if self._labeled:
            batch = (batch, batch_data[label])
        return batch

    def _embed_text_set(self, text_set):
        embeddings = []
        for parsed_text in parse_documents(text_set):
            embeddings.append(self._pad(np.array([token.vector for token in parsed_text])))
        return np.stack(embeddings)

    def _pad(self, text_embedding):
        m = max(self.maximum_tokens - text_embedding.shape[0], 0)
        uniform_length_document_embedding = np.pad(text_embedding[:self.maximum_tokens], ((m, 0), (0, 0)), "constant")
        return uniform_length_document_embedding


def cross_validation_partitions(data, fraction, k):
    """
    Partition data into cross-validation sets.

    :param data: data set
    :type data: pandas.DataFrame
    :param fraction: percentage of data to use for training
    :type fraction: float
    :param k: number of cross-validation splits
    :type k: int
    :return: tuples of (training data, validation data) for each split
    :rtype: list(tuple(pandas.DateFrame, pandas.DateFrame))
    """
    logger.info("Cross validation %0.2f, %d partitions" % (fraction, k))
    n = int(fraction * len(data))
    partitions = []
    for i in range(k):
        data = data.sample(frac=1)
        train = data[:n]
        validate = data[n:]
        partitions.append((train, validate))
    return partitions


def parse_documents(texts):
    """
    Create a set of parsed documents from a set of texts.

    Parsed documents are sequences of tokens whose embedding vectors can be looked up.

    :param texts: text documents to parse
    :type texts: sequence of strings
    :return: parsed documents
    :rtype: sequence of spacy.Doc
    """
    return _load_text_parser().pipe(texts)


def embedding_size():
    return _load_text_parser().vocab.vectors_length


# Singleton instance of text tokenizer and embedder.
text_parser = None


def _load_text_parser():
    global text_parser
    if text_parser is None:
        logger.debug("Load text parser")
        text_parser = spacy.load("en", tagger=None, parser=None, entity=None)
    return text_parser
