"""
Parse text and represent it as embedding matrices.
"""
import math
from itertools import cycle

import numpy as np
import pandas as pd
import spacy
from pandas import DataFrame
from toolz import partition_all

from bisemantic import logger, text_1, text_2, label


class TextPairEmbeddingGenerator(object):
    def __init__(self, data, maximum_tokens=None, batch_size=2048):
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = math.ceil(len(self) / self.batch_size)
        self._labeled = label in self.data.columns
        if self._labeled:
            self.data.loc[:, label] = self.data.loc[:, label].astype("category")
        if maximum_tokens is None:
            m1 = max(len(document) for document in parse_texts(self.data[text_1]))
            m2 = max(len(document) for document in parse_texts(self.data[text_2]))
            maximum_tokens = max(m1, m2)
        self.maximum_tokens = maximum_tokens
        logger.info(self)

    def __len__(self):
        """
        :return: number of samples in the data
        :rtype: int
        """
        return len(self.data)

    def __repr__(self):
        s = "%s: %d samples" % (self.__class__.__name__, len(self))
        if self._labeled:
            s += ", classes %s" % self.classes
        return s + ", batch size %d, maximum tokens %s" % (self.batch_size, self.maximum_tokens)

    @property
    def classes(self):
        """
        :return: the classes used to label the data or None is the data is unlabeled
        :rtype: list or None
        """
        if self._labeled:
            return list(self.data[label].cat.categories)
        else:
            return None

    def __call__(self):
        """
        Iterate eternally over the data yielding batches.

        :return: batches of embedded text matrices and optionally labels
        :rtype: [numpy.array, numpy.array] or ([numpy.array, numpy.array], numpy.array)
        """
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
            l = partition_all(self.batch_size, self.data[label].cat.codes)
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
        for parsed_text in parse_texts(text_set):
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


def data_file(filename, n=None, index=None, text_1_name=None, text_2_name=None, label_name=None,
              invalid_labels=None, comma_delimited=True):
    """
    Load a test or training data file.

    A data file is a CSV file. Any rows with null values in the columns of interest or with optional invalid label
    values are dropped. The file may optionally be clipped to a specified length.

    Rename columns in an input data frame to the ones bisemantic expects. Drop unused columns. If an argument is not
    None the corresponding column must already be in the raw data.

    :param filename: name of data file
    :type filename: str
    :param n: number of samples to limit to or None to use the entire file
    :type n: int or None
    :param index: optional name of the index column
    :type index: str or None
    :param text_1_name: name of column in data that should be mapped to text1
    :type text_1_name: str or None
    :param text_2_name: name of column in data that should be mapped to text2
    :type text_2_name: str or None
    :param label_name: name of column in data that should be mapped to label
    :type label_name: str or None
    :param invalid_labels: disallowed label values
    :type invalid_labels: list of str
    :param comma_delimited: is the data file comma-delimited?
    :type comma_delimited: bool
    :return: data frame of the desired size containing just the needed columns
    :rtype: pandas.DataFrame
    """
    data = load_data_file(filename, index, comma_delimited).head(n)
    data = fix_columns(data, text_1_name, text_2_name, label_name)
    m = len(data)
    data = data.dropna()
    if invalid_labels:
        data = data[~data[label].isin(invalid_labels)]
    n = len(data)
    if m != n:
        logger.info("Dropped %d samples with null values from %s" % (m - n, filename))
    return data


def load_data_file(filename, index=None, comma_delimited=True):
    """
    Load a CSV data file.

    :param filename: name of data file
    :type filename: str
    :param index: optional name of the index column
    :type index: str or None
    :param comma_delimited: is the data file comma-delimited?
    :type comma_delimited: bool
    :return: data stored in the data file
    :rtype: pandas.DataFrame
    """
    if comma_delimited:
        data = pd.read_csv(filename, index_col=index)
    else:
        # Have the Python parser figure out what the delimiter is.
        data = pd.read_csv(filename, index_col=index, sep=None, engine="python")
    return data


def fix_columns(data, text_1_name=None, text_2_name=None, label_name=None):
    """
    Rename columns in an input data frame to the ones bisemantic expects. Drop unused columns. If an argument is not
    None the corresponding column must already be in the raw data.

    :param data: raw data
    :type data: pandas.DataFrame
    :param text_1_name: name of column in data that should be mapped to text1
    :type text_1_name: str or None
    :param text_2_name: name of column in data that should be mapped to text2
    :type text_2_name: str or None
    :param label_name: name of column in data that should be mapped to label
    :type label_name: str or None
    :return: data frame containing just the needed columns
    :rtype: pandas.DataFrame
    """
    for name in [text_1_name, text_2_name, label_name]:
        if name is not None:
            if name not in data.columns:
                raise ValueError("Missing column %s" % name)
    data = data.rename(columns={text_1_name: text_1, text_2_name: text_2, label_name: label})
    if label in data.columns:
        columns = [text_1, text_2, label]
    else:
        columns = [text_1, text_2]
    return data[columns]


def parse_texts(texts):
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
