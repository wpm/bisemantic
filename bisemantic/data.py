"""
Parse text and represent it as embedding matrices.
"""
import math
from itertools import cycle, islice

import numpy as np
import spacy

from bisemantic import logger, text_1, text_2, label


class UniformLengthEmbeddingGenerator(object):
    @classmethod
    def embed(cls, data, batch_size=32, maximum_tokens=None, block_size=100000,
              parser_threads=-1, parser_batch_size=1000):
        g = cls(data, batch_size, maximum_tokens, block_size, parser_threads, parser_batch_size)
        return g(), g.batches_per_epoch

    def __init__(self, data, batch_size=32, maximum_tokens=None, block_size=100000,
                 parser_threads=-1, parser_batch_size=1000):
        assert set(data.columns) in [{text_1, text_2, label}, {text_1, text_2}]
        self._has_labels = label in data.columns
        self.data = data
        self.batch_size = batch_size
        self.maximum_tokens = maximum_tokens
        self.block_size = block_size
        self.parser_threads = parser_threads
        self.parser_batch_size = parser_batch_size
        self.batches_per_epoch = math.ceil(len(self) / self.batch_size)
        self._cached_epoch = None
        if maximum_tokens is None:
            self.maximum_tokens = self._longest_text()

    def _longest_text(self):
        maximum_tokens = 0
        blocks_per_epoch = math.ceil(len(self) / self.block_size)
        for block in islice(self._blocks(), blocks_per_epoch):
            for embedded_text_set in self._parse(block)[:2]:
                m = max(embedded_text.shape[0] for embedded_text in embedded_text_set)
                maximum_tokens = max(m, maximum_tokens)
        return maximum_tokens

    def __len__(self):
        """
        :return: number of samples in the data set
        :rtype: int
        """
        return len(self.data)

    def __call__(self):
        for block in self._blocks():
            if self._cached_epoch is not None:
                # Embeddings for all the data fits in memory and have already been calculated.
                batches = self._cached_epoch
            else:
                embedded_text_set_1, embedded_text_set_2, labels = self._parse(block)
                batches = []
                if self._has_labels:
                    batch_generator = self._batches(embedded_text_set_1, embedded_text_set_2, labels)
                    for embedded_batch_1, embedded_batch_2, labels_batch in batch_generator:
                        embedded_matrix_1 = self._embedding_matrix(embedded_batch_1)
                        embedded_matrix_2 = self._embedding_matrix(embedded_batch_2)
                        batches.append(([embedded_matrix_1, embedded_matrix_2], labels_batch))
                else:
                    batch_generator = self._batches(embedded_text_set_1, embedded_text_set_2)
                    for embedded_batch_1, embedded_batch_2 in batch_generator:
                        embedded_matrix_1 = self._embedding_matrix(embedded_batch_1)
                        embedded_matrix_2 = self._embedding_matrix(embedded_batch_2)
                        batches.append([embedded_matrix_1, embedded_matrix_2])
                if len(self) <= self.block_size:
                    # Embeddings for all the data will fit in memory, so cache them.
                    self._cached_epoch = batches
            for batch in batches:
                yield batch

    def _blocks(self):
        """
        Loop eternally over consecutive blocks of data.

        If the block size is larger than the data, this will just repeatedly yield the entire data set.

        :return: iterator over slices of the input data frame
        :rtype: iterator over pandas.DataFrame
        """
        n = len(self.data)
        return cycle(self.data[i:i + self.block_size] for i in range(0, n, self.block_size))

    def _batches(self, *sequences):
        """
        Make one pass through a set of equal-length sequences, yielding consecutive batches.

        :param sequences:
        :type sequences: list of list
        :return: tuple of corresponding batches from each of the sequences
        :rtype: iterator over tuple
        """
        for i in range(0, len(sequences[0]), self.batch_size):
            yield tuple(sequence[i: i + self.batch_size] for sequence in sequences)

    def _parse(self, block):
        """
        Given text pairs and corresponding labels, convert the text pairs to embedding vectors.

        If no labels are present in the data set, None is returned for labels.

        :param block: text pairs and labels
        :type block: pandas.DataFrame
        :return: lists of embedded text sets for each element of the text pairs and the label corresponding labels
        :rtype: (list(numpy.array), list(numpy.array), numpy.array or None)
        """
        text_set_1, text_set_2 = block[text_1], block[text_2]
        if self._has_labels:
            labels = block[label]
        else:
            labels = None
        embedded_text_set_1 = self._embed_text_set(text_set_1)
        embedded_text_set_2 = self._embed_text_set(text_set_2)
        return embedded_text_set_1, embedded_text_set_2, labels

    def _embed_text_set(self, text_set):
        return [self._embed_text(parsed_text)
                for parsed_text in parse_documents(text_set, self.parser_threads, self.parser_batch_size)]

    @staticmethod
    def _embed_text(parsed_text):
        return np.array([token.vector for token in parsed_text])

    def _embedding_matrix(self, embedded_text_set):
        return np.stack(self._pad(text_embedding) for text_embedding in embedded_text_set)

    def _pad(self, text_embedding):
        m = max(self.maximum_tokens - text_embedding.shape[0], 0)
        uniform_length_document_embedding = np.pad(text_embedding[:self.maximum_tokens], ((m, 0), (0, 0)), "constant")
        return uniform_length_document_embedding


def cross_validation_partitions(data, fraction, k):
    """
    Partition data into of cross-validation sets.

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


def parse_documents(texts, parser_threads=-1, parser_batch_size=1000):
    """
    Create a set of parsed documents from a set of texts.

    Parsed documents are sequences of tokens whose embedding vectors can be looked up.

    :param texts: text documents to parse
    :type texts: sequence of strings
    :param parser_threads: number of parallel parsing threads
    :type parser_threads: int
    :param parser_batch_size: the number of texts to buffer
    :type parser_batch_size: int
    :return: parsed documents
    :rtype: sequence of spacy.Doc
    """
    logger.debug("Parse %d texts" % len(texts))
    return _load_text_parser().pipe(texts, n_threads=parser_threads, batch_size=parser_batch_size)


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
