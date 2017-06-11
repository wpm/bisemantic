"""
Core model functionality
"""

import itertools
import logging
import math

import numpy as np
import spacy
from keras.engine import Model, Input
from keras.layers import LSTM, multiply, concatenate, Dense, Dropout
from keras.models import load_model

from bisemantic import text_1, text_2, label, logger


class TextualEquivalenceModel(object):
    @classmethod
    def train(cls, training_data, lstm_units, epochs, dropout=None, clip_tokens=None, validation_data=None):
        """
        Train a model from aligned questions pairs in data frames.

        :param training_data:
        :type training_data: pandas.DataFrame
        :param lstm_units: number of hidden units in the LSTM
        :type lstm_units: int
        :param epochs: number of training epochs
        :type epochs: int
        :param dropout:  dropout rate or None for no dropout
        :type dropout: float or None
        :param clip_tokens: maximum number of tokens to embed per sample
        :type clip_tokens: int
        :param validation_data: optional validation data
        :type validation_data: pandas.DataFrame or None
        :return: the trained model and its training history
        :rtype: (TextualEquivalenceModel, keras.callbacks.History)
        """

        # noinspection PyShadowingNames
        def embed_data_frame(data):
            embeddings, maximum_tokens = embed(data, clip_tokens)
            labels = data[label]
            embedding_size = embeddings[0].shape[2]
            return embeddings, maximum_tokens, embedding_size, labels

        training_embeddings, maximum_tokens, embedding_size, training_labels = embed_data_frame(training_data)
        model = cls.create(maximum_tokens, embedding_size, lstm_units, dropout)
        logger.info(model)
        history = model.fit(training_embeddings, training_labels, epochs=epochs, validation_data=validation_data)
        return model, history

    @classmethod
    def load(cls, filename):
        """
        Restore a model serialized by TextualEquivalenceModel.save.

        :param filename: file name
        :type filename: str
        :return: the restored model
        :rtype: TextualEquivalenceModel
        """
        return cls(load_model(filename))

    @classmethod
    def create(cls, maximum_tokens, embedding_size, lstm_units, dropout):
        """
        Create a model that detects semantic equivalence between text pairs.

        The text pairs are passed in as two aligned matrices of size
        (batch size, maximum embedding tokens, embedding size). They are created by the embed function.

        :param maximum_tokens: maximum number of embedded tokens
        :type maximum_tokens: int
        :param embedding_size: size of the embedding vector
        :type embedding_size: int
        :param lstm_units: number of hidden units in the shared LSTM
        :type lstm_units: int
        :param dropout:  dropout rate or None for no dropout
        :type dropout: float or None
        :return: the created model
        :rtype: TextualEquivalenceModel
        """
        # Create the model geometry.
        input_shape = (maximum_tokens, embedding_size)
        # Input two sets of aligned question pairs.
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        # Apply the same LSTM to each.
        lstm = LSTM(lstm_units, name="lstm")
        r1 = lstm(input_1)
        r2 = lstm(input_2)
        # Concatenate the embeddings with their product and squared difference.
        p = multiply([r1, r2])
        # Deserialization is broken for squared difference. See Keras issue 6827.
        # negative_r2 = Lambda(lambda x: -x)(r2)
        # d = add([r1, negative_r2])
        # q = multiply([d, d])
        # v = [r1, r2, p, q]
        v = [r1, r2, p]
        lstm_output = concatenate(v)
        if dropout is not None:
            lstm_output = Dropout(dropout, name="dropout")(lstm_output)
        # A single-layer perceptron maps the concatenated vector to the labels. See Addair "Duplicate Question Pair
        # Detection with Deep Learning".
        m = sum(t.shape[1].value for t in v)
        perceptron = Dense(math.floor(math.sqrt(m)))(lstm_output)
        logistic_regression = Dense(1, activation="sigmoid")(perceptron)
        model = Model([input_1, input_2], logistic_regression, "Textual equivalence")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return cls(model)

    def __init__(self, model):
        self.model = model

    @property
    def maximum_tokens(self):
        return self.model.input_shape[0][1]

    @property
    def embedding_size(self):
        return self.model.input_shape[0][2]

    @property
    def lstm_units(self):
        return self.model.get_layer("lstm").units

    @property
    def dropout(self):
        dropout = self.model.get_layer("dropout")
        if dropout is not None:
            dropout = dropout.rate
        return dropout

    def __repr__(self):
        if self.dropout is None:
            d = "No dropout"
        else:
            d = "dropout = %0.2f" % self.dropout
        return "%s(LSTM units = %d, maximum tokens = %d, embedding size = %d, %s)" % \
               (self.__class__.__name__, self.lstm_units, self.maximum_tokens, self.embedding_size, d)

    def parameters(self):
        return {"maximum_tokens": self.maximum_tokens,
                "embedding_size": self.embedding_size,
                "lstm_units": self.lstm_units,
                "dropout": self.dropout}

    def fit(self, training_embeddings=None, training_labels=None, epochs=1, validation_data=None):
        if training_embeddings is None:
            n = 0
        else:
            n = len(training_embeddings[0])
        logger.info("Train model: %d samples, %d epochs" % (n, epochs))
        if training_embeddings is not None:
            assert self._embedding_size_is_correct(training_embeddings)
            assert len(training_embeddings[0]) == len(training_labels)
        if validation_data is not None:
            validation_embeddings, _ = embed(validation_data, self.maximum_tokens)
            assert self._embedding_size_is_correct(validation_embeddings)
            validation_labels = validation_data[label]
            validation_data = (validation_embeddings, validation_labels)
        verbose = {logging.INFO: 2, logging.DEBUG: 1}.get(logger.getEffectiveLevel(), 0)
        return self.model.fit(x=training_embeddings, y=training_labels, epochs=epochs, validation_data=validation_data,
                              verbose=verbose)

    def predict(self, test_data):
        test_embeddings, _ = embed(test_data, self.maximum_tokens)
        probabilities = self.model.predict(test_embeddings)
        return (probabilities > 0.5).astype('int32').reshape((-1,))

    def _embedding_size_is_correct(self, embeddings):
        return embeddings[0].shape[1:] == (self.maximum_tokens, self.embedding_size)

    def save(self, filename):
        self.model.save(filename)


def embed(text_pairs, maximum_tokens=None):
    """
    Convert text pairs into text embeddings.

    :param text_pairs: text pairs
    :type text_pairs: pandas.DataFrame
    :param maximum_tokens: maximum number of tokens to embed per sample
    :type maximum_tokens: int
    :return: embedding matrices for the text pairs, the maximum number of tokens in the pairs
    :rtype: (list(numpy.array), int)
    """

    def flatten(xs):
        return itertools.chain(*xs)

    def embed_text(parsed_text):
        return np.array([token.vector for token in parsed_text])

    def pad(text_embedding):
        m = max(maximum_tokens - text_embedding.shape[0], 0)
        uniform_length_document_embedding = np.pad(text_embedding[:maximum_tokens], ((m, 0), (0, 0)), "constant")
        return uniform_length_document_embedding

    logger.info("Embed %d text pairs" % len(text_pairs))
    # Convert text to embedding vectors.
    text_sets = (text_pairs[text] for text in [text_1, text_2])
    parsed_text_sets = [list(parse_documents(text_set)) for text_set in text_sets]
    # Get the maximum number of tokens from the longest text if not specified.
    if maximum_tokens is None:
        maximum_tokens = max(len(parsed_text) for parsed_text in flatten(parsed_text_sets))
    # Build embeddings from the parsed texts.
    text_embedding_sets = [[embed_text(parsed_text) for parsed_text in parsed_text_set]
                           for parsed_text_set in parsed_text_sets]
    # Pad the embeddings to be of equal length.
    uniform_length_text_embedding_sets = []
    for text_embedding_set in text_embedding_sets:
        uniform_length_text_embeddings = [pad(text_embedding) for text_embedding in text_embedding_set]
        uniform_length_text_embedding_sets.append(uniform_length_text_embeddings)
    # Combine the embeddings into two matrices.
    text_embedding_matrices = [np.stack(uniform_length_text_embeddings) for
                               uniform_length_text_embeddings in uniform_length_text_embedding_sets]
    return text_embedding_matrices, maximum_tokens


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


def parse_documents(texts, n_threads=-1):
    """
    Create a set of parsed documents from a set of texts.

    Parsed documents are sequences of tokens whose embedding vectors can be looked up.

    :param texts: text documents to parse
    :type texts: sequence of strings
    :param n_threads: number of parallel parsing threads
    :type n_threads: int
    :return: parsed documents
    :rtype: sequence of spacy.Doc
    """
    global text_parser
    if text_parser is None:
        logger.debug("Load text parser")
        text_parser = spacy.load("en", tagger=None, parser=None, entity=None)
    return text_parser.pipe(texts, n_threads=n_threads)


# Singleton instance of text tokenizer and embedder.
text_parser = None
