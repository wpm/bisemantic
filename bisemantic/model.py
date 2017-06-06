import itertools
import logging

import numpy as np
import pandas as pd
import spacy
from keras.engine import Model, Input
from keras.layers import LSTM, multiply, Lambda, add, concatenate, Dense

from bisemantic import text_1, text_2, label


class TextualEquivalenceModel(Model):
    @classmethod
    def train(cls, training_data, lstm_units, epochs, validation_data=None):
        """
        Train a model from aligned questions pairs in data frames.

        :param training_data:
        :type training_data: pandas.DataFrame
        :param lstm_units: number of hidden units in the LSTM
        :type lstm_units: int
        :param epochs: number of training epochs
        :type epochs: int
        :param validation_data: optional validation data
        :type validation_data: pandas.DataFrame or None
        :return: the trained model and its training history
        :rtype: (keras.engine.Model, keras.callbacks.History)
        """
        training_embeddings, maximum_tokens, embedding_size, training_labels = \
            TextualEquivalenceModel.embed_data_frame(training_data)
        model = cls(maximum_tokens, embedding_size, lstm_units)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(training_embeddings, training_labels, epochs=epochs, validation_data=validation_data)
        return model, history

    @staticmethod
    def embed_data_frame(data):
        embeddings, maximum_tokens = embed(data)
        labels = data[label]
        embedding_size = embeddings[0].shape[2]
        return embeddings, maximum_tokens, embedding_size, labels

    def __init__(self, maximum_tokens, embedding_size, lstm_units):
        self.maximum_tokens = maximum_tokens
        self.embedding_size = embedding_size
        self.lstm_units = lstm_units
        # Create the model geometry.
        input_shape = (maximum_tokens, embedding_size)
        # Input two sets of aligned question pairs.
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        # Apply the same LSTM to each.
        lstm = LSTM(lstm_units)
        r1 = lstm(input_1)
        r2 = lstm(input_2)
        # Concatenate the embeddings with their product and squared difference.
        p = multiply([r1, r2])
        negative_r2 = Lambda(lambda x: -x)(r2)
        d = add([r1, negative_r2])
        q = multiply([d, d])
        lstm_output = concatenate([r1, r2, p, q])
        # Use logistic regression to map the concatenated vector to the labels.
        logistic_regression = Dense(1, activation="sigmoid")(lstm_output)
        super().__init__([input_1, input_2], logistic_regression, "Textual equivalence")

    def __repr__(self):
        return "%s(LSTM units = %d, maximum tokens = %d, embedding size = %d)" % \
               (self.__class__.__name__, self.lstm_units, self.maximum_tokens, self.embedding_size)

    def fit(self, training_embeddings=None, training_labels=None, epochs=1, validation_data=None, **kwargs):
        if training_embeddings is not None:
            assert self._embedding_size_is_correct(training_embeddings)
            assert len(training_embeddings[0]) == len(training_labels)
        if validation_data is not None:
            validation_embeddings, _ = embed(validation_data, self.maximum_tokens)
            assert self._embedding_size_is_correct(validation_embeddings)
            validation_labels = validation_data[label]
            validation_data = (validation_embeddings, validation_labels)
        return super().fit(x=training_embeddings, y=training_labels, epochs=epochs, validation_data=validation_data,
                           **kwargs)

    def predict(self, test_data, batch_size=32, verbose=0):
        test_embeddings, _ = embed(test_data, self.maximum_tokens)
        probabilities = super().predict(test_embeddings, batch_size, verbose)
        return (probabilities > 0.5).astype('int32').reshape((-1,))

    def _embedding_size_is_correct(self, embeddings):
        return embeddings[0].shape[1:] == (self.maximum_tokens, self.embedding_size)


def load_data(filename):
    """
    Load a test or training data file.

    A data file is a CSV file. Test data files have the columns "text1", and "text2". Training data files additionally
    have a "label" column.

    This returns a data frame containing just these columns. Any rows with null values are dropped.

    :param filename: name of data file
    :type filename: str
    :return: data stored in the data file
    :rtype: pandas.DataFrame
    """
    data = pd.read_csv(filename)
    m = len(data)
    data = data.dropna()
    n = len(data)
    if m != n:
        logging.info("Dropped %d lines with null values from %s" % (m - n, filename))
    if label in data.columns:
        columns = [text_1, text_2, label]
    else:
        columns = [text_1, text_2]
    return data[columns]


def embed(text_pairs, maximum_tokens=None):
    """
    Convert text pairs into text embeddings.

    :param text_pairs: text pairs
    :type text_pairs: pandas.DataFrame
    :param maximum_tokens: the longest string of tokens to embed
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

    logging.info("Embed %d text pairs" % len(text_pairs))
    # Convert text to embedding vectors.
    text_sets = aligned_text_sets(text_pairs)
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
        logging.info("Load text parser")
        text_parser = spacy.load("en", tagger=None, parser=None, entity=None)
    return text_parser.pipe(texts, n_threads=n_threads)


text_parser = None


def aligned_text_sets(text_pairs):
    return (text_pairs[text] for text in [text_1, text_2])
