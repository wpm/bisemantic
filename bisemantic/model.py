import itertools
import logging

import numpy as np
import pandas as pd
import spacy as spacy

from bisemantic import text_1, text_2, label


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
    Convert text into text embeddings.

    :param text_pairs: text pairs
    :type text_pairs: pandas.DataFrame
    :param maximum_tokens: the longest string of tokens to embed
    :type maximum_tokens: int
    :return: embedding matrices for the text pairs
    :rtype: list(numpy.array)
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
    return text_embedding_matrices


def train(training_data, validation_data):
    logging.info("Train on %d text pair samples" % len(training_data))


text_parser = None


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
