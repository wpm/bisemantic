import logging
import pandas as pd


def load_data(filename):
    data = pd.read_csv(filename)
    m = len(data)
    data = data.dropna()
    n = len(data)
    if m != n:
        logging.info("Dropped %d lines with null values from %s" % (m - n, filename))
    return data


def embed(data):
    pass


def train(training_data, validation_data):
    pass
