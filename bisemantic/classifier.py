"""
The model that classifies text pairs
"""
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.engine import Model, Input
from keras.layers import LSTM, multiply, concatenate, Dense, Dropout, Lambda, add, Bidirectional
from keras.models import load_model

from bisemantic import logger, label
from bisemantic.data import TextPairEmbeddingGenerator, embedding_size


class TextPairClassifier(object):
    """
    A model that learns to assign labels to pairs of text.
    """

    @classmethod
    def train(cls, training_data, bidirectional, lstm_units, epochs, dropout=None, maximum_tokens=None,
              batch_size=2048, validation_data=None, model_directory=None):
        """
        Train a model from aligned text pairs in data frames.

        :param training_data: text pairs and labels
        :type training_data: pandas.DataFrame
        :param bidirectional: should the shared LSTM be bidirectional?
        :type bidirectional: bool
        :param lstm_units: number of hidden units in the LSTM
        :type lstm_units: int
        :param epochs: number of training epochs
        :type epochs: int
        :param dropout:  dropout rate or None for no dropout
        :type dropout: float or None
        :param maximum_tokens: maximum number of tokens to embed per sample
        :type maximum_tokens: int
        :param batch_size: number of samples per batch
        :type batch_size: int
        :param validation_data: optional validation data
        :type validation_data: pandas.DataFrame or None
        :param model_directory: directory in which to write model checkpoints
        :type model_directory: str or None
        :return: the trained model and its training history
        :rtype: (TextPairClassifier, keras.callbacks.History)
        """
        training = TextPairEmbeddingGenerator(training_data, batch_size=batch_size, maximum_tokens=maximum_tokens)
        model = cls.create(len(training.classes), training.maximum_tokens, embedding_size(), lstm_units, dropout,
                           bidirectional)
        if model_directory is not None:
            os.makedirs(model_directory)
            with open(cls.info_filename(model_directory), "w") as f:
                f.write("%s\n" % model)
        return cls._train(epochs, model, model_directory, training, validation_data)

    @classmethod
    def continue_training(cls, training_data, epochs, model_directory, batch_size=2048, validation_data=None):
        """
        Continue training a model that was already created by a previous training operation.

        :param training_data: text pairs and labels
        :type training_data: pandas.DataFrame
        :param epochs: number of training epochs
        :type epochs: int
        :param model_directory: directory in which to write model checkpoints
        :type model_directory: str or None
        :param batch_size: number of samples per batch
        :type batch_size: int
        :param validation_data: optional validation data
        :type validation_data: pandas.DataFrame or None
        :return: the trained model and its training history
        :rtype: (TextPairClassifier, keras.callbacks.History)
        """
        model = cls.load(cls.model_filename(model_directory))
        training = TextPairEmbeddingGenerator(training_data, maximum_tokens=model.maximum_tokens, batch_size=batch_size)
        return cls._train(epochs, model, model_directory, training, validation_data)

    @classmethod
    def _train(cls, epochs, model, model_directory, training, validation_data):
        logger.info(model)
        start = time.time()
        history = model.fit(training, epochs=epochs, validation_data=validation_data, model_directory=model_directory)
        training_time = str(timedelta(seconds=time.time() - start))
        training_history = cls._training_history(model_directory, training_time, training, history.history)
        return model, training_history

    @classmethod
    def _training_history(cls, model_directory, training_time, training, history):
        if model_directory is not None:
            training_history_filename = cls.training_history_filename(model_directory)
            if os.path.isfile(training_history_filename):
                training_history = TrainingHistory.load(training_history_filename)
            else:
                training_history = TrainingHistory()
            training_history.add_run(training_time, training, history)
            training_history.save(training_history_filename)
        else:
            training_history = TrainingHistory()
            training_history.add_run(training_time, training, history)
        return training_history

    @classmethod
    def load(cls, filename):
        """
        :param filename: file name
        :type filename: str
        :return: the restored model
        :rtype: TextPairClassifier
        """
        return cls(load_model(filename))

    @classmethod
    def load_from_model_directory(cls, model_directory):
        return cls.load(cls.model_filename(model_directory))

    # noinspection PyShadowingNames
    @classmethod
    def create(cls, classes, maximum_tokens, embedding_size, lstm_units, dropout, bidirectional):
        """
        Create a model that detects semantic equivalence between text pairs.

        The text pairs are passed in as two aligned matrices of size
        (batch size, maximum embedding tokens, embedding size). They are created by the embed function.

        :param classes: the number of distinct classes to categorize
        :type classes: int
        :param maximum_tokens: maximum number of embedded tokens
        :type maximum_tokens: int
        :param embedding_size: size of the embedding vector
        :type embedding_size: int
        :param lstm_units: number of hidden units in the shared LSTM
        :type lstm_units: int
        :param dropout:  dropout rate or None for no dropout
        :type dropout: float or None
        :param bidirectional: should the shared LSTM be bidirectional?
        :type bidirectional: bool
        :return: the created model
        :rtype: TextPairClassifier
        """
        # Create the model geometry.
        input_shape = (maximum_tokens, embedding_size)
        # Input two sets of aligned text pairs.
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)
        # Apply the same LSTM to each.
        if bidirectional:
            lstm = Bidirectional(LSTM(lstm_units), name="lstm")
        else:
            lstm = LSTM(lstm_units, name="lstm")
        r1 = lstm(input_1)
        r2 = lstm(input_2)
        # Concatenate the embeddings with their product and squared difference.
        p = multiply([r1, r2])
        negative_r2 = Lambda(lambda x: -x)(r2)
        d = add([r1, negative_r2])
        q = multiply([d, d])
        v = [r1, r2, p, q]
        lstm_output = concatenate(v)
        if dropout is not None:
            lstm_output = Dropout(dropout, name="dropout")(lstm_output)
        # A single-layer perceptron maps the concatenated vector to the labels. See Addair "Duplicate Question Pair
        # Detection with Deep Learning".
        m = sum(t.shape[1].value for t in v)
        perceptron = Dense(math.floor(math.sqrt(m)), activation="relu")(lstm_output)
        logistic_regression = Dense(classes, activation="softmax", name="softmax")(perceptron)
        model = Model([input_1, input_2], logistic_regression, "Textual equivalence")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
        lstm = self.model.get_layer("lstm")
        if isinstance(lstm, Bidirectional):
            lstm = lstm.layer
        return lstm.units

    @property
    def bidirectional(self):
        return isinstance(self.model.get_layer("lstm"), Bidirectional)

    @property
    def dropout(self):
        dropout = self.model.get_layer("dropout")
        if dropout is not None:
            dropout = dropout.rate
        return dropout

    @property
    def classes(self):
        return self.model.get_layer("softmax").units

    def __repr__(self):
        if self.dropout is None:
            d = "No dropout"
        else:
            d = "dropout = %0.2f" % self.dropout
        s = "%s(" % self.__class__.__name__
        if self.bidirectional:
            s += "bidirectional, "
        return s + "classes = %d, LSTM units = %d, maximum tokens = %d, embedding size = %d, %s)" % \
                   (self.classes, self.lstm_units, self.maximum_tokens, self.embedding_size, d)

    def fit(self, training, epochs=1, validation_data=None, model_directory=None):
        logger.info("Train model: %d samples, %d epochs, batch size %d" % (len(training), epochs, training.batch_size))
        if validation_data is not None:
            g = TextPairEmbeddingGenerator(validation_data, maximum_tokens=self.maximum_tokens)
            validation_embeddings, validation_steps = g(), g.batches_per_epoch
        else:
            validation_embeddings = validation_steps = None
        verbose = {logging.INFO: 2, logging.DEBUG: 1}.get(logger.getEffectiveLevel(), 0)
        if model_directory is not None:
            if validation_data is not None:
                monitor = "val_loss"
            else:
                monitor = "loss"
            callbacks = [
                ModelCheckpoint(filepath=self.model_filename(model_directory), monitor=monitor, save_best_only=True,
                                verbose=verbose)]
        else:
            callbacks = None
        logger.info("Start training")
        return self.model.fit_generator(generator=training(), steps_per_epoch=training.batches_per_epoch, epochs=epochs,
                                        validation_data=validation_embeddings, validation_steps=validation_steps,
                                        callbacks=callbacks, verbose=verbose)

    def predict(self, test_data, batch_size=2048):
        g = TextPairEmbeddingGenerator(test_data, maximum_tokens=self.maximum_tokens, batch_size=batch_size)
        probabilities = self.model.predict_generator(generator=g(), steps=g.batches_per_epoch)
        return pd.DataFrame(probabilities.reshape((len(test_data), self.classes)))

    def score(self, labeled_test_data, batch_size=2048):
        assert label in labeled_test_data
        g = TextPairEmbeddingGenerator(labeled_test_data, maximum_tokens=self.maximum_tokens, batch_size=batch_size)
        if not self.classes == len(g.classes):
            raise ValueError("Test data categories %s do not align with the %d labels in the model" % (
                list(g.classes), self.classes))
        metrics = self.model.evaluate_generator(generator=g(), steps=g.batches_per_epoch)
        return list(zip(self.model.metrics_names, metrics))

    @staticmethod
    def info_filename(model_directory):
        return os.path.join(model_directory, "model.info.txt")

    @staticmethod
    def model_filename(model_directory):
        return os.path.join(model_directory, "model.h5")

    @staticmethod
    def training_history_filename(model_directory):
        return os.path.join(model_directory, "training-history.json")


class TrainingHistory(object):
    """
    Record of all the training runs made on a given model. This records the training date, the size of the sample, and
    the training and validation scores.
    """

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            return cls(json.load(f))

    def __init__(self, runs=None):
        self.runs = runs or []

    def __repr__(self):
        return "Training history, %d runs" % (len(self.runs))

    def add_run(self, training_time, training, history):
        self.runs.append({"training-time": training_time,
                          "training": str(training),
                          "history": history,
                          "run-date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.runs, f, sort_keys=True, indent=4, separators=(",", ": "))

    def latest_run_summary(self):
        lines = []
        if self.runs:
            last_run = self.runs[-1]
            history = last_run["history"]
            lines.append("Training time %s" % last_run["training-time"])
            if "val_loss" in history:
                metric = "val_loss"
            else:
                metric = "loss"
            i = history[metric].index(min(history[metric]))
            lines.append("Best epoch: %d" % (i + 1))
            lines.append("Training: accuracy=%0.4f, loss=%0.4f" % (history["acc"][i], history["loss"][i]))
            if "val_loss" in history:
                lines.append("Validation: accuracy=%0.4f, loss=%0.4f" % (history["val_acc"][i], history["val_loss"][i]))
        return "\n".join(lines)
