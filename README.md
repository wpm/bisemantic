# Text Pair Semantic Equivalence

This tool learns semantic equivalence relationships between pairs of text.
It can be used for tasks like question de-duplication or textual entailment.


## Installation

This tools uses the [spaCy](https://spacy.io/) text natural language processing tools.
It may be necessary to install spaCy's English language text model with a command like `python -m spacy download en` before running.
See spaCy's [models documentation](https://spacy.io/docs/usage/models) for more information.

## Semantic Equivalence

Semantic equivalence is framed as a supervised learning problem.
The sample is a pair of texts and the label is a boolean indication of whether they have a particular semantic relationship.
The exact nature of the relationship depends on the data set, for example "entails" or "is a duplicate question".

[GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the texts into matrices of size `(tokens, embedding size)`, clipping or padding the first dimension as needed.
A shared LSTM model converts these to single vectors, _r<sub>1</sub>_ and _r<sub>2</sub>_ which are then concatenated into the vector [_r<sub>1</sub>, r<sub>2</sub>, r<sub>1</sub> · r<sub>2</sub>_].
A single-layer perceptron is used to map this vector to the label.

The method is similar to that described in Homma et al. ["Detecting Duplicate Questions with Deep Learning"](https://web.stanford.edu/class/cs224n/reports/2748045.pdf).

## Running

Run this tool with the command `bisemantic`.
Run `bisemantic --help` for details about specific commands.

Data takes the form of comma-separated-value documents.
Training data has the columns `text1`, `text2`, and `label`.
Test data takes the same form minus the `label` column.