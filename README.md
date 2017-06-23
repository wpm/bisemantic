# Text Pair Classification

Bisemantic learns semantic relationships between pairs of text.


## Installation

This tools uses the [spaCy](https://spacy.io/) text natural language processing tools.
It may be necessary to install spaCy's English language text model with a command like `python -m spacy download en` 
before running.
See spaCy's [models documentation](https://spacy.io/docs/usage/models) for more information.


## Running

Run this tool with the command `bisemantic`.
Subcommands enable you to train and use models, and partition data into cross-validation sets.
Run `bisemantic --help` for details about specific commands.

Data takes the form of comma-separated-value documents.
Training data has the columns `text1`, `text2`, and `label`.
Test data takes the same form minus the `label` column.
Command line options allow you to read files with different formatting.


## Classifier Model

Text pair classification is framed as a supervised learning problem.
The sample is a pair of texts and the label is a categorical class label.
The meaning of the class varies from data set to data set but usually represents some kind of semantic relationship 
between the two texts.

[GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the texts into matrices of size
_(tokens, 300)_, clipping or padding the first dimension as needed.
An optionally bidirectional shared LSTM converts these embeddings to single vectors, _r<sub>1</sub>_ and _r<sub>2</sub>_ which are then concatenated
into the vector [_r<sub>1</sub>, r<sub>2</sub>, r<sub>1</sub> · r<sub>2</sub>, (r<sub>1</sub> - r<sub>2</sub>)<sup>2</sup>_].
A single-layer perceptron maps this vector to a softmax prediction over the labels.


## Example Uses

Bisemantic can be used for tasks like question de-duplication or textual entailment.

### Question Deduplication

The [Quora question pair corpus](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) contains  contains
pairs of questions annotated as either asking the same thing or not.

Bisemantic creates a model similar to that described in Homma et al. ["Detecting Duplicate Questions with Deep Learning"](https://web.stanford.edu/class/cs224n/reports/2748045.pdf) and Addair ["Duplicate Question Pair Detection with Deep Learning"](https://web.stanford.edu/class/cs224n/reports/2759336.pdf).
The following command can be used to train a model on the `train.csv` file in this data set.

    bisemantic train train.csv \
        --text-1-name question1 --text-2-name question2 --label-name is_duplicate \
        --validation-fraction 0.2 --maximum-tokens 150 --dropout 0.5 --batch-size 1024 --units 256 \
        --model-directory-name quora.model

This achieved validation split accuracy of 83.49% after 10 epochs of training. 

### Textual Entailment

The [Stanford Natural Language Inference corpus](https://nlp.stanford.edu/projects/snli/) is a corpus for the
recognizing textual entailment (RTE) task.
It labels "text" sentences as either entailing, contradicting, or being neutral with respect to a "hypothesis" sentence.

Bisemantic creates a model similar to that described in Bowman et al. ["A large annotated corpus for learning natural language inference"](https://nlp.stanford.edu/pubs/snli_paper.pdf).
The following command can be used to train a model on the `train snli_1.0_train.txt` and `snli_1.0_dev.txt` files in
this data set.

    bisemantic train snli_1.0_train.txt \
   			--text-1-name sentence1 --text-2-name sentence2 --label-name gold_label --index-name pairID \
			--invalid-labels "-" --not-comma-delimited \
			--validation-set snli_1.0_dev.txt --dropout 0.5 --batch-size 1024 --units 256 \
			--epochs 20 --model-directory-name snli.model

This achieved development set accuracy of 79.63% after 11 epochs of training.