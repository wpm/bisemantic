# Bisemantic

Bisemantic uses deep learning to identify semantic relationships between pairs of text.
It uses a shared LSTM which maps two texts to representations in a common format which are then aligned with training
labels.


## Installation

This tools uses the [spaCy](https://spacy.io/) text natural language processing tools.
It may be necessary to install spaCy's English language text model with a command like `python -m spacy download en` 
before running.
See spaCy's [models documentation](https://spacy.io/docs/usage/models) for more information.


## Running

Run this tool with the command `bisemantic`.
Subcommands enable you to train and use models and partition data into cross-validation sets.
Run `bisemantic --help` for details about specific commands.

Input data takes the form of comma-separated-value documents.
Training data has the columns `text1`, `text2`, and `label`.
Test data takes the same form minus the `label` column.
Command line options allow you to read in files with different formatting.

Trained models are written to a directory that contains the following files:

* _model.info.text_: a human-readable description of the model and training parameters
* _training-history.json_: history of the training procedure, including the loss and accuracy for each epoch
* _model.h5_: serialization of the model structure and its weights

Weights from the epoch with the best loss score are saved in model.h5.

The model directory can be used to predict probability distributions over labels and score test sets.
Further training can be done using a model directory as a starting point.


## Classifier Model

Text pair classification is framed as a supervised learning problem.
The sample is a pair of texts and the label is a categorical class label.
The meaning of the class varies from data set to data set but usually represents some kind of semantic relationship 
between the two texts.

[GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the texts into matrices of size
_(maximum tokens, 300)_, clipping or padding the first dimension as needed.
If maximum tokens is not specified, the number of tokens in the longest text in the pairs is used.
An optionally bidirectional shared LSTM converts these embeddings to single vectors,
 _r<sub>1</sub>_ and _r<sub>2</sub>_, which are then concatenated
into the vector
_[r<sub>1</sub>, r<sub>2</sub>, r<sub>1</sub> · r<sub>2</sub>, (r<sub>1</sub> - r<sub>2</sub>)<sup>2</sup>]_.
A single-layer perceptron maps this vector to a softmax prediction over the labels.


## Example Uses

Bisemantic can be used for tasks like question de-duplication or textual entailment.

### Question Deduplication

The [Quora question pair corpus](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) contains
pairs of questions annotated as either asking the same thing or not.

Bisemantic creates a model similar to that described in \[[Homma et al.](https://web.stanford.edu/class/cs224n/reports/2748045.pdf)\]
and \[[Addair](https://web.stanford.edu/class/cs224n/reports/2759336.pdf)\].  
The following command can be used to train a model on the `train.csv` file in this data set.

    bisemantic train train.csv \
        --text-1-name question1 --text-2-name question2 \
        --label-name is_duplicate --index-name id \
        --validation-fraction 0.2 --batch-size 1024 \
        --maximum-tokens 75 --dropout 0.5 --units 256 --bidirectional \
        --model-directory-name quora.model

This achieved an accuracy of 83.71% on the validation split after 9 epochs of training. 

### Textual Entailment

The [Stanford Natural Language Inference corpus](https://nlp.stanford.edu/projects/snli/) is a corpus for the
recognizing textual entailment (RTE) task.
It labels a "text" sentence as either entailing, contradicting, or being neutral with respect to a "hypothesis"
sentence.

Bisemantic creates a model similar to that described in
\[[Bowman et al., 2015](https://nlp.stanford.edu/pubs/snli_paper.pdf)\].
The following command can be used to train a model on the `train snli_1.0_train.txt` and `snli_1.0_dev.txt` files in
this data set.

    bisemantic train snli_1.0_train.txt \
   			--text-1-name sentence1 --text-2-name sentence2 \
   			--label-name gold_label --index-name pairID \
			--invalid-labels "-" --not-comma-delimited \
			--validation-set snli_1.0_dev.txt --batch-size 1024 \
			--dropout 0.5 --units 256 --bidirectional \
			--model-directory-name snli.model

This achieved an accuracy of 80.16% on the development set and 79.49% on the test set after 9 epochs of training.


## References

* Travis Addair. Duplicate Question Pair Detection with Deep Learning
[[pdf](https://web.stanford.edu/class/cs224n/reports/2759336.pdf)]

* Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for 
learning natural language inference. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language 
Processing (EMNLP)_. [[pdf](https://nlp.stanford.edu/pubs/snli_paper.pdf)]
 
* Yushi Homma, Stuart Sy, Christopher Yeh. Detecting Duplicate Questions with Deep Learning.
[[pdf](https://web.stanford.edu/class/cs224n/reports/2748045.pdf)]

