
import numpy as np

from textwrap import dedent
from functools import partial
from collections import Counter


DESCR = dedent("""\
    CoNLL 2000 Text Chunking Data Set
    =================================

    The classical CoNLL 2000 text chunking data set from [sang2000conll].

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 17872
        :Structure of the input examples:
            - length (int): Number of words in the sentence.
            - attributes (list of np.ndarray): List of arrays, one per word,
              each containing 19 attributes.
        :Structure of the output labels:
            - sequence (list of int): List of chunk type identifiers.
        :Creator: Erik F. Tjong Kim Sang and Sabine Buchholz

    The dataset is obtained from:
      https://www.clips.uantwerpen.be/conll2000/chunking/

    The data preprocessing is done as in [osokin2016minding]:
      http://www.di.ens.fr/sierra/research/gapBCFW/

    REFERENCES
    ----------
       .. [sang2000conll] Tjong Kim Sang, Erik F., and Sabine Buchholz.
          "Introduction to the CoNLL-2000 shared task: Chunking." Proceedings of
          the 2nd workshop on Learning language in logic and the 4th conference
          on Computational natural language learning-Volume 7. Association for
          Computational Linguistics, 2000.
       .. [osokin2016minding] Osokin, Anton, Jean-Baptiste Alayrac, Isabella
          Lukasewitz, Puneet Dokania, and Simon Lacoste-Julien. "Minding the
          gaps for block frank-wolfe optimization of structured svms." In
          International Conference on Machine Learning, pp. 593-602. 2016.
""")


label2int = {
    "B-ADJP": 1, "B-ADVP": 2, "B-CONJP": 3, "B-INTJ": 4, "B-LST": 5, "B-NP": 6,
    "B-PP": 7, "B-PRT": 8, "B-SBAR": 9, "B-UCP": 10, "B-VP": 11, "I-ADJP": 12,
    "I-ADVP": 13, "I-CONJP": 14, "I-INTJ": 15, "I-LST": 16, "I-NP": 17,
    "I-PP": 18, "I-PRT": 19, "I-SBAR": 20, "I-UCP": 21, "I-VP": 22, "O": 23
}


int2label = {
    1: "B-ADJP", 2: "B-ADVP", 3: "B-CONJP", 4: "B-INTJ", 5: "B-LST", 6: "B-NP",
    7: "B-PP", 8: "B-PRT", 9: "B-SBAR", 10: "B-UCP", 11: "B-VP", 12: "I-ADJP",
    13: "I-ADVP", 14: "I-CONJP", 15: "I-INTJ", 16: "I-LST", 17: "I-NP",
    18: "I-PP", 19: "I-PRT", 20: "I-SBAR", 21: "I-UCP", 22: "I-VP", 23: "O"
}


def neighborhood(sentence, i):
    w = [sentence[i][0].lower()]
    w.append(sentence[i+1][0].lower() if i < len(sentence) - 1 else '')
    w.append(sentence[i+2][0].lower() if i < len(sentence) - 2 else '')
    w.append(sentence[i-2][0].lower() if i >= 2 else '')
    w.append(sentence[i-1][0].lower() if i >= 1 else '')
    pos = [sentence[i][1]]
    pos.append(sentence[i+1][1] if i < len(sentence) - 1 else '')
    pos.append(sentence[i+2][1] if i < len(sentence) - 2 else '')
    pos.append(sentence[i-2][1] if i >= 2 else '')
    pos.append(sentence[i-1][1] if i >= 1 else '')
    return w, pos


attribute_keys = [
    # word-unigrams
    lambda w, pos: w[-2],
    lambda w, pos: w[-1],
    lambda w, pos: w[0],
    lambda w, pos: w[1],
    lambda w, pos: w[2],

    # word-bigrams
    lambda w, pos: '|'.join([w[-1], w[0]]),
    lambda w, pos: '|'.join([w[0], w[1]]),

    # POS-unigrams
    lambda w, pos: pos[-2],
    lambda w, pos: pos[-1],
    lambda w, pos: pos[0],
    lambda w, pos: pos[1],
    lambda w, pos: pos[2],

    # POS-bigrams
    lambda w, pos: '|'.join([pos[-2], pos[-1]]),
    lambda w, pos: '|'.join([pos[-1], pos[0]]),
    lambda w, pos: '|'.join([pos[0], pos[1]]),
    lambda w, pos: '|'.join([pos[1], pos[2]]),

    # POS-trigrams
    lambda w, pos: '|'.join([pos[-2], pos[-1], pos[0]]),
    lambda w, pos: '|'.join([pos[-1], pos[0], pos[1]]),
    lambda w, pos: '|'.join([pos[0], pos[1], pos[2]]),
]


def extract_counts(train_file_name):
    counts = [Counter() for _ in range(len(attribute_keys))]
    with open(train_file_name) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:
                for i in range(len(sentence)):
                    w, pos = neighborhood(sentence, i)
                    attrs = [attr_key(w, pos) for attr_key in attribute_keys]
                    for i, attr in enumerate(attrs):
                        counts[i][attr] += 1
                sentence = []
                continue
            sentence.append(line.split(' ')[:2])
    return counts


def sentence_transform(sentence, counts, min_count, attribute_indices):
    attributes = []
    for i in range(len(sentence)):
        attributes.append([])
        w, pos = neighborhood(sentence, i)
        attrs = [attr_key(w, pos) for attr_key in attribute_keys]
        for j in range(len(attrs)):
            if (
                    attrs[j] not in counts[j]
                or counts[j][attrs[j]] < min_count
            ):
                idx = attribute_indices[j][None]
            else:
                idx = attribute_indices[j][attrs[j]]
            attributes[-1].append(idx)
    return np.array(attributes)


def load_data(file_names):
    """Load the data and target of the CoNLL 2000 dataset.

    Parameters
    ----------
    file_names : [str]
        The names of the training and test files (in this order) containing the
        dataset.

    Returns
    -------
    X, Y, args : np.ndarray, np.ndarray, dict
        The lists containing the preprocessed examples and labels. Additionally,
        the boolean masks for the folds is returned in the optional arguments
        dictionary.
    """
    min_count = 3
    counts = extract_counts(file_names[0])

    train_attributes = []
    train_labels = []
    with open(file_names[0]) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:
                for i in range(len(sentence)):
                    w, pos = neighborhood(sentence, i)
                    attrs = [attr_key(w, pos) for attr_key in attribute_keys]
                    for j in range(len(attrs)):
                        if (
                               attrs[j] not in counts[j]
                            or counts[j][attrs[j]] < min_count
                        ):
                            attrs[j] = None
                    train_attributes.append(attrs)
                    train_labels.append(sentence[i][2])
                train_attributes.append([None] * len(attribute_keys))
                train_labels.append(None)
                sentence = []
                continue
            sentence.append(line.split(' ')[:3])

    test_attributes = []
    test_labels = []
    with open(file_names[1]) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:
                for i in range(len(sentence)):
                    w, pos = neighborhood(sentence, i)
                    attrs = [attr_key(w, pos) for attr_key in attribute_keys]
                    for j in range(len(attrs)):
                        if (
                               attrs[j] not in counts[j]
                            or counts[j][attrs[j]] < min_count
                        ):
                            attrs[j] = None
                    test_attributes.append(attrs)
                    test_labels.append(sentence[i][2])
                test_attributes.append([None] * len(attribute_keys))
                test_labels.append(None)
                sentence = []
                continue
            sentence.append(line.split(' ')[:3])

    attribute_curmax = [0 for _ in range(len(attribute_keys))]
    attribute_indices = [{None: 0} for _ in range(len(attribute_keys))]
    train_attributes_num = []
    for attrs in train_attributes:
        train_attributes_num.append([])
        for i, attr in enumerate(attrs):
            if attr in attribute_indices[i]:
                idx = attribute_indices[i][attr]
            else:
                idx = attribute_curmax[i] + 1
                attribute_indices[i][attr] = idx
                attribute_curmax[i] += 1
            train_attributes_num[-1].append(idx)

    test_attributes_num = []
    for attrs in test_attributes:
        test_attributes_num.append([])
        for i, attr in enumerate(attrs):
            if attr in attribute_indices[i]:
                idx = attribute_indices[i][attr]
            else:
                idx = 0
            test_attributes_num[-1].append(idx)

    del train_attributes, test_attributes

    train_attributes = np.array(train_attributes_num)
    test_attributes = np.array(test_attributes_num)

    del train_attributes_num, test_attributes_num

    train_X = []
    train_Y = []
    sentence_attrs = []
    sentence_labels = []
    for attrs, label in zip(train_attributes, train_labels):
        if label is None:
            train_X.append({
                'length': len(sentence_attrs),
                'attributes': np.vstack(sentence_attrs)
            })
            train_Y.append({'labels': np.vstack(sentence_labels)})
            sentence_attrs = []
            sentence_labels = []
            continue
        sentence_attrs.append(attrs)
        sentence_labels.append(label2int[label])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    test_X = []
    test_Y = []
    sentence_attrs = []
    sentence_labels = []
    for attrs, label in zip(test_attributes, test_labels):
        if label is None:
            test_X.append({
                'length': len(sentence_attrs),
                'attributes': np.vstack(sentence_attrs)
            })
            test_Y.append({'labels': np.vstack(sentence_labels)})
            sentence_attrs = []
            sentence_labels = []
            continue
        sentence_attrs.append(attrs)
        sentence_labels.append(label2int[label])

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    X = np.concatenate((train_X, test_X))
    Y = np.concatenate((train_Y, test_Y))

    training_set = range(train_X.shape[0])
    test_set = range(test_X.shape[0])

    return X, Y, {
        'training': training_set, 'test': test_set,
        'transform': partial(
            sentence_transform, counts=counts, min_count=min_count,
            attribute_indices=attribute_indices
        )
    }

