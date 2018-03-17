
import numpy as np

from textwrap import dedent
from operator import itemgetter
from collections import defaultdict


DESCR = dedent("""\
    OCR Data Set
    ============

    The classical Optical Character Recognition data set from [taskar2004max].

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances:
        :Structure of the Examples:
            - length (int): Number of characters in word.
            - images (list of np.ndarray): List of the 16x8 images of the
              characters in the word.
        :Structure of the Labels:
            - sequence (list of int): List of symbols identifiers.
        :Creator: Ben Taskar

    The dataset is obtained from: http://ai.stanford.edu/~btaskar/ocr/

    REFERENCES
    ----------
       .. [taskar2004max] Taskar, Ben, Carlos Guestrin, and Daphne Koller.
          "Max-margin Markov networks." NIPS (2004).
""")


def load_data(file_name):
    """Load the data and target of the OCR dataset.

    Parameters
    ----------
    file_name : str
        The name of the file containing the OCR dataset.

    Returns
    -------
    X, Y, args : np.ndarray, np.ndarray, dict
        The lists containing the preprocessed examples and labels. Additionally,
        the boolean masks for the folds is returned in the optional arguments
        dictionary.
    """
    if isinstance(file_name, list):
        file_name = file_name[0]

    data_raw = np.genfromtxt(
        file_name, delimiter='\t',
        converters={1: lambda x: x} # Avoids NaN for field 1
    )

    n_folds = 10
    words = defaultdict(dict)
    folds = [set() for _ in range(n_folds)]
    for x_raw in data_raw:
        x = tuple(x_raw)
        word_id = int(x[3])
        position = int(x[4])
        fold = int(x[5])
        letter = ord(x[1].decode()) - ord('a') + 1
        image = np.array(x[6:134], dtype=np.int32).reshape(16, 8)
        words[word_id][position] = (letter, image)
        folds[fold].add(word_id)

    X = []
    Y = []
    mask = [[] for _ in range(n_folds)]
    for word_id, word in words.items():
        letters = list(zip(*sorted(word.items(), key=itemgetter(0))))[1]
        labels, images = list(zip(*letters))
        X.append({'length': len(images), 'images': images})
        Y.append({'sequence': labels})
        for fold in range(n_folds):
            mask[fold].append(word_id in folds[fold])

    return np.array(X), np.array(Y), {'folds': np.array(mask)}

