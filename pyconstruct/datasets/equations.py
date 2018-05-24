
import numpy as np

from textwrap import dedent

from ..utils import load


DESCR = dedent("""\
    OCR Equations Data Set
    ======================

    The OCR equations dataset used in [dragone2018pyconstruct].

    This is a simple dataset of sequences of images containing math symbols. The
    sequences contain equations of the type: a + b = c. The numbers a, b and c
    are all positive integers of variable length (max 3 digits). The equations
    are all valid.

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 10000
        :Structure of the Examples:
            - length (int): Number of symbols in the equation.
            - images (list of np.ndarray): List of the 9x9 images of the
              symbols in the equation.
        :Structure of the Labels:
            - sequence (list of int): List of symbols identifiers.

    Labels:
        0-9 : digits
         10 : '+' (plus symbol)
         11 : '=' (equal symbol)

    The dataset was extracted from the data available at:
        https://www.kaggle.com/xainano/handwrittenmathsymbols

    Check out the extractor from:
        https://github.com/paolodragone/handwritten-formulas-generator

    The original dataset was extracted from the CROHME competition, and was
    extracted with the tool from Thomas Lech:
        https://github.com/ThomasLech/CROHME_extractor

    REFERENCES
    ----------
       .. [dragone2018pyconstruct] Dragone, Paolo, Teso, Stefano, and Andrea
          Passerini. "Pyconstruct: Constraint Programming meets Structured
          Prediction" IJCAI (2018).
""")


def load_data(file_name):
    """Load the data and target of the OCR equations dataset.

    Parameters
    ----------
    file_name : str
        The name of the file containing the OCR equations dataset.

    Returns
    -------
    X, Y, args : np.ndarray, np.ndarray, dict
        The lists containing the preprocessed examples and labels.
    """
    if isinstance(file_name, list):
        file_name = file_name[0]
    return load(file_name)

