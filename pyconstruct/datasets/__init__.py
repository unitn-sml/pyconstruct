"""\
Pyconstruct provides methods for loading a number of datasets for standard tasks
in structured-output prediction. The current list of available datasets
includes:

  - **ocr** : Ben Taskar's ORC dataset
  - **conll00** : CoNLL 2000 Text Chunking dataset
  - **horseseg** : HorseSeg dataset (coming soon)

Datasets can be loaded using the `load` function provided by this module. In
most cases, the dataset is downloaded upon first loading and stored in a local
directory on your computer for faster retrieval from the second loading onwards
(the actual directory depends on the operating system). The data is preprocessed
and made available in a format that can be already used for learning with any
algorithm provided by Pyconstruct.
"""

from .base import *


__all__ = [base.__all__]

