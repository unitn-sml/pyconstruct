#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import codecs

here = os.path.abspath(os.path.dirname(__file__))

def find_version(*parts):
    # Open in Latin-1 so that we avoid encoding errors.
    # Use codecs.open for Python 2 compatibility
    with codecs.open(os.path.join(here, *parts), 'r', 'latin1') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath'
    # 'numpydoc'
]

autosummary_generate = True
# numpydoc_show_class_members = False

imgmath_image_format = 'svg'
imgmath_latex_preamble = r"""
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
\newcommand{\abs}[1]{\lvert#1\rvert}
\newcommand{\norm}[1]{\lVert#1\rVert}
\newcommand{\inner}[2]{\langle#1,#2\rangle}
\newcommand{\vect}[1]{\ensuremath{\boldsymbol{#1}}}
\newcommand{\vw}{\vect{w}}
\newcommand{\vphi}{\vect{\phi}}
\newcommand{\calA}{\ensuremath{\mathcal{A}}}
\newcommand{\calD}{\ensuremath{\mathcal{D}}}
\newcommand{\calX}{\ensuremath{\mathcal{X}}}
\newcommand{\calY}{\ensuremath{\mathcal{Y}}}
\newcommand{\bbR}{\ensuremath{\mathbb{R}}}
"""

templates_path = ['_templates']
exclude_patterns = []

source_suffix = '.rst'
master_doc = 'index'
html_theme = 'sphinx_rtd_theme'

language = None
today_fmt = '%B %d, %Y'
default_role = 'autolink'
pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'pyconstruct'

project = 'Pyconstruct'
copyright = '2017, Paolo Dragone (MIT Licence)'
author = 'Paolo Dragone, Stefano Teso, Andrea Passerini'

version = find_version('..', '..', 'pyconstruct', '__init__.py')
release = find_version('..', '..', 'pyconstruct', '__init__.py')
print(version)

