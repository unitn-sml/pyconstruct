Pyconstruct
===========

<div align="center">
  <img height="300px" src="docs/_images/pyconstruct.png"><br><br>
</div>

**Pyconstruct** is a Python library for declarative, constrained,
structured-output prediction. When using Pyconstruct, the problem specification
can be encoded in MiniZinc, a high-level constraint programming language. This
means that domain knowledge can be declaratively included in the inference
procedure as constraints over the optimization variables.

Check out the [Quick Start
guide](https://unitn-sml.github.io/pyconstruct/quick_start.html) to learn
how to solve your first problem with Pyconstruct.

Have a look at the [docs](https://unitn-sml.github.io/pyconstruct/index.html)
and the [reference
manual](https://unitn-sml.github.io/pyconstruct/reference/index.html) too, to
learn more about it!

Install
-------
Pyconstruct can be installed through `pip`:

```bash
pip install pyconstruct
```

Or by downloading the code from Github and running the following from the
downloaded directory:

```bash
python setup.py install
```

Before using Pyconstruct you will need to install **MiniZinc** as well.
Download the latest release of MiniZincIDE and follow the instructions.

Check out the [Installation
guide](https://unitn-sml.github.io/pyconstruct/install.html) for more details.

Authors
-------
This project is developed at the [SML research group](http://sml.disi.unitn.it/)
at the University of Trento (Italy). Main developers and maintainers:

* [Paolo Dragone](http://paolodragone.com)
* [Stefano Teso](http://disi.unitn.it/~teso/) (now at KU Leuven)
* [Andrea Passerini](http://disi.unitn.it/~passerini/)

