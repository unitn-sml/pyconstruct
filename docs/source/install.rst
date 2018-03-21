Install
=======

Pyconstruct can be installed via Pip::

    pip install pyconstruct

or from the source code available
on `GitHub <https://github.com/unitn-sml/pyconstruct/releases/latest>`__::

    python setup.py install

Pyconstruct requires PyMzn to be installed on your machine, together with
`MiniZinc <https://github.com/MiniZinc/MiniZincIDE/releases/latest>`. Check-out
`PyMzn installation guide <http://paolodragone.com/pymzn/install.html>`__ for
details.

Alongside MiniZinc, you will need a constraint solver. PyMzn supports many
open-source solvers such as `Gecode <http://www.gecode.org/>` and `Chuffed
<https://github.com/chuffed/chuffed>`. However, we recommend you give `Gurobi
<http://www.gurobi.com/>`__ a try, it's blazingly fast for the kind of problems
we usually solve in structured-output prediction, even more so when adding
constraints to them. If you are in academia, they provide `free licenses
<http://www.gurobi.com/academia/for-universities>` for research purpose.

