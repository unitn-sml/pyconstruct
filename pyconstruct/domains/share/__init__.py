"""\
Pyconstruct provides a Jinja2 API with number of utilities for building domains.

To use the available macros, simply import them from `pyconstruct.pmzn`, e.g.::

    {% from 'pyconstruct.pmzn' import domain, n_features, features, solve  %}


CONSTANTS
=========
    BOOL_SET
        An integer set of boolean values {0, 1}.
    INT_SET
        A standard set of integers.
    FLOAT_SET
        A standard set of floats.


MACROS
======

    n_features(
        model=none, n_features_var='N_FEATURES', feature_set_var='FEATURES'
    )
        Boilerplate for feature number definition.

        PARAMETERS
        ----------
        model : dict
            Parameters of the model.
        n_feature_var : str
            The variable name of the feature array. Default is 'phi'.
        feature_set : str
            The variable name of the feature index set: Default is 'FEATURES'.

        USAGE
        -----
        Only fixed domain features:
            {% call n_features() %}
                N_FEATURE_SET_1 + N_FEATURE_SET_2 + 5
            {% endcall %}

        Only model-dependent features:
            {{ call n_features(model) }}

        Combination of both:
            {% call n_features(model) %}
                N_FEATURE_SET_1 + N_FEATURE_SET_2 + 5
            {% endcall %}

    features(
        model=none, feature_var='phi', feature_set='FEATURES',
        feature_type='float'
    )
        Boilerplate for features definition.

        PARAMETERS
        ----------
        model : dict
            Parameters of the model.
        feature_var : str
            The variable name of the feature array. Default is 'phi'.
        feature_set : str
            The variable name of the feature index set: Default is 'FEATURES'.
        feature_type : str
            The type of the features. Default is 'float'.

        USAGE
        -----
        Only fixed domain features:
            {% call features() %}
                % Your features definition, e.g.
                % [ feature1, feature2 ] ++ [ feature3 ]
            {% endcall %}

        Only model-dependent features:
            {{ call features(model) }}

        Combination of both:
            {% call features(model) %}
                % Your features definition, e.g.
                % [ feature1, feature2 ] ++ [ feature3 ]
            {% endcall %}



    domain(
        problem, domain='', allowed=('phi', 'map', 'loss_augmented_map')
    )
        Boilerplate for domain definition.

        PARAMETERS
        ----------
        problem : str in ['phi', 'n_features', 'map', 'loss_augmented_map']
            The problem to solve. This parameter is usually passed by the Python
            domain class.
        domain : str
            The domain definition. Use this parameter in alternative to caller.
        allowed : tuple
            Allowed problems for this domain.

        USAGE
        -----
            {% call domain(problem) %}
                % Your domain definition, e.g.
                % int: x;
                % var 0 .. 10: y;
                % constraint x + y <= 10;
            {% endcall %}


    solve(
        problem, model={}, discretize=False, factor=100, score='',
        feature_var='phi', feature_set='FEATURES', w_type='', score_type='',
        loss='', loss_type=''
    )
        Boilerplate for problem dependent solve statement.

        PARAMETERS
        ----------
        problem : str in ['phi', 'n_features', 'map', 'loss_augmented_map']
            The problem to solve. This parameter is usually passed by the Python
            domain class.
        model : dict
            Dictionary containing the model's paramenters.
        discretize : bool
        Wheter to discretize the weight vector and the scoring function. Use only
        if feature_type and loss_type are int. Default 'False'.
        factor : int
            The pre-multiplicative factor before discretization.
        score : str
            The score function. Default to dot product between weights and features.
        feature_var : str
            Name of the variable containing the feature array.
        feature_set : str
            Name or formula for the index set of the feature array.
        w_type : str
            The type of the weights. Defaults to float unless discretize is True, in
            which case defaults to int.
        score_type : str
            The type of the score. Defaults to float unless discretize is True, in
            which case defaults to int.
        loss : str
            Formula for the loss. Used in loss_augmented_map problems.
        loss_type : str
            The type of the loss. Defaults to float unless discretize is True, in
            which case defaults to int.

        USAGE
        -----
            {{ solve(problem, model, ...) }}
"""
