""" formulas.py

Example of code for training a StructuredPerceptron over the formulas data.
"""

import pymzn
pymzn.config.set('solver', pymzn.gurobi)

from time import time
from pyconstruct import Domain, StructuredPerceptron
from pyconstruct.metrics import hamming
from pyconstruct.utils import batches, load

from sklearn.model_selection import train_test_split


def loss(Y_pred, Y_true, parallel=4):
    return hamming(
        Y_pred, Y_true, key='sequence', n_jobs=parallel
    ).mean()


def train(args):

    dom = Domain(args.domain_file, n_jobs=args.parallel)

    X, Y = load(args.data_file)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.005, random_state=42
    )

    print('Splitting dataset: {} training example, {} test examples'.format(
        X_train.shape[0], X_test.shape[0]
    ))

    sp = StructuredPerceptron(domain=dom)

    for i, (X_b, Y_b) in enumerate(batches(X_train, Y_train, batch_size=50)):

        # Learning
        t0 = time()
        sp.partial_fit(X_b, Y_b)
        learn_time = time() - t0

        # Validation
        t0 = time()
        Y_pred = sp.predict(X_test)
        infer_time = time() - t0

        print('Batch {}'.format(i))
        print('Loss = {}'.format(loss(Y_pred, Y_test, parallel=args.parallel)))
        print('Learn time = {}'.format(learn_time))
        print('Infer time = {}'.format(infer_time))
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-file', default='formulas.pickle')
    parser.add_argument('-D', '--domain-file', default='formulas.pmzn')
    parser.add_argument('-p', '--parallel', type=int, default=4)
    args = parser.parse_args()

    train(args)

