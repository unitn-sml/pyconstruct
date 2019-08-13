""" equations.py

Example of code for training an SSG learner over the equations data.
"""

import pymzn
pymzn.config.no_output_annotations = True

from time import time
from pyconstruct import Domain, SSG
from pyconstruct.datasets import load_equations
from pyconstruct.metrics import hamming
from pyconstruct.utils import batches, load, save

from sklearn.model_selection import train_test_split


def loss(Y_pred, Y_true, n_jobs=1):
    return hamming(
        Y_pred, Y_true, key='sequence', n_jobs=n_jobs
    )


def train(args):

    dom = Domain(
        args.domain_file, n_jobs=args.parallel,
        no_constraints=args.no_constraints,
        timeout=1
    )

    eq_data = load_equations()
    X, Y = eq_data.data, eq_data.target
    X, Y = X[:args.n_samples], Y[:args.n_samples]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print('Splitting dataset: {} training examples, {} test examples'.format(
        X_train.shape[0], X_test.shape[0]
    ))

    sp = SSG(dom, inference='map', n_jobs=args.parallel)
    bs = 2 * args.parallel

    losses = []
    times = []
    for i, (X_b, Y_b) in enumerate(batches(X_train, Y_train, batch_size=bs)):

        # Validation
        t0 = time()
        Y_pred = sp.predict(X_b)
        infer_time = time() - t0

        losses.append(loss(Y_pred, Y_b, n_jobs=args.parallel).mean())
        avg_loss = sum(losses) / len(losses)

        # Learning
        t0 = time()
        sp.partial_fit(X_b, Y_b)
        learn_time = time() - t0

        times.append((infer_time, learn_time))

        print('Batch {}'.format(i + 1))
        print('Examples: {}'.format(i * bs + X_b.shape[0]))
        print('Training loss = {}'.format(losses[-1]))
        print('Average training loss = {}'.format(avg_loss))
        print('Infer time = {}'.format(infer_time))
        print('Learn time = {}\n'.format(learn_time))

    print('Training complete!')
    print('Inference on the test set...')

    t0 = time()
    Y_pred = sp.predict(X_test)
    infer_time = time() - t0

    test_losses = loss(Y_pred, Y_test, n_jobs=args.parallel)
    print('Test loss = {}'.format(test_losses.mean()))
    print('Infer time = {}\n'.format(infer_time))

    save({
        'train-losses': losses, 'test-losses': test_losses, 'times': times
    }, args.output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--domain-file', default='equations.pmzn')
    parser.add_argument('-n', '--n_samples', type=int, default=1000)
    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-N', '--no-constraints', action='store_true')
    parser.add_argument('-O', '--output', default='results.pickle')
    args = parser.parse_args()

    train(args)

