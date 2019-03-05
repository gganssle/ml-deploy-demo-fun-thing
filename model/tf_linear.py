import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.feature_column as fc

import os, shutil
import argparse

import matplotlib.pyplot as plt


def build_estimator(config, file='../dat/clean.csv'):
    clean = pd.read_csv(file)
    names = clean.columns.values

    for i, name in enumerate(names):
        names[i] = name.replace(' ', '')

    feature_columns = []
    for i in range(clean.shape[1] - 1):
        feature_columns.append(tf.feature_column.numeric_column(names[i]))

    return tf.estimator.LinearClassifier(
        config=config,
        feature_columns=feature_columns,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    )


class data_pipeline():
    def __init__(self, testpercentage=0.1):
        clean = pd.read_csv('../dat/clean.csv')

        self.names = clean.columns.values

        for i, name in enumerate(self.names):
            self.names[i] = name.replace(' ', '')

        trainsize = int(np.ceil(clean.shape[0]*(1-testpercentage)))
        self.train, self.test = clean.iloc[:trainsize].values, clean.iloc[trainsize:].values

    def training(self):
        features = {}

        for i, name in enumerate(self.names):
            features[name] = self.train[:,i]

        labels = self.train[:,-1]

        return  tf.data.Dataset.from_tensors((features, labels)).shuffle(labels.shape[0]).repeat()

    def testing(self):
        features = {}

        for i, name in enumerate(self.names):
            features[name] = self.test[:,i]

        labels = self.test[:,-1]

        return  tf.data.Dataset.from_tensors((features, labels))


def run_it(args):
    dp = data_pipeline()

    train_input = lambda: dp.training()
    eval_input = lambda: dp.testing()

    train_spec = tf.estimator.TrainSpec(train_input, max_steps=3000)
    eval_spec = tf.estimator.EvalSpec(eval_input, steps=1)

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=args.model_dir)

    estimator = build_estimator(config=run_config)

    output, _ = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    for key in output:
        print(key, ': ', output[key])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', required=True)

    args = parser.parse_args()

    shutil.rmtree(args.model_dir, ignore_errors=True)
    os.mkdir(args.model_dir)

    run_it(args)
