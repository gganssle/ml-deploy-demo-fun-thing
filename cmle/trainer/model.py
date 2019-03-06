import numpy as np
import tensorflow as tf
from io import BytesIO
from tensorflow.python.lib.io import file_io


def build_estimator(config):
    feature_columns = [
      tf.feature_column.numeric_column(key='time')
    ]

    return tf.estimator.DNNRegressor(
        config=config,
        feature_columns=feature_columns,
        label_dimension=1,
        hidden_units=[10, 100, 10],
        activation_fn=tf.nn.relu,
        #dropout=0.05,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    )


class data_pipeline():
    def __init__(self, label_file, feature_file, testpercentage=0.1):
        #labels = np.load(label_file)
        #features = np.load(feature_file)

        # this shitty thing is bc tf doesn't like gs://
        f = BytesIO(file_io.read_file_to_string(label_file, binary_mode=True))
        labels = np.load(f)
        f = BytesIO(file_io.read_file_to_string(feature_file, binary_mode=True))
        features = np.load(f)


        assert features.shape[0] == labels.shape[0]

        combo = np.stack([labels, features]).transpose()
        np.random.shuffle(combo)

        trainsize = int(np.ceil(features.shape[0]*(1-testpercentage)))
        self.train, self.test = combo[:trainsize], combo[trainsize:]

    def training(self):
        features, labels = {'time': self.train[:,1]}, self.train[:,0]
        return  tf.data.Dataset.from_tensors((features, labels)).shuffle(labels.shape[0]).repeat()

    def testing(self):
        features, labels = {'time': self.test[:,1]}, self.test[:,0]
        return  tf.data.Dataset.from_tensors((features, labels))
