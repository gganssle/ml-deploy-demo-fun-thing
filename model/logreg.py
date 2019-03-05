import numpy as np
import pandas as pd

from util import confusion_plot

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

titanic = pd.read_csv('../dat/clean.csv')

y = titanic['Survived']
X = titanic.drop('Survived', axis=1)

X = X.values
y = y.values

X = np.delete(X, 3, 1)

print('Percent survived: ', 100 * np.count_nonzero(titanic['Survived']) / titanic.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf = LogisticRegression()
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print('accuracy score of classifier: ', accuracy_score(preds, y_test))

conf = confusion_matrix(y_test, preds)

confusion_plot.plot_confusion_matrix(conf, classes=['Died', 'Survived'])

confusion_plot.plot_confusion_matrix(conf, classes=['Died', 'Survived'], normalize=True)
