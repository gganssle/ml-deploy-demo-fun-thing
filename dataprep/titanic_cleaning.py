import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer

titanic = pd.read_csv('../dat/titanic.csv')

titanic.drop(['Name','Ticket'], axis=1, inplace=True)

titanic['Family Members'] = titanic['Parch'] + titanic['SibSp']

titanic['Pclass category'] = pd.DataFrame(titanic['Pclass']).applymap(lambda x: str(x))

for i in list(titanic.columns):
    print('{} contains NaNs: '.format(i), np.amax(titanic[i].isnull()))


titanic['Embarked'] = titanic['Embarked'].fillna('NaN')

titanic['Cabin'] = titanic['Cabin'].fillna('NaN')

titanic.fillna('NaN', inplace=True)

one_hot = pd.get_dummies(titanic, columns=['Cabin', 'Embarked', 'Sex', 'Pclass category'])

y = one_hot['Survived']
X = one_hot.drop('Survived', axis=1)

X = X.values
y = y.values

X, y = shuffle(X, y)

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

X = imputer.fit_transform(X)

cols = list(one_hot.columns)
cols.pop(6)
cols.append('Survived')

clean = pd.DataFrame(np.hstack((X,y.reshape(-1,1))), columns=cols)

clean.to_csv('../dat/clean.csv', index=False)
