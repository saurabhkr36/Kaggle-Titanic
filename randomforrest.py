# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:50:08 2019

@author: Heisenberg
"""
import pandas as pd
import numpy as np

test=pd.read_csv("test.csv")
train=pd.read_csv("train.csv")

del test['Name']
del train['Name']
del test['Ticket']
del train['Ticket']
del test['Cabin']
del train['Cabin']
del test['PassengerId']
del train['PassengerId']

test.apply(lambda x: sum(x.isnull()))
train.apply(lambda x: sum(x.isnull()))

train['Age'].fillna(np.mean(train['Age']),inplace = True)
test['Age'].fillna(np.mean(test['Age']),inplace = True)
test['Fare'].fillna(np.mean(test['Fare']),inplace = True)

train['Embarked'].fillna('S', inplace = True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
train['Sex'] = labelencoder_x.fit_transform(train['Sex'])
test['Sex'] = labelencoder_x.fit_transform(test['Sex'])
train['Embarked'] = labelencoder_x.fit_transform(train['Embarked'])
test['Embarked'] = labelencoder_x.fit_transform(test['Embarked'])

#model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(train.iloc[:,1:],  train.iloc[:,0])


# Predicting the Test set results
y_pred = classifier.predict(test)

y_pred=pd.DataFrame(y_pred)

ID=test['PassengerId']
ID.to_csv("Submission.csv")
submission =y_pred
submission.to_csv("survived.csv")