# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:45:01 2021

@author: anil.ms
"""
import os

pwd

os.chdir('d:/data_science/app')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
#from sklearn.externals import joblib

import pickle


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30, random_state=67)


#Pipeline for logistic regression
pipeline_lr = Pipeline([('scaler_1', StandardScaler()),
                        ('pca_1', PCA(n_components=2)),
                        ('lr_classifier', LogisticRegression(random_state=12))])


pipeline_lr.fit(X_train, y_train)


pipeline_lr.predict(X_test)


pickle.dump(pipeline_lr, open('model.pickle', 'wb'))


model = pickle.load(open('model.pickle', 'rb'))

model.predict(X_test)

X_test[0].reshape(1,-1)

#input_feats = [int(x) for x in request.form.values()]

input_feats = [2,3,4,5]
final_feats = np.array(input_feats).reshape(1,-1)
prediction = model.predict(np.array(input_feats).reshape(1,-1))
    
output=round(prediction[0],2)
  
list(prediction)  
return render_template('index.html', predictions='classified as {}'.format(output))

