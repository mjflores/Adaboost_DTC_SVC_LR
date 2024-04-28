#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                    Adaboost
                    
Adaboost con tres clasificadores débiles:
 - DecisionTreeClassifier
 - SVC
 - LogisticRegression
 
@author: mjflores
@date: Mon Dec 20 21:26:19 2021
@Ref:
    https://www.programcreek.com/python/example/86712/sklearn.ensemble.AdaBoostRegressor

"""

import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold


from sklearn.datasets import load_iris

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#import sklearn
#print("sklearn.__version__", sklearn.__version__)

           
def train_gridsearch_classification_DTC(iris,cv_kf):
        
    # Check that base trees can be grid-searched.
    # AdaBoost classification
    boost_DTC = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    parameters = {'n_estimators': (1,2,3,4,5,6,7,8,9,10,15,20,25),                  
                  'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
                  'algorithm': ('SAMME', 'SAMME.R'),
                  'base_estimator__max_depth': (1, 2,3,4,5,6,8,10,12,15,20)}
    
    clf = GridSearchCV(boost_DTC, parameters,cv=cv_kf)
    clf.fit(iris.data, iris.target)
    
    print("Accuracy boost_DTC = ", clf.best_score_)
    print("Best params boost_DTC = ", clf.best_params_)
    
def train_gridsearch_classification_SVC(iris,cv_kf):
   
    boost_SVC = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',probability=True))
    
    parameters = {'n_estimators': (1,2,3,4,5,6,7,8,9,10,15,20,25),                  
                  'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
                  'algorithm': ('SAMME', 'SAMME.R'),
                  'base_estimator__C': (0.1,0.5,1.0,5.0, 10.0), 'base_estimator__gamma': (1.0,0.75,0.5,0.25,0.1,0.01) }
    
    clf = GridSearchCV(boost_SVC,parameters, cv=cv_kf, scoring='accuracy', n_jobs=-1)
    clf.fit(iris.data, iris.target)
    print("Accuracy boost_SVM = ",clf.best_score_)
    print("Best params boost_DTC = ", clf.best_params_)
    
    boost_SVC.fit(iris.data, iris.target)
    print("Accuracy SVC= " ,boost_SVC.score(iris.data, iris.target))
    
    
def train_gridsearch_classification_LogReg(iris,cv_kf):
    
    boost_LogReg = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter = 1000))
    
    parameters = {'n_estimators': (1,2,3,4,5,6,7,8,9,10,15,20,25),                  
                  'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
                  'algorithm': ('SAMME', 'SAMME.R'),
                  'base_estimator__C':(0.1,0.5,1.0,5.0, 10.0),
                  'base_estimator__solver': ('newton-cg','lbfgs')}    
    
    clf = GridSearchCV(boost_LogReg,parameters, cv=cv_kf ,scoring='accuracy', n_jobs=-1)    
    clf.fit(iris.data, iris.target)
    print("Accuracy boost_SVM = ",clf.best_score_)
    print("Best params boost_DTR = ", clf.best_params_)
    
    boost_LogReg.fit(iris.data, iris.target)
    print("Accuracy Log_Reg= " ,boost_LogReg.score(iris.data, iris.target))
    

#===================================================================================

cv_kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
iris_dt   = load_iris()

#print("len(iris_dt)", len(iris_dt))
#print("type(iris_dt)", type(iris_dt))
print(iris_dt.data)
print(iris_dt.target)


#train_gridsearch_classification_DTC(iris_dt,cv_kf)
#train_gridsearch_classification_SVC(iris_dt,cv_kf)
train_gridsearch_classification_LogReg(iris_dt,cv_kf)

