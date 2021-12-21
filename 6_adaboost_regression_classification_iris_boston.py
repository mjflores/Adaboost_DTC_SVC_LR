#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: mjflores
@date: Mon Dec 20 21:26:19 2021
@REf:
    https://www.programcreek.com/python/example/86712/sklearn.ensemble.AdaBoostRegressor

"""


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold


from sklearn.datasets import load_iris

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
    
    #boost_SVC = AdaBoostClassifier(base_estimator=SVC(kernel='rbf'))
    boost_SVC = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',probability=True))
    
    #parameters = {'C': (0.00001,0.0001,0.001,0.01,0.1,0.5,1.0,5.0, 10.0, 20.0, 50.0, 100.0,500.0, 1000.0), 
    #            'gamma': (10.0,5.0,3.0,2.0,1.0,0.75,0.5,0.25,0.1,0.01,0.001,0.0001,0.00001,0.000001)}
    #parameters = {'C': (0.1,0.5,1.0,5.0, 10.0), 'gamma': (1.0,0.75,0.5,0.25,0.1,0.01,0.001,0.0001,0.00001,0.000001)}
    
    
    parameters = {'n_estimators': (1,2,3,4,5,6,7,8,9,10,15,20,25),                  
                  'learning_rate': (0.0001, 0.001, 0.01, 0.1, 1.0),
                  'algorithm': ('SAMME', 'SAMME.R'),
                  'base_estimator__C': (0.1,0.5,1.0,5.0, 10.0), 'base_estimator__gamma': (1.0,0.75,0.5,0.25,0.1,0.01) }
    
    #clf = GridSearchCV(boost_SVM, clf = GridSearchCV(boost_SVC, parameters,cv=cv_kf, n_jobs=-1),cv=cv, n_jobs=-1)
    clf = GridSearchCV(boost_SVC, parameters)
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
    



cv_kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
iris_dt   = load_iris()

#train_gridsearch_classification_DTC(iris_dt,cv_kf)
#train_gridsearch_classification_SVC(iris_dt,cv_kf)
train_gridsearch_classification_LogReg(iris_dt,cv_kf)



#==============================================================================
# Regression
#==============================================================================

from sklearn.datasets import load_boston
    
def train_gridsearch_regression(boston):
    cv = KFold(n_splits=10,shuffle=True,random_state=2)
    # AdaBoost regression
    boost_DTR = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), 
                              random_state=0)
    parameters = {'n_estimators': (1, 2,3,4,5,6,7,8,9,10,11,12,13,15,20,30,40,50),
                  'base_estimator__max_depth': (1, 2, 4 ,6 ,8 ,10,15,20,25,30,40)}
    
    clf = GridSearchCV(boost_DTR, parameters,cv=cv,scoring = 'r2', n_jobs=-1)
    clf.fit(boston.data, boston.target) 
    
    print("MSE boost_DTR = ",clf.best_score_)
    print("Best params boost_DTR = ", clf.best_params_)
    
    boost = AdaBoostRegressor()
    #clf = GridSearchCV(boost)
    boost.fit(boston.data, boston.target) 
    
    print("MSE boost = ",boost.score(boston.data, boston.target))
    



#boston_dt = load_boston()
#train_gridsearch_regression(boston_dt)
