# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:33:58 2021

@author: hsanthanam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import auc, roc_curve, confusion_matrix, roc_auc_score

from imblearn.under_sampling import RandomUnderSampler
from collections import defaultdict

import shap
import seaborn as sn
import matplotlib.pyplot as plt


# create feature blocks
def feature_blocks():
    i = 0
    feature_block_names = defaultdict(list)
    while i < (len(df_X_train.columns)):
        
        col = df_X_train.columns[i]
        
        # to handle last feature 
        if i == (len(df_X_train.columns) - 1):
            feature_block_names[col].append(col)
            break
        
        col_next = df_X_train.columns[i+1]
        block_name = col.split('_')[0]
        
        # systolic and venuous pressure only have _indicators and no actual values
        if block_name == 'VenousPressure' or block_name == 'SystolicBP':
            if '_N' in col:
                feature_block_names[col].append(col)
            else:  
                feature_block_names[block_name + '_indicator'].append(col) 
        
            
        # these indicate physician decision on whether to find measurement
        elif '_N' in col:
            if (col.index('_N') + 2) == len(col):
                feature_block_names[col].append(col)
            if 'indicator' in col:
                feature_block_names[col].append(col)
            
        elif col in col_next and 'indicator' in col_next:
            feature_block_names[block_name].append(col)
            feature_block_names[block_name + '_indicator'].append(col_next)
            i += 1
            
        else:
            feature_block_names[block_name].append(col) 
            
        i += 1 
    
    return feature_block_names

#----------------------------------------------------------------------------#
# plot with correlation and shap values
def ABS_SHAP(df_shap, df):
    # make copy of input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index', axis=1)
    
    # determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    
    # make data frame. column1 is the feature, and column2 is correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')
    
    # plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns=['Variable','SHAP_abs']
    k2 = k.merge(corr_df, left_on = 'Variable', right_on = 'Variable', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=False)
    k3 = k2[0:20]
    colorlist = k3['Sign']
    
    ax = k3.plot.barh(x='Variable',y='SHAP_abs',color=colorlist,figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    ax.invert_yaxis()
    plt.show()
    
    return k3

#----------------------------------------------------------------------------#
# generate the individual and group shapley plots
def shap_plots(shap_values, classifier_type):
    shap.summary_plot(shap_values, df_X_train, plot_type = 'bar')

    if classifier_type == 'random_forest':
        shap_values = shap_values[1]

    feature_block_names = feature_blocks()
 
    #------------------------------------------------------------------------------------------#
    # combine feature importances in blocks
    feature_block_vals = {}
    shap_values1 = np.mean(np.absolute(shap_values), axis=0)
    for i, col in enumerate(df_X_train.columns):
        for key, val in feature_block_names.items():
            if col in val:
                if key not in feature_block_vals.keys():
                     feature_block_vals[key] = shap_values1[i]
                else:
                    feature_block_vals[key] += shap_values1[i]
                
            
    print(feature_block_vals['WeightKg'])
    vals = np.expand_dims(np.array(list(feature_block_vals.values())), axis=0)
    df = pd.DataFrame(columns=list(feature_block_vals.keys()))
    shap.summary_plot(vals, df.columns, plot_type = 'bar')
    
    # plotting shap values with correlations
    abs_shap = ABS_SHAP(shap_values, df_X_train)
    abs_shap = abs_shap.sort_values(by='SHAP_abs', ascending=False)

    # correlations
    num_features = 30
    corr_list = sorted(abs_shap.iloc[:num_features, 0].tolist())
    corr_df = df_X_train[corr_list]
    corrMatrix = corr_df.corr()
    sn.heatmap(corrMatrix,cmap="RdBu_r",annot=True)
    plt.show()
    
    # dendrogram of clusters
    g = sn.clustermap(corrMatrix, cmap="RdBu_r")
    plt.show()

#-----------------------------------------------------------------------------#
# download data
df_X = pd.read_pickle('C:/Users/hsanthanam/Documents/Hari_Sepsis/data/processed_sepsis_data')
df_y = pd.read_pickle('C:/Users/hsanthanam/Documents/Hari_Sepsis/data/label_data')

X_total = df_X.to_numpy()
y_total = np.ravel(df_y.to_numpy())

# train and test for full data
X_full_train, X_full_test, y_full_train, y_full_test= train_test_split(X_total, y_total, test_size=0.2, random_state=42)

# balance the full data
undersample = RandomUnderSampler(sampling_strategy=0.333) # 855 - 0s, 285 - 1s
X_under, y_under = undersample.fit_resample(X_total, y_total)

# extract test and train for small dataset
X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(X_under, y_under, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------#
dataset_type = 'Full'

print("Analysis on : " + dataset_type + " Dataset")
if dataset_type == 'Undersampled':
    X_train = X_small_train
    y_train = y_small_train
    X_test = X_small_test
    y_test = y_small_test

if dataset_type == 'Full':
    X_train = X_full_train
    y_train = y_full_train
    X_test = X_full_test
    y_test = y_full_test

df_X_train = pd.DataFrame(X_train, columns = df_X.columns)


# LOGISTIC REGRESSION ------------------------------------------------------#

print("Logistic Regression")

# do grid search for elastic net
#grid={"l1_ratio": [0, 0.25, 0.5, 0.75, 1]}# l1 lasso l2 ridge
#log = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, max_iter=500)
#log_cv = GridSearchCV(log,grid,cv=2)
#log_cv.fit(X_train,y_train)

#print("tuned hyperparameters :(best parameters) ",log_cv.best_params_)

# train model with grid best parameters
#best_l1_ratio = log_cv.best_params_['l1_ratio']
#best_l1_ratio = 0.1
#clf = LogisticRegression(penalty='elasticnet',solver='saga', l1_ratio=0, random_state=0, max_iter=10000)
clf = LogisticRegression(random_state=0, tol=1e-03, max_iter=1000).fit(X_train, y_train)
clf.fit(X_train, y_train)

# evaluation
y_pred = clf.predict(X_test)
rf_probs = clf.predict_proba(X_test)[:,1]
roc_value = roc_auc_score(y_test, rf_probs)
print("AUC: ", roc_value)
metrics.plot_roc_curve(clf, X_test, y_test)
plt.show()

# feature importance
background_adult = shap.maskers.Independent(X_train, max_samples=100)
explainer = shap.Explainer(clf, background_adult)
shap_values_lg = explainer.shap_values(X_train)
shap_plots(shap_values_lg, 'logistic_regression')


# RANDOM FOREST CLASSIFIER -------------------------------------------------#


print("\n Random Forest Classifier")

# hyper parameter definition
params = {'n_estimators':1000}

# train model on subset of dataset
rf = RandomForestClassifier(random_state=0, n_estimators=params['n_estimators'])
rf.fit(X_train, y_train)

# evaluation
y_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:,1]
roc_value = roc_auc_score(y_test, rf_probs)
print("AUC: ", roc_value)
metrics.plot_roc_curve(rf, X_test, y_test)
plt.show()

# tree explainer feature importances
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(df_X_train)
shap_plots(shap_values, 'random_forest')
