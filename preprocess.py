# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:37:10 2021

@author: hsanthanam
"""

import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# set condition to just keep first values
first = False

# download the sepsis pickle file to pandas dataframe df
df = pd.read_pickle('C:/Users/hsanthanam/Documents/Hari_Sepsis/data/raw_sepsis_data')
# removing following featuers
x = df.pop('PedScreens_Id')
x = df.pop('loadDate')
x = df.pop('SubmissionYYYYMM')
x = df.pop('DataSetId')


# extract the outcome variable from dataframe
df_y = df.pop('hospDeath')
df_y[df_y == 'No'] = 0
df_y[df_y == 'Yes'] = 1
df_y = df_y.astype('int')

    
# categorical feature list
feature_cat = ['SiteAcronym', 'Race', 'Ethnicity', 'Sex', 'Gender', 
               'EDDayOfWeek', 'TriageCategory', 'PrimaryPayer', 'ArrivalMode',
               'AdmittingSource', 'SpokenLanguage', 'WrittenLanguage', 'PreferredLanguage',
               'TempCRoute_max', 'PainScoreType_first', 'Interpreter']
 
# numerical feature list
feature_num = []
for col in df:
    if col not in feature_cat:
        if first and ('last' in col or 'Min' in col or 'Max' in col or 'Mean' in col or 'StdDev' in col or 'best' in col or 'worst' in col):
            continue
        feature_num.append(col)
        

# create dataframe for numerical and categorical
df_num = df[feature_num]
df_cat = df[feature_cat]

# handle categorical variables
df_cat_dummy = pd.get_dummies(df_cat, df_cat.columns, dummy_na=True)

# need to make sure all features are floats and missing values are NaNs
df_num = df_num.apply(pd.to_numeric)
for col in df_num:
    df_num[col].values[df_num[col].values == None] = float('NaN')

# handle numerical variables
df_num_dummy = df_num[df_num.columns].isna().astype('float').add_suffix('_indicator')

def mygen(lst):
    for item in lst:
        yield item
        yield item + '_indicator'

# standardize numerical data
df_num_impute = df_num.fillna(df_num.mean())
scaler = StandardScaler()
scaler.fit(df_num_impute)
scaler.mean_
standard_values = scaler.transform(df_num_impute)
for i, col in enumerate(df_num_impute):
    df_num_impute[col] = standard_values[:, i]

df_num_dummy = pd.concat([df_num_impute, df_num_dummy], axis=1).reindex(list(mygen(df_num.columns)), axis=1)

# merge final dataframe
df_X = df_cat_dummy.merge(df_num_dummy, left_index=True, right_index=True)

# remove all NaN features (Venous Pressure/ Systolic Pressure)
for col in df_X:
    if df_X[col].isna().sum() == df_X.shape[0]:
        df_X = df_X.drop([col], axis=1)

# fix columns names 
for i, col in enumerate(df_X.columns):

    list_col = col.split('_')
    
    # change it so that count values all have '_N'
    if 'n' in list_col:
        list_col[list_col.index('n')] = 'N'
        col_new = '_'.join(list_col)
        df_X = df_X.rename(columns={col: col_new})
        
    # change certain feature blocks names to make them uniform
    if 'GcsTotal' in col:
        new = 'GCSTotal'
    elif 'ETCO2' in col or 'Etco2' in col:
        new = 'Etco2'
    elif 'PEEP' in col or 'Peep' in col:
        new = 'Peep'
    elif 'PIP' in col or 'Pip' in col:
        new = 'Pip'
    elif 'PainScoreType_first' in col:
        new = 'PainScoreType-first'
        col_new = col.replace('PainScoreType_first', new)
        df_X = df_X.rename(columns={col: col_new})
        continue
    elif 'TempCRoute_max' in col:
        new = 'TempCRoute-max'
        col_new = col.replace('TempCRoute_max', new)
        df_X = df_X.rename(columns={col: col_new})
        continue
    else:
        continue

    col_new = col.replace(list_col[0], new)
    df_X = df_X.rename(columns={col: col_new})
    


# save this new dataframe as pickle file
df_X.to_pickle('C:/Users/hsanthanam/Documents/Hari_Sepsis/data/processed_sepsis_data')
df_y.to_pickle('C:/Users/hsanthanam/Documents/Hari_Sepsis/data/label_data')

