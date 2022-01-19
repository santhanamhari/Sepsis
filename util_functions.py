# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 00:11:04 2021

@author: hsanthanam
"""
import numpy as np
import pandas as pd

def age_adjusted(ages_in_days, weights_min, feature_name):
    ages = ages_in_days.copy()
    weights = weights_min.copy()
    
    # convert ages to years
    ages_in_years = (ages/365).astype('int')
    
    # compute normal weights for years 2 to 21
    normal_weight = {} 
    for year in range(2, 22): # 2 years to 21 years
        w = weights[ages_in_years == year]
        normal_weight[str(year) + ' years'] = w.mean()
    
    # compute normal weights for 0 to 2 years (based on month)
    days_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    intervals = np.cumsum(days_month)
    
    for year in range(0, 2):    
        if year == 1:
            intervals = intervals + 365
        
        current_ages = ages[ages_in_years == year]
        current_weights = weights[ages_in_years == year]
        
        for month in range(0, 12):
            cond = (current_ages > intervals[month]) & (current_ages <= intervals[month + 1])
            w = current_weights[cond]
            
            if year == 0:
                normal_weight[str(month) + ' months'] = w.mean()
            elif year == 1:
                normal_weight[str(12 + month) + ' months'] = w.mean()

    #--------------------------------------------------------------------------#
    # create age adjusted weight feature
    age_adjusted_weight_list = []
    for i in range(len(ages)):
        
        age_year = int(ages[i]/ 365)
        
        if age_year > 21:
            age_adjusted_weight_list.append(weights[i])
            continue
        
        elif age_year >= 2:
            new_weight = (weights[i] - normal_weight[str(age_year) + ' years']) / normal_weight[str(age_year) + ' years']
        
        elif age_year == 0:
            days_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            intervals = np.cumsum(days_month)
            for month in range(0, 12):
                if ages[i] > intervals[month] and ages[i] <= intervals[month+1]:
                    new_weight = (weights[i] - normal_weight[str(month) + ' months']) / normal_weight[str(month) + ' months']
                
        elif age_year == 1:
            days_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) + 365
            intervals = np.cumsum(days_month)
            for month in range(0, 12):
                if ages[i] > intervals[month] and ages[i] <= intervals[month+1]:
                    new_weight = (weights[i] - normal_weight[str(month + 12) + ' months']) / normal_weight[str(month + 12) + ' months']
            
        age_adjusted_weight_list.append(new_weight)
        
    age_adjusted_weight_df = pd.DataFrame(age_adjusted_weight_list,
                                              columns =[feature_name])

    return age_adjusted_weight_df