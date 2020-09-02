import numpy as np
import pandas as pd

def compute_confusion_matrix(actual, predicted):
    
    pd_actual = pd.Series(actual, name='Actual')
    pd_predicted = pd.Series(predicted, name='Predicted')

    CM =  pd.crosstab(pd_actual, pd_predicted)
    return CM

def compute_F1_score(actual, predicted):
    
    precision = compute_precision(actual, predicted)
    recall = compute_recall(actual, predicted)
    
    F1_score = 2 * precision * recall / (precision + recall)
    
    return F1_score

def compute_precision(actual, predicted):
    
    pd_actual = pd.Series(actual, name='Actual')
    pd_predicted = pd.Series(predicted, name='Predicted')
    
    CM =  pd.crosstab(pd_actual, pd_predicted).to_numpy()   # CM is converted into a 2 X 2 array.
    
    TN = CM[0,0]; FP = CM[0,1]; FN = CM[1,0]; TP =  CM[1,1];
    
    precision = TP / (TP + FP)
    
    return precision

def compute_recall(actual, predicted):
    
    pd_actual = pd.Series(actual, name='Actual')
    pd_predicted = pd.Series(predicted, name='Predicted')
    
    CM =  pd.crosstab(pd_actual, pd_predicted).to_numpy()   # CM is converted into a 2 X 2 array.
    
    TN = CM[0,0]; FP = CM[0,1]; FN = CM[1,0]; TP =  CM[1,1];
    
    recall = TP / (TP + FN)
    
    return recall
