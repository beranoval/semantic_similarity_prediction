import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_plot(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()
    

def fi_plot(trained_model,X):
    feature_importance = trained_model.feature_importances_
    sorted_indices = feature_importance.argsort()[::-1]
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance[sorted_indices], align='center')
    plt.xticks(range(len(feature_importance)), feature_names[sorted_indices], rotation=90)
    plt.xlabel('Function')
    plt.ylabel('Importance')
    plt.show()
