from sklearn.metrics import roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


def plot_roc_curves(results, title):
    sns.reset_orig()
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    
    for key, result in zip(results.keys(), results.values()):
        fpr, tpr, thr = roc_curve(result['true'], result['preds'])
        plt.plot(fpr, tpr, label=f'{key} AUC: {auc(fpr, tpr):.2}')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.title(f"{title} ROC curves")
    plt.legend()
    plt.show()


def plot_classification_reports_averages(reports, name):
    avg_reports = []
    model = reports[name]
    for technique in model:
        avg_reports.append({
            'sampling' : technique,
            'report' : {
                '0' : {
                    'precision' : np.average([x['report']['0']['precision'] for x in model[technique]]),
                    'recall': np.average([x['report']['0']['recall'] for x in model[technique]]),
                    'f1-score' : np.average([x['report']['0']['f1-score'] for x in model[technique]]),
                    'support': np.average([x['report']['0']['support'] for x in model[technique]]),
                },
                '1' : {
                    'precision' :  np.average([x['report']['1']['precision'] for x in model[technique]]),
                    'recall': np.average([x['report']['1']['recall'] for x in model[technique]]),
                    'f1-score' : np.average([x['report']['1']['f1-score'] for x in model[technique]]),
                    'support': np.average([x['report']['1']['support'] for x in model[technique]]),
                },
                'accuracy' : np.average([x['report']['accuracy'] for x in model[technique]]),
                'macro avg': {
                    'precision' : np.average([x['report']['macro avg']['precision'] for x in model[technique]]),
                    'recall' : np.average([x['report']['macro avg']['recall'] for x in model[technique]]),
                    'f1-score' : np.average([x['report']['macro avg']['f1-score'] for x in model[technique]]),
                    'support' : np.average([x['report']['macro avg']['support'] for x in model[technique]]),
                },
                'weighted avg' : {
                    'precision': np.average([x['report']['weighted avg']['precision'] for x in model[technique]]),
                    'recall' : np.average([x['report']['weighted avg']['recall'] for x in model[technique]]),
                    'f1-score' : np.average([x['report']['weighted avg']['f1-score'] for x in model[technique]]),
                    'support' : np.average([x['report']['weighted avg']['support'] for x in model[technique]]),
                }
                }
            })

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    sns.reset_orig()
    sns.set_theme()

    # for report, ax in zip(avg_reports, axes):
    sns.heatmap(pd.DataFrame(avg_reports[0]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True).set_title(avg_reports[0]['sampling'])
    sns.heatmap(pd.DataFrame(avg_reports[1]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True).set_title(avg_reports[1]['sampling'])
    sns.heatmap(pd.DataFrame(avg_reports[2]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True).set_title(avg_reports[2]['sampling'])
    sns.heatmap(pd.DataFrame(avg_reports[3]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True).set_title(avg_reports[3]['sampling'])
    plt.suptitle(f'{name} classification report')
    plt.show()


def plot_confusion_matrices(models, name):
    cf_matrices = [confusion_matrix(model['true'], model['preds']) for model in models.values()]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    sns.reset_orig()
    sns.set_theme()

    labels = [name for name in models]


    sns.heatmap(pd.DataFrame(cf_matrices[0]), fmt='g', annot=True, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[0])
    sns.heatmap(pd.DataFrame(cf_matrices[1]), fmt='g', annot=True, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[1])
    sns.heatmap(pd.DataFrame(cf_matrices[2]), fmt='g', annot=True, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[2])
    sns.heatmap(pd.DataFrame(cf_matrices[3]), fmt='g', annot=True, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[3])
    
    plt.suptitle(f'{name} - Confusion Matrices')
    plt.show()