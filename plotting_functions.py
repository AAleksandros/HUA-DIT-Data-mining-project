from sklearn.metrics import roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


def plot_roc_curves(best_a, best_b, best_c, best_d, title):
    fpr_a, tpr_a, thresholds = roc_curve(best_a['true'].flatten(), best_a['preds'].flatten())
    fpr_b, tpr_b, thresholds = roc_curve(best_b['true'].flatten(), best_b['preds'].flatten())
    fpr_c, tpr_c, thresholds = roc_curve(best_c['true'].flatten(), best_c['preds'].flatten())
    fpr_d, tpr_d, thresholds = roc_curve(best_d['true'].flatten(), best_d['preds'].flatten())
    sns.reset_orig()
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_a, tpr_a, label=f'{best_a["label"]} AUC: {auc(fpr_a, tpr_a):.2}')
    plt.plot(fpr_b, tpr_b, label=f'{best_b["label"]} AUC: {auc(fpr_b, tpr_b):.2}')
    plt.plot(fpr_c, tpr_c, label=f'{best_c["label"]} AUC: {auc(fpr_c, tpr_c):.2}')
    plt.plot(fpr_d, tpr_d, label=f'{best_d["label"]} AUC: {auc(fpr_d, tpr_d):.2}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.title(title)
    plt.legend()
    plt.show()
    
    
def plot_classification_reports_averages(reports, title):
    avg_reports = []
    for report in reports:
        avg_reports.append({
            'label' : report[0]['label'],
            'report' : {
                '0.0' : {
                    'precision' :  np.average([x['report']['0.0']['precision'] for x in report]),
                    'recall': np.average([x['report']['0.0']['recall'] for x in report]),
                    'f1-score' : np.average([x['report']['0.0']['f1-score'] for x in report]),
                    'support': np.average([x['report']['0.0']['support'] for x in report]),
                },
                '1.0' : {
                    'precision' :  np.average([x['report']['1.0']['precision'] for x in report]),
                    'recall': np.average([x['report']['1.0']['recall'] for x in report]),
                    'f1-score' : np.average([x['report']['1.0']['f1-score'] for x in report]),
                    'support': np.average([x['report']['1.0']['support'] for x in report]),
                },
                'accuracy' : np.average([x['report']['accuracy'] for x in report]),
                'macro avg': {
                    'precision' : np.average([x['report']['macro avg']['precision'] for x in report]),
                    'recall' : np.average([x['report']['macro avg']['recall'] for x in report]),
                    'f1-score' : np.average([x['report']['macro avg']['f1-score'] for x in report]),
                    'support' : np.average([x['report']['macro avg']['support'] for x in report]),
                },
                'weighted avg' : {
                    'precision': np.average([x['report']['weighted avg']['precision'] for x in report]),
                    'recall' : np.average([x['report']['weighted avg']['recall'] for x in report]),
                    'f1-score' : np.average([x['report']['weighted avg']['f1-score'] for x in report]),
                    'support' : np.average([x['report']['weighted avg']['support'] for x in report]),
                }
                }
            })

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    sns.reset_orig()
    sns.set_theme()

    sns.heatmap(pd.DataFrame(avg_reports[0]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True).set_title(avg_reports[0]['label'])
    sns.heatmap(pd.DataFrame(avg_reports[1]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True).set_title(avg_reports[1]['label'])
    sns.heatmap(pd.DataFrame(avg_reports[2]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True).set_title(avg_reports[2]['label'])
    sns.heatmap(pd.DataFrame(avg_reports[3]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True).set_title(avg_reports[3]['label'])
    plt.suptitle(title)
    plt.show()


def plot_confusion_matrices(models, title):
    cf_matrices = [confusion_matrix(model['true'], torch.round(torch.sigmoid(torch.tensor(model['preds'])))) for model in models]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    sns.reset_orig()
    sns.set_theme()

    sns.heatmap(pd.DataFrame(cf_matrices[0]), fmt='g', annot=True, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True, linewidths=0.5, linecolor='black').set_title(models[0]['label'])
    sns.heatmap(pd.DataFrame(cf_matrices[1]), fmt='g', annot=True, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True, linewidths=0.5, linecolor='black').set_title(models[1]['label'])
    sns.heatmap(pd.DataFrame(cf_matrices[2]), fmt='g', annot=True, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True, linewidths=0.5, linecolor='black').set_title(models[2]['label'])
    sns.heatmap(pd.DataFrame(cf_matrices[3]), fmt='g', annot=True, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True, linewidths=0.5, linecolor='black').set_title(models[3]['label'])
    
    plt.suptitle(title)
    plt.show()



'''
# fig, ax = plt.subplots()
# ax.plot(range(len(x_points)), y_points, 'r-')

# ax.errorbar(range(len(x_points)), y_points, xerr=0, yerr=error_bars, ecolor = 'b',fmt='ro')
# ax.set_xticks(range(len(x_points)), [str(x) for x in x_points])
# ax.set_ylim(-0.1, 1.1)
# plt.show()
'''



'''
# error_bars_min = []
# error_bars_max = []
# x_points = []
# y_points = []

# for ratio in balance_ratios:
#     scores = []
#     for item in results_nn:
#         if item['ratio'] == ratio:
#             # scores.append(item['classification_report']['accuracy'])
#             scores.append(item['accuracy'])
    
#     x_points.append(ratio)
#     y_points.append(np.average(scores))
#     error_bars_min.append(min(scores))
#     error_bars_max.append(max(scores))

# error_bars = [[],[]]

# for min_p, max_p, center in zip(error_bars_min, error_bars_max, y_points):
#     error_bars[0].append(center - min_p)
#     error_bars[1].append(max_p - center)
'''