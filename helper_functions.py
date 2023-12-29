from sklearn.metrics import roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
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

    plt.plot(fpr_a, tpr_a, label=f'{best_a["label"]} AUC: {auc(fpr_a, tpr_a):.2}')
    plt.plot(fpr_b, tpr_b, label=f'{best_b["label"]} AUC: {auc(fpr_b, tpr_b):.2}')
    plt.plot(fpr_c, tpr_c, label=f'{best_c["label"]} AUC: {auc(fpr_c, tpr_c):.2}')
    plt.plot(fpr_d, tpr_d, label=f'{best_d["label"]} AUC: {auc(fpr_d, tpr_d):.2}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()
    
'''
def confusion_matrices(best_a, best_b, best_c, title):

    y_test, y_test_pred = torch.round(torch.tensor((best_y_true))), torch.round(torch.tensor((best_y_pred)))
    cf_matrix = confusion_matrix(y_test, y_test_pred)
    sns.reset_orig()
    # sns.set_theme()
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=['0. non-oscar','1. oscar'])
    disp.plot()
    plt.title('Confusion Matrix for Neural Network model predictions')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(14,4))


    ax[0].plot(fpr_up, tpr_up, label=f'AUC: {roc_auc_up}')
    # ax[0].plot(range(1, epochs+1), v_acc, '--', label='Ευστοχία επικύρωσης')
    # ax[0].set_xticks(range(1, epochs+1))
    # ax[0].set_ylim(0, 1.1)
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title(titles[0])
    ax[0].legend()

    
    ax[1].plot(fpr_dwn, tpr_dwn, label=f'AUC: {roc_auc_dwn}')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(titles[1])
    ax[1].legend()

    ax[2].plot(fpr_smote, tpr_smote, label=f'AUC: {roc_auc_smote}')
    ax[2].set_xlabel('False Positive Rate')
    ax[2].set_ylabel('True Positive Rate')
    ax[2].set_title(titles[1])
    ax[2].legend()

    plt.suptitle('Roc Curves')
    plt.show()
'''

def avg_classification_reports(reports, title):

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

    # report_a = classification_report(torch.round(torch.tensor(best_a['true'])), torch.round(torch.tensor(best_a['preds'])), output_dict=True, target_names=['0. non-oscar', '1. oscar'])
    # report_b = classification_report(torch.round(torch.tensor(best_b['true'])), torch.round(torch.tensor(best_b['preds'])), output_dict=True, target_names=['0. non-oscar', '1. oscar'])
    # report_c = classification_report(torch.round(torch.tensor(best_c['true'])), torch.round(torch.tensor(best_c['preds'])), output_dict=True, target_names=['0. non-oscar', '1. oscar'])

    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.reset_orig()
    sns.set_theme()
    sns.set(rc={'figure.figsize':(8, 6)})


    sns.heatmap(pd.DataFrame(avg_reports[0]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True).set_title(avg_reports[0]['label'])
    sns.heatmap(pd.DataFrame(avg_reports[1]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True).set_title(avg_reports[1]['label'])
    sns.heatmap(pd.DataFrame(avg_reports[2]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True).set_title(avg_reports[2]['label'])
    sns.heatmap(pd.DataFrame(avg_reports[3]['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True).set_title(avg_reports[3]['label'])
    plt.suptitle(title)

    plt.show()