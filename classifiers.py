from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.utils import resample
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np

import pandas as pd
from imblearn.over_sampling import SMOTE


def build_resampled_datasets(dataset):
    # Separate testing split from dataset
    df_train, df_test = train_test_split(dataset, test_size=0.2, stratify=dataset['oscar_winners'])

    # Separate classes
    df_majority = df_train[(df_train['oscar_winners']==0)].reset_index().drop('index', axis=1)
    df_minority = df_train[(df_train['oscar_winners']==1)].reset_index().drop('index', axis=1)

    # Upsample the minority class of the dataset
    df_minority_upsampled = resample(df_minority, replace=True, n_samples= len(df_majority), random_state=42)
    df_minority_upsampled = df_minority_upsampled.reset_index().drop('index', axis=1)
    df_upsampled = pd.concat([df_minority_upsampled, df_majority]).sort_index(kind='merge')
    df_upsampled = df_upsampled.reset_index().drop('index', axis=1)

    # Downsample the majority class of the dataset
    df_majority_downsampled = resample(df_majority, replace=True, n_samples= len(df_minority), random_state=42)
    df_majority_downsampled = df_majority_downsampled.reset_index().drop('index', axis=1)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sort_index(kind='merge')
    df_downsampled = df_downsampled.reset_index().drop('index', axis=1)

    # Resample using SMOTE method
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    df_smote_X, df_smote_Y = sm.fit_resample(df_train.drop('oscar_winners', axis=1), df_train['oscar_winners'])
    df_smote = pd.concat([pd.DataFrame(df_smote_X), pd.DataFrame(df_smote_Y)], axis=1)
    
    # Dataframes are returned
    return {
        'default' : dataset,
        'upsampled' : df_upsampled,
        'downsampled' : df_downsampled,
        'SMOTE' : df_smote
    }, df_test


def k_fold_fit_and_test(model, datasets, test, param=None):
    results = {}    


    for sampling, dataset in zip(datasets.keys(), datasets.values()):
        # Prepare dataset for training k-fold splits

        results[sampling] = []

        X = dataset.drop(['oscar_winners'], axis=1)
        y = dataset[['oscar_winners']]

        # Create k-fold splits and stratify classes
        kf = StratifiedKFold(n_splits=5, shuffle=False)

        for i, train_idx in enumerate(kf.split(X, y), 1):

            train = dataset.iloc[train_idx[1]] 
            
            # Separate labels
            X_train, y_train = train.drop(['oscar_winners'], axis=1).values, train[['oscar_winners']].values
            y_train = y_train.squeeze(axis=1)

            model.fit(X_train, y_train)
            
            # Test on test split
            X_test, y_test = test.drop(['oscar_winners'], axis=1).values, test[['oscar_winners']].values
            y_test = y_test.squeeze(axis=1)

            y_pred = model.predict(X_test)             
            
            # y_pred = [float(x) for x in y_pred]
            # y_test = [float(x) for x in y_test]
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)

            print(f"Fold {i} 1-F1 score: {report['1']['f1-score']:.4f} for {sampling}")
           
        # best_performing_knn_sampling = max(reports, key=lambda a:a['report']['1.0']['f1-score'])

            results[sampling].append({
                'fold' : str(i),
                'report' : report,
                'preds' : y_pred,
                'true' : y_test,
                'X_train' : X_train,
                'y_train' : y_train
            })

    return results

def k_fold_fit_and_test_knn(datasets, test, param=None):
    results = {}    

    for sampling, dataset in zip(datasets.keys(), datasets.values()):

        # Find best k parameter
        k_values = [i for i in range (2, 30)] 
        scores = []        

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, dataset.drop(['oscar_winners'], axis=1), dataset['oscar_winners'], cv=5)
            scores.append(np.mean(score))
        
        best_index = np.argmax(scores)
        best_k = k_values[best_index]

        # Prepare dataset for training k-fold splits

        results[sampling] = []
        

        X = dataset.drop(['oscar_winners'], axis=1)
        y = dataset[['oscar_winners']]


        # Create k-fold splits and stratify classes
        kf = StratifiedKFold(n_splits=5, shuffle=False)

        for i, train_idx in enumerate(kf.split(X, y), 1):

            train = dataset.iloc[train_idx[1]] 
            
            # Separate labels
            X_train, y_train = train.drop(['oscar_winners'], axis=1).values, train[['oscar_winners']].values
            y_train = y_train.squeeze(axis=1)

            model = KNeighborsClassifier(n_neighbors=best_k)
            model.fit(X_train, y_train)
            
            # Test on test split
            X_test, y_test = test.drop(['oscar_winners'], axis=1).values, test[['oscar_winners']].values
            y_test = y_test.squeeze(axis=1)

            y_pred = model.predict(X_test)             
            
            # y_pred = [float(x) for x in y_pred]
            # y_test = [float(x) for x in y_test]
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)

            print(f"Fold {i} 1-F1 score: {report['1']['f1-score']:.4f} for {sampling}")
           
        # best_performing_knn_sampling = max(reports, key=lambda a:a['report']['1.0']['f1-score'])

            results[sampling].append({
                'fold' : str(i),
                'report' : report,
                'preds' : y_pred,
                'true' : y_test,
                'best_k' : best_k,
                'X_train' : X_train,
                'y_train' : y_train
            })

    return results


def k_fold_fit_and_test_rf(datasets, test, param=None):
    results = {}    

    for sampling, dataset in zip(datasets.keys(), datasets.values()):

        # Find best k parameter
        k_values = [i for i in range (2, 15)] 
        scores = []

        

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, dataset.drop(['oscar_winners'], axis=1), dataset['oscar_winners'], cv=5)
            scores.append(np.mean(score))

        
        best_index = np.argmax(scores)
        best_k = k_values[best_index]

        # Prepare dataset for training k-fold splits

        results[sampling] = []
        

        X = dataset.drop(['oscar_winners'], axis=1)
        y = dataset[['oscar_winners']]


        # Create k-fold splits and stratify classes
        kf = StratifiedKFold(n_splits=5, shuffle=False)

        for i, train_idx in enumerate(kf.split(X, y), 1):

            train = dataset.iloc[train_idx[1]] 
            
            # Separate labels
            X_train, y_train = train.drop(['oscar_winners'], axis=1).values, train[['oscar_winners']].values
            y_train = y_train.squeeze(axis=1)

            model = RandomForestClassifier(max_depth=best_k)
            model.fit(X_train, y_train)
            
            # Test on test split
            X_test, y_test = test.drop(['oscar_winners'], axis=1).values, test[['oscar_winners']].values
            y_test = y_test.squeeze(axis=1)

            y_pred = model.predict(X_test)             
            
            # y_pred = [float(x) for x in y_pred]
            # y_test = [float(x) for x in y_test]
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)

            print(f"Fold {i} 1-F1 score: {report['1']['f1-score']:.4f} for {sampling}")
           
        # best_performing_knn_sampling = max(reports, key=lambda a:a['report']['1.0']['f1-score'])

            results[sampling].append({
                'fold' : str(i),
                'report' : report,
                'preds' : y_pred,
                'true' : y_test,
                'best_k' : best_k,
                'X_train' : X_train,
                'y_train' : y_train
            })

    return results

def best_f1_score_for_each_sampling_method(results, model):
    best = {}
    best['default'] = max(results[model]['default'], key=lambda a:a['report']['1']['f1-score'])
    best['upsampled'] = max(results[model]['upsampled'], key=lambda a:a['report']['1']['f1-score'])
    best['downsampled'] = max(results[model]['downsampled'], key=lambda a:a['report']['1']['f1-score'])
    best['SMOTE'] = max(results[model]['SMOTE'], key=lambda a:a['report']['1']['f1-score'])
    return best