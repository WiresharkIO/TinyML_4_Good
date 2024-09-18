import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
from definitions import ROOT_DIR
from os import walk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sys import platform
from joblib import Parallel, delayed
import traceback
import shap
import os
from mrmr import mrmr_classif

sourceFolder = ROOT_DIR
modelFolder = sourceFolder + '\\models\\sequence\\SVM_Linear_models\\'

isExist = os.path.exists(modelFolder)
if not isExist:
    os.makedirs(modelFolder)

featureFolder = sourceFolder + '\\models\\sequence\\chew_dataset_3s.csv'


def call_svm():
    print("Start SVM training at: " + str(datetime.now()))
    featuresDataframe = pd.read_csv(featureFolder)
    featuresDataframe.fillna(0)

    features_list = ['TD_MAX', 'TD_MIN', 'TD_MAX_MIN', 'TD_RMS', 'TD_MEDIAN', 'TD_VARIANCE', 'TD_STD', 'TD_SKEW',
                     'TD_KURT', 'TD_IQR', 'FD_MEAN', 'FD_POWB', 'FD_MEDIAN', 'TFD_MIN', 'TFD_MAX', 'TFD_PSD_MEAN',
                     'TFD_STD', 'TFD_S_ENT', 'TFD_S_KURT', 'TFD_KURT', 'TFD_SKEW', 'TFD_PSD_MEDIAN', 'TFD_AMP_KURT',
                     'TFD_AMP_SKEW', 'TFD_ERG_SUM', 'TFD_ERG_MIN', 'TFD_ERG_MAX', 'TFD_ERG_MEAN', 'TFD_ERG_Q1',
                     'TFD_ERG_Q2', 'TFD_ERG_Q3', 'TFD_ERG_Q4', 'TFD_ERG_1', 'TFD_ERG_2', 'TFD_ERG_3', 'TFD_ERG_4',
                     'TFD_ERG_5', 'TFD_ERG_6', 'TFD_ERG_7', 'TFD_CONCENTRATION']
    # print(len(features_list)) --> 40

    label = ['SEQUENCE_TRUTH_FRAME']

    featuresDataframe_MRMR = featuresDataframe[features_list]
    labelDataframe_MRMR = featuresDataframe[label]

    featuresArray = featuresDataframe[features_list].to_numpy()
    featuresArray[np.isnan(featuresArray)] = 0

    labelArray = featuresDataframe[label].to_numpy()
    labelArray = labelArray.flatten()

    X_train, X_test, y_train, y_test = train_test_split(featuresArray, labelArray, test_size=0.2,
                                                        random_state=42)

    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # joblib.dump(scaler, 'SVM_Linear_models/scaler.joblib')

    svc = LinearSVC(dual=False, loss='squared_hinge') # nsamples > nfeatures, so primal problem is solved
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2', 'l1'],
        # 'loss': ['hinge', 'squared_hinge'],
        # 'dual': [True, False],
        'class_weight': [{0: 1, 1: 10}, 'balanced'],
        'max_iter': [10000, 15000, 20000]
    }
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring=scoring, verbose=5, refit='f1', error_score='raise')

    try:

        grid_search.fit(X_train_standardized, y_train)

        best_params = grid_search.best_params_
        print("Best hyperparameters found:", best_params)
        # svc.fit(X_train_standardized, y_train)

        best_svc = grid_search.best_estimator_

        # Save the model
        joblib.dump(best_svc, modelFolder + 'best_svc_linear_3s.joblib')
        print("Best model saved as 'best_svm_model.joblib'")

        # Save the scaler
        joblib.dump(scaler, modelFolder + 'svc_linear_scaler_3s.joblib')
        print("Scaler saved as 'scaler.joblib'")

        y_pred = best_svc.predict(X_test_standardized)

        accuracy_overall = accuracy_score(y_test, y_pred)
        f1_overall = f1_score(y_test, y_pred)
        precision_overall = precision_score(y_test, y_pred)
        recall_overall = recall_score(y_test, y_pred)

        accuracy_ClassOne = accuracy_score(y_test, y_pred, pos_label=1)
        f1_ClassOne = f1_score(y_test, y_pred, pos_label=1)
        precision_ClassOne = precision_score(y_test, y_pred, pos_label=1)
        recall_ClassOne = recall_score(y_test, y_pred, pos_label=1)

        print(f"Accuracy: {accuracy_overall:.4f}")
        print(f"Precision: {f1_overall:.4f}")
        print(f"Recall: {precision_overall:.4f}")
        print(f"F1-score: {recall_overall:.4f}")

        print(f"Accuracy: {accuracy_ClassOne:.4f}")
        print(f"Precision: {f1_ClassOne:.4f}")
        print(f"Recall: {precision_ClassOne:.4f}")
        print(f"F1-score: {recall_ClassOne:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(modelFolder + 'confusion_matrix_svmW_mMRMR12_rbf.png')
        plt.close()

        print("Ended SVM Training at: " + str(datetime.now()))

        svm_fi = np.abs(svc.coef_).sum(axis=0)
        svm_fr = np.argsort(-svm_fi)

        mrmr_features = mrmr_classif(X=featuresDataframe_MRMR, y=labelDataframe_MRMR, K=30)

        fea_weighting = [features_list[i] for i in svm_fr[:30]]
        fea_mRMR = mrmr_features

        intersection = set(fea_weighting) & set(fea_mRMR)
        print(f"\nNumber of common features: {len(intersection)}")
        print("Common features:")
        print(list(intersection))


        try:
            explainer = shap.LinearExplainer(best_svc, X_train_standardized)
            shap_values = explainer.shap_values(X_train_standardized)

            # Create feature names
            feature_names = [f'Feature {i}' for i in range(X_train_standardized.shape[1])]

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_train_standardized, plot_type="bar", feature_names=feature_names)
            plt.title(f"SHAP Feature Importance (Best Model: {best_params['penalty']} penalty)")
            plt.tight_layout()
            plt.savefig('shap_feature_importance.png')
            plt.close()

            # Detailed summary plot
            plt.figure(figsize=(10, 12))
            shap.summary_plot(shap_values, X_train_standardized, feature_names=feature_names)
            plt.title(f"SHAP Summary Plot (Best Model: {best_params['penalty']} penalty)")
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png')
            plt.close()

            # SHAP dependence plot for the most important feature
            most_important_feature = np.argmax(np.abs(shap_values).mean(0))
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(most_important_feature, shap_values, X_train_standardized,
                                 feature_names=feature_names)
            plt.title(f"SHAP Dependence Plot for {feature_names[most_important_feature]}")
            plt.tight_layout()
            plt.savefig('shap_dependence_plot.png')
            plt.close()

            print("SHAP analysis completed and plots saved.")

        except Exception as e:
            print("An error occurred during SHAP analysis:")
            print(traceback.format_exc())
            print("But don't worry, the best model and scaler are still saved.")

    except Exception as e:
        print("An error occurred during model training:")
        print(traceback.format_exc())

    print('------------------Training-Ended----------------------')

if __name__ == '__main__':
    print('------------------Training-Begins----------------------')
    call_svm()