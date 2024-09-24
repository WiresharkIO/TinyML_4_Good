import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    def train_svm_linear():
        svc = LinearSVC(C=0.1, dual=False, loss='squared_hinge', penalty='l2', class_weight={0: 1, 1: 10}, max_iter=10000,
                        verbose=True)
        svc.fit(X_train_standardized, y_train)
        return svc

    svc_linear= train_svm_linear()

    try:

        # joblib.dump(svc_linear, modelFolder + 'best_svc_linear_3s.joblib')
        # print("model saved as 'best_svm_model.joblib'")
        #
        # joblib.dump(scaler, modelFolder + 'svc_linear_scaler_3s.joblib')
        # print("Scaler saved as 'scaler.joblib'")

        y_pred = svc_linear.predict(X_test_standardized)

        accuracy_overall = accuracy_score(y_test, y_pred)
        f1_overall = f1_score(y_test, y_pred)
        precision_overall = precision_score(y_test, y_pred)
        recall_overall = recall_score(y_test, y_pred)

        print(f"Accuracy: {accuracy_overall:.4f}")
        print(f"Precision: {f1_overall:.4f}")
        print(f"Recall: {precision_overall:.4f}")
        print(f"F1-score: {recall_overall:.4f}")

        # report = classification_report(y_test, y_pred, output_dict=True)
        # f1_score_class_1 = report['1']['f1-score']
        # print(f"F1-score: {f1_score_class_1:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(modelFolder + 'confusion_matrix_svc_linear.png')
        plt.show()
        plt.close()

        print("Ended SVM Training at: " + str(datetime.now()))

        svm_fi = np.abs(svc_linear.coef_).sum(axis=0)
        svm_fr = np.argsort(-svm_fi)

        mrmr_features = mrmr_classif(X=featuresDataframe_MRMR, y=labelDataframe_MRMR, K=30)

        fea_weighting = [features_list[i] for i in svm_fr[:30]]
        fea_mRMR = mrmr_features

        intersection = set(fea_weighting) & set(fea_mRMR)
        print(f"\nNumber of common features: {len(intersection)}")
        print("Common features:")
        print(list(intersection))


        try:
            explainer = shap.LinearExplainer(svc_linear, X_train_standardized)
            shap_values = explainer.shap_values(X_train_standardized)

            feature_names = [f'Feature {i}' for i in range(X_train_standardized.shape[1])]

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_train_standardized, plot_type="violin", feature_names=feature_names)
            plt.title(f"SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig('shap_feature_importance.png')
            plt.close()

            # Detailed summary plot
            plt.figure(figsize=(10, 12))
            shap.summary_plot(shap_values, X_train_standardized, plot_type="violin", feature_names=feature_names)
            plt.title(f"SHAP Summary Plot")
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
            print("Ended SVM Training at: " + str(datetime.now()))

    except Exception as e:
        print("An error occurred during model training:")
        print(traceback.format_exc())
        print("Ended SVM Training at: " + str(datetime.now()))

    print('------------------Training-Ended----------------------')

if __name__ == '__main__':
    print('------------------Training-Begins----------------------')
    call_svm()