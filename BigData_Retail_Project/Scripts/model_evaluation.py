import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd

# Load models
rf_model = joblib.load('models/rf_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

# Load the cleaned data
data = pd.read_csv('gs://retail-data-bucket/processed_data/cleaned_data.csv')

# Features and target variable
X = data[['Age', 'Salary']]
y = data['High_Spender']

# Predictions from all models
rf_predictions = rf_model.predict(X)
svm_predictions = svm_model.predict(X)
knn_predictions = knn_model.predict(X)

# Confusion Matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Spender', 'High Spender'], yticklabels=['Low Spender', 'High Spender'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices
rf_cm = confusion_matrix(y, rf_predictions)
svm_cm = confusion_matrix(y, svm_predictions)
knn_cm = confusion_matrix(y, knn_predictions)

plot_confusion_matrix(rf_cm, 'Random Forest')
plot_confusion_matrix(svm_cm, 'SVM')
plot_confusion_matrix(knn_cm, 'KNN')

# ROC Curve
def plot_roc_curve(fpr, tpr, auc_value, model_name):
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Compute ROC curve and AUC for each model
rf_fpr, rf_tpr, _ = roc_curve(y, rf_model.predict_proba(X)[:, 1])
svm_fpr, svm_tpr, _ = roc_curve(y, svm_model.predict_proba(X)[:, 1])
knn_fpr, knn_tpr, _ = roc_curve(y, knn_model.predict_proba(X)[:, 1])

# Plot ROC curves
plot_roc_curve(rf_fpr, rf_tpr, auc(rf_fpr, rf_tpr), 'Random Forest')
plot_roc_curve(svm_fpr, svm_tpr, auc(svm_fpr, svm_tpr), 'SVM')
plot_roc_curve(knn_fpr, knn_tpr, auc(knn_fpr, knn_tpr), 'KNN')
