import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the cleaned data
data = pd.read_csv('gs://retail-data-bucket/processed_data/cleaned_data.csv')

# Features and target variable
X = data[['Age', 'Salary']]
y = data['High_Spender']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Save models to disk
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(knn_model, 'models/knn_model.pkl')

# Evaluate models
metrics = {}
for model, name in zip([rf_model, svm_model, knn_model], ["Random Forest", "SVM", "KNN"]):
    predictions = model.predict(X_test)
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1 Score': f1_score(y_test, predictions)
    }

# Print evaluation metrics
print(metrics)
