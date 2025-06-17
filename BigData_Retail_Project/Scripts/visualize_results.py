import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Random Forest model
rf_model = joblib.load('models/rf_model.pkl')

# Get the feature importances
importances = rf_model.feature_importances_
features = ['Age', 'Salary']

# Plot Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=importances)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
