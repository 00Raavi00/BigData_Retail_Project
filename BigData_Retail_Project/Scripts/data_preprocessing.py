import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Retail Data Preprocessing") \
    .getOrCreate()

# Load the raw data from Google Cloud Storage
data = pd.read_csv('gs://retail-data-bucket/raw_data/store_customers.csv')

# Data Exploration
print("Initial Data Overview:")
print(data.head())

# Data Cleaning
# Handle missing values (dropping rows with any missing values)
data_clean = data.dropna()

# Remove duplicates
data_clean = data_clean.drop_duplicates()

# Feature Engineering
# Create 'High_Spender' label: Spending Amount > 1000 is a High Spender
data_clean['High_Spender'] = (data_clean['Spending_Amount'] > 1000).astype(int)

# Convert 'Gender' to numeric (Male = 1, Female = 0)
data_clean['Gender'] = data_clean['Gender'].map({'Male': 1, 'Female': 0})

# Handle outliers in 'Salary' using 1st and 99th percentiles
salary_lower = data_clean['Salary'].quantile(0.01)
salary_upper = data_clean['Salary'].quantile(0.99)
data_clean['Salary'] = data_clean['Salary'].clip(lower=salary_lower, upper=salary_upper)

# Select relevant features for model training
selected_features = data_clean[['Age', 'Salary', 'Gender', 'High_Spender']]

# Convert to Spark DataFrame
spark_data = spark.createDataFrame(data_clean)

# Save cleaned data to Google Cloud Storage (GCS)
spark_data.write.csv('gs://retail-data-bucket/processed_data/cleaned_data.csv', header=True)

# End of preprocessing script
