
# Big Data Retail Project

## Overview
This project demonstrates the application of **Big Data** techniques and **machine learning** algorithms in the **retail industry** to predict customer spending behavior. We utilize **Google Cloud Platform (GCP)** for cloud computing, with **Apache Spark**, **Hadoop**, and **Hive** to process large datasets and build machine learning models.

The main objective of this project is to:
- Analyze customer data to predict **High Spenders**.
- Use **Big Data tools** such as **Apache Spark** and **Hive** for data processing.
- Train machine learning models (e.g., **Random Forest**, **SVM**, **KNN**) to predict customer spending behavior.
- Evaluate model performance using appropriate metrics (e.g., **accuracy**, **precision**, **recall**, **ROC curve**).

## Folder Structure
The folder structure of the project is as follows:

```plaintext
BigData_Retail_Project/
│
├── data/
│   ├── raw_data/               # Contains the raw dataset in CSV format
│   ├── processed_data/         # Contains cleaned data ready for model training
│
├── scripts/                    # Python scripts for preprocessing, model training, evaluation, and visualization
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── visualize_results.py
│   └── hive_queries.sql
│
├── models/                     # Trained models saved in pickle format
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   └── knn_model.pkl
│
├── README.md                  # Project overview and instructions
└── requirements.txt           # List of required Python libraries
```

## Setup

1. **Clone the Repository**:

```bash
git clone https://github.com/your-repo-link.git
cd BigData_Retail_Project
```

2. **Set up Google Cloud Platform (GCP)**:
    - Create a project in **Google Cloud Platform**.
    - Set up a **Google Dataproc** cluster for managing **Apache Hadoop** and **Apache Spark** environments.
    - Store your raw dataset in **Google Cloud Storage** under the path `gs://retail-data-bucket/raw_data/`.

3. **Install the Required Libraries**:
    - Install the necessary libraries by running:

```bash
pip install -r requirements.txt
```

4. **Run the Scripts**:
    - After setting up your environment, run the following scripts in order:

    1. **`data_preprocessing.py`**: Cleans and preprocesses the data and uploads it to Google Cloud Storage.
    2. **`model_training.py`**: Trains machine learning models (Random Forest, SVM, KNN).
    3. **`model_evaluation.py`**: Evaluates the models using various metrics and visualizes the performance.
    4. **`visualize_results.py`**: Generates feature importance plots and displays model evaluation results.

## Libraries

This project uses the following libraries:

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computing library.
- `pyspark`: Apache Spark Python API for big data processing.
- `scikit-learn`: Machine learning models and metrics.
- `matplotlib`: Plotting library for visualizations.
- `seaborn`: Statistical data visualization.
- `joblib`: For saving and loading trained models.

## Running the Scripts

### **`data_preprocessing.py`**:
- Cleans the dataset, handles missing values, creates new features like `High_Spender`, and stores the cleaned data in Google Cloud Storage (GCS).

### **`model_training.py`**:
- Trains three machine learning models: **Random Forest**, **SVM**, and **KNN**.
- Evaluates their performance and saves the trained models in the `models/` directory.

### **`model_evaluation.py`**:
- Evaluates the trained models using **accuracy**, **precision**, **recall**, **F1-score**, and **ROC curves**.
- Plots the **confusion matrices** and **ROC curves** for each model.

### **`visualize_results.py`**:
- Generates feature importance plots for the **Random Forest** model and displays various evaluation results.

## License

This project is licensed under the *
