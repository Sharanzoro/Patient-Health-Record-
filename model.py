import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set environment variables to fix joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# Load the dataset
file_path = ("patient_dataset.csv"
             ""
             ""
             "")
df = pd.read_csv(file_path, encoding='latin1')

# Drop unnecessary columns
df_cleaned = df.drop(columns=['Name', 'Contact'])

# Fill missing values using loc to avoid FutureWarning
df_cleaned.loc[:, 'Past Illnesses'] = df_cleaned['Past Illnesses'].fillna('None')
df_cleaned.loc[:, 'Surgeries'] = df_cleaned['Surgeries'].fillna('None')
df_cleaned.loc[:, 'Family Medical History'] = df_cleaned['Family Medical History'].fillna('None')

# Encode categorical variables
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

# Define target variable
y = df_encoded['Lab Results_Normal']
X = df_encoded.drop(columns=['Lab Results_Normal'])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model with balanced class weights
model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42, n_jobs=1)
model.fit(X_train_scaled, y_train)

# Save the model and scaler as .pkl files
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate accuracy and report
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))