import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = pd.read_csv('placementdata.csv')

# Data preprocessing (Handling missing values, encoding, scaling)
data.drop(columns=['StudentID'], inplace=True)  # Remove unnecessary ID column
label_encoder = LabelEncoder()
data['ExtracurricularActivities'] = label_encoder.fit_transform(data['ExtracurricularActivities'])
data['PlacementTraining'] = label_encoder.fit_transform(data['PlacementTraining'])
data['PlacementStatus'] = label_encoder.fit_transform(data['PlacementStatus'])

X = data.drop(['PlacementStatus'], axis=1)
y = data['PlacementStatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if name == 'Linear Regression':  # Use regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2 Score': r2}
    else:  # Use classification metrics
        y_pred = np.round(y_pred)  # Ensure predictions are binary
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'Accuracy': accuracy}

# Print results
for model, metrics in results.items():
    print(f"{model}: {metrics}")

# Select the best model (ignore regression models)
classification_results = {k: v['Accuracy'] for k, v in results.items() if 'Accuracy' in v}
best_model = max(classification_results, key=classification_results.get)
print(f"Best Model: {best_model}")

# Select the best model and Save the Trained Model
# Here Best Model is Naive Bayes with Accuracy: 0.7935
import joblib

# Save the trained model
joblib.dump(models['Naive Bayes'], 'naive_bayes_model.pkl')

# Save the scaler for consistent input processing
joblib.dump(scaler, 'scaler.pkl')