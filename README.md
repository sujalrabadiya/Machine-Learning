# Placement Prediction Using Machine Learning

## 🚀 Project Overview
This project predicts whether a student will get placed based on various academic and skill-related features. It uses **Naive Bayes** as the best-performing model and is deployed using **Flask API on Render** with a frontend hosted on **GitHub Pages**.

## 📂 Project Structure
```
Machine-Learning/
│── placement-prediction/
|   │── model-comparision.py        # Training and Model Selection
|   │── app.py                      # Flask Backend (Deployed on Render)
|   │── placementdata.csv           # Dataset used for training
|   │── requirements.txt            # Dependencies for Flask
|   │── Procfile                    # Deployment instructions for Render
|   │── README.md                   # Project Documentation
│── placement-prediction.html  # Frontend UI (GitHub Pages)
```

## 🎯 Features
- **Machine Learning Model:** Naive Bayes
- **Backend:** Flask (Deployed on Render)
- **Frontend:** HTML, Bootstrap, JavaScript (Hosted on GitHub Pages)
- **API Integration:** GitHub Pages fetches predictions from Flask API

## 🛠️ Setup Instructions
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/sujalrabadiya/Machine-Learning.git
cd Machine-Learning
```
### 2️⃣ Install Dependencies (For Local Testing)
```sh
pip install -r placement-prediction/requirements.txt
```
### 3️⃣ Run Flask API Locally
```sh
python placement-prediction/app.py
```
### 4️⃣ Access the Frontend
- If running locally, open `placement-prediction.html` in a browser.

## 📊 Machine Learning Model Training
This project evaluates **7 different machine learning models** and selects the best-performing one based on accuracy. The models include:
- **Linear Regression**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors**
- **Support Vector Machine (SVM)**
- **Naive Bayes** (Best Model: **Accuracy = 79.35%**)
- **Gradient Boosting**

📌 **Full Comparative Study & Model Training Script:**  
[View `model-comparision.py` on GitHub](https://github.com/sujalrabadiya/Machine-Learning/blob/main/placement-prediction/model-comparision.py)

### 🔹 Training & Model Selection Process
1. **Data Preprocessing**:
   - Load `placementdata.csv`
   - Encode categorical variables (`ExtracurricularActivities`, `PlacementTraining`, etc.)
   - Apply `StandardScaler` for feature scaling
2. **Train & Evaluate Models**:
   - Train all models on `X_train, y_train`
   - Evaluate using `accuracy_score` for classification models
   - Compare results and select the best-performing model
3. **Save the Trained Model**:
   ```python
   import joblib
   joblib.dump(models['Naive Bayes'], 'naive_bayes_model.pkl')
   joblib.dump(scaler, 'scaler.pkl')
   ```
   
**Example Request:**
```json
{
    "features": [8.5, 2, 3, 85, 4.5, 1, 1, 80, 85, 3]
}
```
**Example Response:**
```json
{
    "prediction": "Placed"
}
```

