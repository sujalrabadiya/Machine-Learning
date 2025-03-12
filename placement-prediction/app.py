from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://sujalrabadiya.github.io"}})

# Load trained model and scaler
model = joblib.load('naive_bayes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data)[0]
        result = 'Placed' if prediction == 1 else 'Not Placed'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask in background in Jupyter Notebook
def run_app():
    app.run(debug=True, use_reloader=False)

thread = threading.Thread(target=run_app)
thread.start()
