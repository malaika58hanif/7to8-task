from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp_min = float(request.form['temp_min'])
        temp_max = float(request.form['temp_max'])
        precipitation = float(request.form['precipitation'])
        wind = float(request.form['wind'])

        features = np.array([[temp_min, temp_max, precipitation, wind]])
        prediction = model.predict(features)[0]
        predicted_weather = le.inverse_transform([prediction])[0]

        return render_template('result.html', weather=predicted_weather)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
