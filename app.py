from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib

app = Flask(__name__)

# ✅ Load encoders and scaler
le_location = joblib.load('le_location.pkl')
le_season = joblib.load('le_season.pkl')
scaler = joblib.load('scaler_dl.pkl')

# ✅ Define Neural Network (same as training)
class WeatherNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.model(x)

# ✅ Load trained model
model = WeatherNN()
model.load_state_dict(torch.load('weather_nn.pth'))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # ✅ Encode categorical inputs (MUST match training)
            season = le_season.transform([request.form['season']])[0]
            location = le_location.transform([request.form['location']])[0]

            # ✅ Collect features (same order as training)
            features = [
                float(request.form['temperature']),
                int(request.form['humidity']),
                float(request.form['wind_speed']),
                float(request.form['precipitation']),
                int(request.form['cloud_cover']),
                int(request.form['uv_index']),
                season,
                float(request.form['visibility']),
                location
            ]

            # ✅ Scale features
            features = scaler.transform([features])
            features = torch.tensor(features, dtype=torch.float32)

            # ✅ Predict
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

            # ✅ Route to result pages
            if prediction == 0:
                return render_template('rainy.html')
            elif prediction == 1:
                return render_template('cloudy.html')
            elif prediction == 2:
                return render_template('sunny.html')
            elif prediction == 3:
                return render_template('snowy.html')

        except Exception as e:
            print("Error:", e)
            return render_template('index.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
