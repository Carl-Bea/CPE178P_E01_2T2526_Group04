from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the models and scaler
models = {
    'Logistic Regression': joblib.load('Logistic Regression.pkl'),
    'K-Nearest Neighbors': joblib.load('K-Nearest Neighbors.pkl'),
    'Decision Tree': joblib.load('Decision Tree.pkl'),
    'Random Forest': joblib.load('Random Forest.pkl'),
    'Support Vector Machine': joblib.load('Support Vector Machine.pkl')
}
scaler = joblib.load('scale.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get selected model
        selected_model_name = request.form['model']
        model = models.get(selected_model_name)
        
        if model is None:
            return render_template('home.html')  # Handle error case
        
        # Get user inputs
        features = [
            float(request.form['temperature']),
            int(request.form['humidity']),
            float(request.form['wind_speed']),
            float(request.form['precipitation']),
            int(request.form['cloud_cover']),
            int(request.form['uv_index']),
            int(request.form['season']),
            float(request.form['visibility']),
            int(request.form['location'])
        ]
        
        features = [features]  # Convert to 2D array
        
        # Standardize features
        features = scaler.transform(features)
        
        # Predict with the selected model
        prediction = model.predict(features)[0]
        
        # Return results to appropriate HTML page based on prediction
        if prediction == 0:
            return render_template('rainy.html')
        elif prediction == 1:
            return render_template('cloudy.html')
        elif prediction == 2:
            return render_template('sunny.html')
        elif prediction == 3:
            return render_template('snowy.html')
        else:
            return render_template('home.html')
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
