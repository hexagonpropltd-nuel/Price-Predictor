# PricePredictor
python
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Function to train and save the model (run once at startup)
def train_save_model():
    # Example training data
    data = pd.DataFrame({
        'size': [1500, 1800, 2400, 3200, 3600],
        'bedrooms': [3, 4, 3, 5, 4],
        'location_encoded': [1, 2, 3, 1, 2],
        'price': [400000, 500000, 600000, 650000, 700000]
    })

    X = data[['size', 'bedrooms', 'location_encoded']]
    y = data['price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'price_predictor_model.pkl')

# Train and save the model when app starts (comment this out after first run)
train_save_model()

app = Flask(__name__)

# Load the saved model
model = joblib.load('price_predictor_model.pkl')

# HTML form for user input
form_html = '''
<!DOCTYPE html>
<html>
<head><title>Real Estate Price Predictor</title></head>
<body>
<h2>Real Estate Price Predictor</h2>
<form method="POST">
  Size (sqft): <input type="number" name="size" step="0.01" required><br><br>
  Bedrooms: <input type="number" name="bedrooms" required><br><br>
  Location Index (0-5): <input type="number" name="location" min="0" max="5" required><br><br>
  <input type="submit" value="Predict Price">
</form>

{% if prediction %}
<h3>Predicted Price: ${{ prediction }}</h3>
{% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        location = int(request.form['location'])

        features = [[size, bedrooms, location]]
        pred_price = model.predict(features)
        prediction = f"{pred_price:,.2f}"

    return render_template_string(form_html, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
```

**How to run:**

1. Save this as `app.py` in your project folder.  
2. Install dependencies with:  
   `pip install flask scikit-learn pandas joblib`  
3. Run the app:  
   `python app.py`  
4. Open `http://127.0.0.1:5000` in your browser to test!
