import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, render_template

app = Flask(__name__)

# Load dataset
file_path = 'car_prices.csv'
df = pd.read_csv(file_path)

# Handle missing data
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['make', 'model', 'condition']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Scale numerical features
scaler = StandardScaler()
df[['year', 'mileage']] = scaler.fit_transform(df[['year', 'mileage']])

# Define features and target
X = df[['make', 'model', 'year', 'mileage', 'condition']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
lr = LinearRegression()
lr.fit(X_train, y_train)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

rmse_lr, r2_lr = evaluate_model(lr, X_test, y_test)
rmse_dt, r2_dt = evaluate_model(dt, X_test, y_test)
rmse_rf, r2_rf = evaluate_model(rf, X_test, y_test)

print(f"Linear Regression: RMSE={rmse_lr}, R²={r2_lr}")
print(f"Decision Tree: RMSE={rmse_dt}, R²={r2_dt}")
print(f"Random Forest: RMSE={rmse_rf}, R²={r2_rf}")

# Predict price function
def predict_price(make, model, year, mileage, condition):
    make_encoded = label_encoders['make'].transform([make])[0]
    model_encoded = label_encoders['model'].transform([model])[0]
    condition_encoded = label_encoders['condition'].transform([condition])[0]
    year_scaled = scaler.transform([[year]])[0][0]
    mileage_scaled = scaler.transform([[mileage]])[0][0]
    features = [[make_encoded, model_encoded, year_scaled, mileage_scaled, condition_encoded]]
    predicted_price = rf.predict(features)[0]
    return predicted_price

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    make = request.form['make']
    model = request.form['model']
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])
    condition = request.form['condition']

    predicted_price = predict_price(make, model, year, mileage, condition)
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
