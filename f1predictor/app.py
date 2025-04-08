{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
from sklearn.model_selection import train_test_split\
from sklearn.preprocessing import StandardScaler, OneHotEncoder\
from sklearn.compose import ColumnTransformer\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.metrics import accuracy_score, classification_report\
from sklearn.pipeline import Pipeline\
import xgboost as xgb\
from sklearn.model_selection import GridSearchCV\
import joblib\
import requests\
from datetime import datetime\
from bs4 import BeautifulSoup\
import tensorflow as tf\
from tensorflow.keras.models import Sequential\
from tensorflow.keras.layers import Dense, LSTM, Dropout\
from tensorflow.keras.optimizers import Adam\
from tensorflow.keras.callbacks import EarlyStopping\
from flask import Flask, render_template, jsonify\
\
app = Flask(__name__)\
\
# Data fetching and feature engineering functions (same as before)\
def fetch_data():\
    # From Ergast API\
    url = "http://ergast.com/api/f1/current.json"\
    response = requests.get(url)\
    data = response.json()\
    return data\
\
def feature_engineering(data):\
    # Extract key features\
    races = data['MRData']['RaceTable']['Races']\
    features = []\
    for race in races:\
        race_data = \{\
            'raceName': race['raceName'],\
            'circuitId': race['Circuit']['circuitId'],\
            'date': race['date'],\
            'drivers': []\
        \}\
        for driver in race['Results']:\
            driver_data = \{\
                'driverId': driver['Driver']['driverId'],\
                'constructorId': driver['Constructor']['constructorId'],\
                'grid': int(driver['grid']),\
                'position': int(driver['position']),\
                'points': float(driver['points']),\
                'status': driver['status']\
            \}\
            race_data['drivers'].append(driver_data)\
        features.append(race_data)\
    return features\
\
def train_model(X, y):\
    # Data preprocessing\
    numeric_features = ['grid', 'points']\
    categorical_features = ['driverId', 'constructorId', 'circuitId']\
\
    preprocessor = ColumnTransformer(\
        transformers=[\
            ('num', StandardScaler(), numeric_features),\
            ('cat', OneHotEncoder(), categorical_features)\
        ])\
\
    # Create a pipeline with preprocessing and classifier\
    pipeline = Pipeline(steps=[\
        ('preprocessor', preprocessor),\
        ('classifier', RandomForestClassifier())\
    ])\
\
    # Split data\
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\
\
    # Train model\
    pipeline.fit(X_train, y_train)\
\
    # Evaluate model\
    y_pred = pipeline.predict(X_test)\
    print("Accuracy:", accuracy_score(y_test, y_pred))\
    print("Classification Report:")\
    print(classification_report(y_test, y_pred))\
\
    return pipeline\
\
def hybrid_model_predict(X, pipeline):\
    # Traditional ML prediction\
    ml_prediction = pipeline.predict(X)\
\
    # Deep learning model\
    dl_model = Sequential()\
    dl_model.add(Dense(128, activation='relu', input_dim=X.shape[1]))\
    dl_model.add(Dropout(0.5))\
    dl_model.add(Dense(64, activation='relu'))\
    dl_model.add(Dropout(0.3))\
    dl_model.add(Dense(1, activation='sigmoid'))\
\
    dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])\
\
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)\
\
    dl_model.fit(X, ml_prediction, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])\
\
    dl_prediction = dl_model.predict(X)\
\
    # Combine predictions\
    final_prediction = (ml_prediction + dl_prediction) / 2\
\
    return final_prediction\
\
def get_real_time_data():\
    # From F1 official website\
    url = "https://www.formula1.com"\
    response = requests.get(url)\
    soup = BeautifulSoup(response.text, 'html.parser')\
    real_time_data = \{\
        'weather': soup.find('span', class_='weather').text,\
        'track_condition': soup.find('span', class_='track-condition').text\
    \}\
    return real_time_data\
\
def chaos_factor_simulation(real_time_data):\
    # Simulate chaos factors\
    chaos_factors = \{\
        'accident': np.random.choice([0, 1], p=[0.95, 0.05]),\
        'safety_car': np.random.choice([0, 1], p=[0.9, 0.1])\
    \}\
    return chaos_factors\
\
@app.route('/')\
def index():\
    return render_template('index.html')\
\
@app.route('/predict')\
def predict():\
    # Get data\
    data = fetch_data()\
    features = feature_engineering(data)\
\
    # Prepare data\
    X = []\
    y = []\
    for race in features:\
        for driver in race['drivers']:\
            X.append(\{\
                'driverId': driver['driverId'],\
                'constructorId': driver['constructorId'],\
                'circuitId': race['circuitId'],\
                'grid': driver['grid'],\
                'points': driver['points']\
            \})\
            y.append(1 if driver['position'] == 1 else 0)\
\
    X = pd.DataFrame(X)\
    y = np.array(y)\
\
    # Train model\
    pipeline = train_model(X, y)\
\
    # Get real-time data\
    real_time_data = get_real_time_data()\
\
    # Simulate chaos factors\
    chaos_factors = chaos_factor_simulation(real_time_data)\
\
    # Predict next race winner\
    next_race_features = X.iloc[-5:]  # Assume we predict recent 5 race winners\
    ml_prediction = pipeline.predict(next_race_features)\
    dl_prediction = hybrid_model_predict(next_race_features, pipeline)\
\
    # Combine with chaos factors\
    final_prediction = dl_prediction * (1 + chaos_factors['accident'] * 0.1 + chaos_factors['safety_car'] * 0.05)\
\
    # Prepare response\
    driver_names = ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Sergio Perez", "Carlos Sainz"]\
    predictions = []\
    for driver, prob in zip(driver_names, final_prediction):\
        predictions.append(\{\
            'driver': driver,\
            'probability': f"\{prob * 100:.1f\}%"\
        \})\
\
    return jsonify(predictions)\
\
if __name__ == "__main__":\
    app.run(debug=True)}