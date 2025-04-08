import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 数据获取函数
def fetch_data():
    # 从Ergast API获取数据
    url = "http://ergast.com/api/f1/current.json"
    response = requests.get(url)
    data = response.json()
    return data

# 特征工程函数
def feature_engineering(data):
    # 提取关键特征
    races = data['MRData']['RaceTable']['Races']
    features = []
    for race in races:
        race_data = {
            'raceName': race['raceName'],
            'circuitId': race['Circuit']['circuitId'],
            'date': race['date'],
            'drivers': []
        }
        for driver in race['Results']:
            driver_data = {
                'driverId': driver['Driver']['driverId'],
                'constructorId': driver['Constructor']['constructorId'],
                'grid': int(driver['grid']),
                'position': int(driver['position']),
                'points': float(driver['points']),
                'status': driver['status']
            }
            race_data['drivers'].append(driver_data)
        features.append(race_data)
    return features

# 模型训练函数
def train_model(X, y):
    # 数据预处理
    numeric_features = ['grid', 'points']
    categorical_features = ['driverId', 'constructorId', 'circuitId']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # 创建一个包含预处理和分类器的管道
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 预测
    y_pred = pipeline.predict(X_test)

    # 评估模型
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline

# 混合模型预测函数
def hybrid_model_predict(X, pipeline):
    # 使用传统机器学习模型进行预测
    ml_prediction = pipeline.predict(X)

    # 使用深度学习模型进行预测
    dl_model = Sequential()
    dl_model.add(Dense(128, activation='relu', input_dim=X.shape[1]))
    dl_model.add(Dropout(0.5))
    dl_model.add(Dense(64, activation='relu'))
    dl_model.add(Dropout(0.3))
    dl_model.add(Dense(1, activation='sigmoid'))

    dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    dl_model.fit(X, ml_prediction, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    dl_prediction = dl_model.predict(X)

    # 结合两种模型的预测结果
    final_prediction = (ml_prediction + dl_prediction) / 2

    return final_prediction

# 实时数据获取函数
def get_real_time_data():
    # 从F1官方网站获取实时数据
    url = "https://www.formula1.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # 提取实时数据，如天气、赛道状况等
    real_time_data = {
        'weather': soup.find('span', class_='weather').text,
        'track_condition': soup.find('span', class_='track-condition').text
    }
    return real_time_data

# 混乱因子模拟函数
def chaos_factor_simulation(real_time_data):
    # 模拟混乱因子，如事故、安全车等
    chaos_factors = {
        'accident': np.random.choice([0, 1], p=[0.95, 0.05]),
        'safety_car': np.random.choice([0, 1], p=[0.9, 0.1])
    }
    return chaos_factors

# 主函数
def main():
    # 获取数据
    data = fetch_data()
    features = feature_engineering(data)

    # 准备数据
    X = []
    y = []
    for race in features:
        for driver in race['drivers']:
            X.append({
                'driverId': driver['driverId'],
                'constructorId': driver['constructorId'],
                'circuitId': race['circuitId'],
                'grid': driver['grid'],
                'points': driver['points']
            })
            y.append(1 if driver['position'] == 1 else 0)

    X = pd.DataFrame(X)
    y = np.array(y)

    # 训练模型
    pipeline = train_model(X, y)

    # 获取实时数据
    real_time_data = get_real_time_data()

    # 模拟混乱因子
    chaos_factors = chaos_factor_simulation(real_time_data)

    # 预测下一场F1比赛的获胜者
    next_race_features = X.iloc[-5:]  # 假设我们预测最近5场比赛的获胜者
    ml_prediction = pipeline.predict(next_race_features)
    dl_prediction = hybrid_model_predict(next_race_features, pipeline)

    # 结合混乱因子
    final_prediction = dl_prediction * (1 + chaos_factors['accident'] * 0.1 + chaos_factors['safety_car'] * 0.05)

    # 输出预测结果
    driver_names = ["Driver A", "Driver B", "Driver C", "Driver D", "Driver E"]
    for driver, prob in zip(driver_names, final_prediction):
        print(f"{driver}: {prob * 100:.1f}% chance to win")

    # 保存模型
    joblib.dump(pipeline, 'f1_predictor_model.pkl')

if __name__ == "__main__":
    main()
