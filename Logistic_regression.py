import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import time

# Загрузка датасета
file_path = 'your_dataset.csv'  # Укажите путь к вашему CSV-файлу
df = pd.read_csv(file_path)

# Разделение данных на признаки и метки
X = df.drop(columns=['fraud'])
y = df['fraud']

# Масштабирование признаков (необходимо для Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Определение гиперпараметров для GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Параметр регуляризации
    'penalty': ['l2'],  # Тип регуляризации (для логистической регрессии)
    'solver': ['liblinear', 'lbfgs']  # Решатели для оптимизации
}

# Инициализация модели Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Инициализация GridSearchCV для подбора гиперпараметров
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)

# Засекаем время обучения
start_time = time.time()

# Обучение модели с подбором гиперпараметров
grid_search.fit(X_train, y_train)

# Засекаем время предсказания
end_time = time.time()

# Время обучения и предсказания
training_time = end_time - start_time
prediction_time = (end_time - start_time) / len(X_test)  # Время предсказания на одну строку

# Лучшая модель после подбора гиперпараметров
best_model = grid_search.best_estimator_

# Предсказания на тестовой выборке
y_pred = best_model.predict(X_test)

# Вычисление метрик
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Вывод результатов
print(f"Лучшие гиперпараметры: {grid_search.best_params_}")
print(f"Время обучения модели: {training_time:.4f} секунд")
print(f"Среднее время предсказания на одну строку: {prediction_time:.4e} секунд")
print(f"F1-Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
