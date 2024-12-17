import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import time

# Загрузка датасета
file_path = 'your_dataset.csv' 
df = pd.read_csv(file_path)

# Разделение данных на признаки и метки
X = df.drop(columns=['fraud'])
y = df['fraud']

# Масштабирование признаков (обязательно для MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Определение гиперпараметров для GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Разные конфигурации скрытых слоев
    'activation': ['relu', 'tanh', 'logistic'],  # Функции активации
    'solver': ['adam', 'lbfgs', 'sgd'],  # Алгоритмы оптимизации
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Скорость обучения
    'max_iter': [200, 300]  # Максимальное количество итераций
}

# Инициализация модели MLP
mlp = MLPClassifier(random_state=42)

# Инициализация GridSearchCV для подбора гиперпараметров
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)

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
