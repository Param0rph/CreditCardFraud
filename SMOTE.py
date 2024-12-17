# Данный код аугментирует датасет методом SMOTE, после чего перемешивает строки


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

# Загрузка несбалансированного датасета
file_path = 'your_dataset.csv'  
df = pd.read_csv(file_path)

# Разделение признаков и меток
X = df.drop(columns=['fraud'])  # Удаляем колонку с меткой класса
y = df['fraud']  # Целевая переменная (метка класса)

# Применение SMOTE для расширения минорного класса
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Создание нового сбалансированного датасета
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['fraud'] = y_resampled  # Добавляем колонку с метками класса

# Перемешивание строк
df_resampled = shuffle(df_resampled, random_state=42)

# Сохранение итогового датасета в новый CSV-файл
df_resampled.to_csv('balanced_dataset.csv', index=False)

print("Минорный класс расширен, строки перемешаны, результат сохранен в 'balanced_dataset.csv'")
