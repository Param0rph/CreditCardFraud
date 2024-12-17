# Данный код аугментирует код методом GAN, после чего перемешивает строки

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

# Загрузка несбалансированного датасета
file_path = 'your_dataset.csv'  # Укажите путь к вашему CSV-файлу
df = pd.read_csv(file_path)

# Разделение данных на признаки и метки
X = df.drop(columns=['fraud'])
y = df['fraud']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на минорный и мажорный классы
X_minority = X_scaled[y == 1]
X_majority = X_scaled[y == 0]

# Параметры GAN
noise_dim = 10  # Размерность входного шума для генератора
data_dim = X_minority.shape[1]  # Количество признаков
batch_size = 128
epochs = 5000
learning_rate = 0.0002

# Создание генератора
def build_generator():
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(64),
        LeakyReLU(0.2),
        Dense(128),
        LeakyReLU(0.2),
        Dense(data_dim, activation='tanh')
    ])
    return model

# Создание дискриминатора
def build_discriminator():
    model = Sequential([
        Input(shape=(data_dim,)),
        Dense(128),
        LeakyReLU(0.2),
        Dense(64),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# Компиляция GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])

# Построение полной модели GAN
discriminator.trainable = False  # Замораживаем дискриминатор при обучении генератора
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate))

# Обучение GAN
for epoch in range(epochs):
    # Генерация шума и синтетических данных
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    generated_data = generator.predict(noise, verbose=0)

    # Выбор случайных реальных данных минорного класса
    idx = np.random.randint(0, X_minority.shape[0], batch_size)
    real_data = X_minority[idx]

    # Создание меток для обучения дискриминатора
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Обучение дискриминатора
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Обучение генератора через замороженный дискриминатор
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Вывод прогресса
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Генерация новых данных для минорного класса
num_samples_to_generate = len(X_majority) - len(X_minority)  # Балансировка классов
noise = np.random.normal(0, 1, (num_samples_to_generate, noise_dim))
synthetic_data = generator.predict(noise, verbose=0)

# Обратное масштабирование
synthetic_data = scaler.inverse_transform(synthetic_data)

# Создание нового сбалансированного датасета
df_synthetic = pd.DataFrame(synthetic_data, columns=X.columns)
df_synthetic['fraud'] = 1  # Добавляем метку минорного класса

# Объединение синтетических и оригинальных данных
df_majority = df[y == 0]
df_minority = df[y == 1]
df_resampled = pd.concat([df_majority, df_minority, df_synthetic], axis=0)

# Перемешивание строк
df_resampled = shuffle(df_resampled, random_state=42)

# Сохранение сбалансированного датасета
df_resampled.to_csv('balanced_dataset_gan.csv', index=False)
print("Минорный класс расширен с использованием GAN, результат сохранен в 'balanced_dataset_gan.csv'")
