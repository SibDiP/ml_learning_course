# День 6: Современные подходы в ML и начало работы над проектом

## Расписание дня

| Время | Активность |
|-------|------------|
| 09:00 - 10:30 | Теория: Генеративные модели и обучение с подкреплением |
| 10:30 - 10:45 | Перерыв |
| 10:45 - 12:15 | Теория: MLOps и развертывание моделей |
| 12:15 - 13:15 | Обед |
| 13:15 - 15:00 | Практика: Инструменты для отслеживания экспериментов и упаковка моделей |
| 15:00 - 15:15 | Перерыв |
| 15:15 - 17:00 | Практика: Начало работы над финальным проектом |
| 17:00 - 18:00 | Работа над домашним заданием и вопросы |

## Теоретические материалы

### Генеративные модели

#### Введение в генеративные модели
- Отличия генеративных и дискриминативных моделей
- Основные типы генеративных моделей
- Применение генеративных моделей
- Современные достижения в области генеративных моделей

#### Вариационные автоэнкодеры (VAE)
- Принцип работы автоэнкодеров
- Латентное пространство
- Вариационные автоэнкодеры и их отличия от обычных автоэнкодеров
- Функция потерь VAE
- Генерация новых данных с помощью VAE

#### Генеративно-состязательные сети (GAN)
- Архитектура GAN
- Генератор и дискриминатор
- Процесс обучения GAN
- Проблемы обучения GAN (режим коллапса, нестабильность)
- Разновидности GAN (DCGAN, WGAN, StyleGAN, CycleGAN)
- Применение GAN для генерации изображений, текста, музыки

#### Диффузионные модели
- Принцип работы диффузионных моделей
- Прямой и обратный процессы диффузии
- Обучение диффузионных моделей
- Сравнение с VAE и GAN
- Современные диффузионные модели (DALL-E, Stable Diffusion)

### Обучение с подкреплением (Reinforcement Learning)

#### Основы обучения с подкреплением
- Агент, среда, состояния, действия, награды
- Марковский процесс принятия решений
- Функция ценности и функция политики
- Исследование и использование (exploration vs exploitation)
- Применение обучения с подкреплением

#### Алгоритмы обучения с подкреплением
- Q-learning
- Deep Q-Network (DQN)
- Policy Gradient
- Actor-Critic методы
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)

#### Примеры применения обучения с подкреплением
- Игры (AlphaGo, OpenAI Five)
- Робототехника
- Оптимизация ресурсов
- Рекомендательные системы
- Автономное вождение

### MLOps: развертывание и мониторинг ML-моделей

#### Введение в MLOps
- Определение MLOps
- Жизненный цикл ML-проекта
- Отличия от DevOps
- Основные компоненты MLOps
- Преимущества внедрения MLOps

#### Отслеживание экспериментов
- Важность отслеживания экспериментов
- Инструменты для отслеживания экспериментов (MLflow, Weights & Biases, TensorBoard)
- Логирование метрик, параметров и артефактов
- Сравнение экспериментов
- Воспроизводимость экспериментов

#### Упаковка и развертывание моделей
- Сериализация моделей
- Контейнеризация с Docker
- Создание API для моделей (Flask, FastAPI)
- Развертывание в облачных сервисах
- Масштабирование и балансировка нагрузки

#### Мониторинг моделей
- Мониторинг производительности модели
- Обнаружение дрейфа данных
- Мониторинг инфраструктуры
- Алерты и уведомления
- Инструменты для мониторинга

#### CI/CD для ML-проектов
- Непрерывная интеграция для ML
- Непрерывное развертывание ML-моделей
- Автоматизация тестирования моделей
- Управление версиями моделей и данных
- Инструменты для CI/CD в ML

### Этические аспекты и ответственное использование ML

#### Этические проблемы в ML
- Предвзятость и дискриминация в моделях
- Конфиденциальность данных
- Прозрачность и объяснимость моделей
- Безопасность и надежность систем ML
- Социальные последствия автоматизации

#### Методы обеспечения справедливости в ML
- Выявление и устранение предвзятости в данных
- Алгоритмические подходы к справедливости
- Методы объяснимого ИИ (XAI)
- Регулирование и стандарты в области ИИ
- Лучшие практики ответственного использования ML

## Практические задания

### Задание 1: Реализация простого вариационного автоэнкодера (VAE)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Загрузка данных MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Нормализация и изменение формы данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Параметры VAE
latent_dim = 2  # Размерность латентного пространства
input_shape = (28, 28, 1)

# Создание энкодера
encoder_inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

# Параметры латентного пространства
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Слой выборки
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

# Создание энкодера как модели
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Создание декодера
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

# Создание декодера как модели
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# Создание VAE как модели
outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, outputs, name="vae")

# Функция потерь VAE
def vae_loss(x, x_decoded):
    reconstruction_loss = keras.losses.binary_crossentropy(
        tf.keras.layers.Flatten()(x), tf.keras.layers.Flatten()(x_decoded)
    )
    reconstruction_loss *= 28 * 28
    
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    return total_loss

# Компиляция модели
vae.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss)

# Обучение модели
history = vae.fit(
    x_train, x_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, x_test)
)

# Визуализация процесса обучения
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('VAE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Визуализация латентного пространства
def plot_latent_space(encoder, decoder):
    # Отображение сетки точек в латентном пространстве
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Создание сетки значений в латентном пространстве
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size,
                   j * digit_size : (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.title("Latent Space Visualization")
    plt.axis("off")
    plt.show()

# Визуализация кодирования цифр в латентном пространстве
def plot_label_clusters(encoder, data, labels):
    # Отображение кластеров цифр в латентном пространстве
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap="rainbow")
    plt.colorbar()
    plt.title("Latent Space Clusters")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

# Визуализация латентного пространства
plot_latent_space(encoder, decoder)

# Визуализация кластеров в латентном пространстве
(_, y_test), _ = keras.datasets.mnist.load_data()
plot_label_clusters(encoder, x_test, y_test)
```

### Задание 2: Знакомство с инструментами для отслеживания экспериментов (MLflow)

```python
# Установка MLflow
!pip install mlflow

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Настройка MLflow
mlflow.set_experiment("RandomForest-Iris")

# Функция для обучения модели с разными гиперпараметрами
def train_model(n_estimators, max_depth, min_samples_split):
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # Обучение модели
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Сохранение модели
        mlflow.sklearn.log_model(model, "model")
        
        # Создание и сохранение графика важности признаков
        feature_importance = pd.DataFrame({
            'Feature': iris.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Сохранение графика
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        return model, accuracy

# Запуск нескольких экспериментов с разными гиперпараметрами
params_list = [
    {"n_estimators": 10, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 5},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 3}
]

results = []

for params in params_list:
    model, accuracy = train_model(**params)
    results.append({**params, "accuracy": accuracy})

# Вывод результатов
results_df = pd.DataFrame(results)
print(results_df.sort_values('accuracy', ascending=False))

# Запуск UI MLflow (в реальном окружении)
print("Для просмотра результатов экспериментов запустите: mlflow ui")
```

### Задание 3: Упаковка модели в Docker-контейнер

```python
# Создание простой модели для демонстрации
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
import os

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Создание директории для модели
os.makedirs("model", exist_ok=True)

# Сохранение модели
joblib.dump(model, "model/iris_model.joblib")

# Создание файла app.py для Flask API
with open("app.py", "w") as f:
    f.write("""
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Загрузка модели
model = joblib.load("model/iris_model.joblib")

# Создание Flask приложения
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API"

@app.route("/predict", methods=["POST"])
def predict():
    # Получение данных из запроса
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    
    # Предсказание
    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0].tolist()
    
    # Классы
    class_names = ["setosa", "versicolor", "virginica"]
    
    # Формирование ответа
    response = {
        "prediction": class_names[prediction],
        "prediction_id": prediction,
        "probabilities": {
            class_names[i]: float(prob) for i, prob in enumerate(probability)
        }
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
""")

# Создание файла requirements.txt
with open("requirements.txt", "w") as f:
    f.write("""
flask==2.0.1
joblib==1.0.1
numpy==1.21.0
scikit-learn==0.24.2
""")

# Создание Dockerfile
with open("Dockerfile", "w") as f:
    f.write("""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ ./model/

EXPOSE 5000

CMD ["python", "app.py"]
""")

# Вывод инструкций по сборке и запуску Docker-контейнера
print("Для сборки Docker-образа выполните:")
print("docker build -t iris-classifier .")
print("\nДля запуска контейнера выполните:")
print("docker run -p 5000:5000 iris-classifier")
print("\nПосле запуска API будет доступно по адресу http://localhost:5000")
print("\nПример запроса к API:")
print("""
import requests
import json

url = "http://localhost:5000/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Пример признаков для Iris setosa
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
""")
```

### Задание 4: Создание простого API для модели с использованием FastAPI

```python
# Создание файла main.py для FastAPI
with open("main.py", "w") as f:
    f.write("""
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Загрузка модели
model = joblib.load("model/iris_model.joblib")

# Создание FastAPI приложения
app = FastAPI(
    title="Iris Classifier API",
    description="API для классификации ирисов на основе их характеристик",
    version="1.0.0"
)

# Определение схемы входных данных
class IrisFeatures(BaseModel):
    features: List[float]

# Определение схемы выходных данных
class IrisPrediction(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: Dict[str, float]

@app.get("/")
def read_root():
    return {"message": "Iris Classifier API"}

@app.post("/predict", response_model=IrisPrediction)
def predict(iris: IrisFeatures):
    try:
        # Проверка входных данных
        if len(iris.features) != 4:
            raise HTTPException(status_code=400, detail="Требуется 4 признака")
        
        # Преобразование входных данных
        features = np.array(iris.features).reshape(1, -1)
        
        # Предсказание
        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0].tolist()
        
        # Классы
        class_names = ["setosa", "versicolor", "virginica"]
        
        # Формирование ответа
        response = IrisPrediction(
            prediction=class_names[prediction],
            prediction_id=prediction,
            probabilities={
                class_names[i]: float(prob) for i, prob in enumerate(probability)
            }
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
""")

# Обновление requirements.txt для FastAPI
with open("requirements_fastapi.txt", "w") as f:
    f.write("""
fastapi==0.68.0
uvicorn==0.15.0
joblib==1.0.1
numpy==1.21.0
scikit-learn==0.24.2
pydantic==1.8.2
""")

# Создание Dockerfile для FastAPI
with open("Dockerfile.fastapi", "w") as f:
    f.write("""
FROM python:3.9-slim

WORKDIR /app

COPY requirements_fastapi.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""")

# Вывод инструкций по сборке и запуску Docker-контейнера с FastAPI
print("Для сборки Docker-образа с FastAPI выполните:")
print("docker build -t iris-fastapi -f Dockerfile.fastapi .")
print("\nДля запуска контейнера выполните:")
print("docker run -p 8000:8000 iris-fastapi")
print("\nПосле запуска API будет доступно по адресу http://localhost:8000")
print("Документация API будет доступна по адресу http://localhost:8000/docs")
print("\nПример запроса к API:")
print("""
import requests
import json

url = "http://localhost:8000/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Пример признаков для Iris setosa
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
""")
```

### Задание 5: Начало работы над финальным проектом

```python
# Создание структуры проекта
import os

# Создание директорий проекта
project_dirs = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "reports/figures"
]

for directory in project_dirs:
    os.makedirs(directory, exist_ok=True)
    print(f"Создана директория: {directory}")

# Создание файла README.md
with open("README.md", "w") as f:
    f.write("""# ML Project

## Структура проекта

```
├── data
│   ├── processed      # Обработанные данные
│   └── raw            # Исходные данные
├── models             # Обученные модели
├── notebooks          # Jupyter notebooks
├── reports            # Отчеты и визуализации
│   └── figures        # Графики и диаграммы
└── src                # Исходный код
    ├── data           # Скрипты для загрузки и обработки данных
    ├── features       # Скрипты для создания признаков
    ├── models         # Скрипты для обучения и оценки моделей
    └── visualization  # Скрипты для визуализации
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

1. Загрузка и обработка данных:
```bash
python src/data/make_dataset.py
```

2. Создание признаков:
```bash
python src/features/build_features.py
```

3. Обучение модели:
```bash
python src/models/train_model.py
```

4. Оценка модели:
```bash
python src/models/evaluate_model.py
```

5. Визуализация результатов:
```bash
python src/visualization/visualize.py
```
""")

# Создание файла requirements.txt
with open("requirements.txt", "w") as f:
    f.write("""
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
matplotlib==3.4.2
seaborn==0.11.1
jupyter==1.0.0
tensorflow==2.5.0
mlflow==1.18.0
fastapi==0.68.0
uvicorn==0.15.0
python-dotenv==0.19.0
""")

# Создание файла .gitignore
with open(".gitignore", "w") as f:
    f.write("""
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Unit test / coverage reports
htmlcov/
.coverage
.coverage.*
.cache

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*
!models/.gitkeep

# Reports
reports/figures/*
!reports/figures/.gitkeep

# MLflow
mlruns/
""")

# Создание пустых файлов .gitkeep
for directory in project_dirs:
    with open(f"{directory}/.gitkeep", "w") as f:
        pass

# Создание базовых скриптов
scripts = {
    "src/data/make_dataset.py": """
import pandas as pd
import numpy as np
import os

def load_data(filepath):
    \"\"\"
    Загрузка данных из файла
    \"\"\"
    print(f"Загрузка данных из {filepath}")
    # Здесь код для загрузки данных
    
def preprocess_data(data):
    \"\"\"
    Предобработка данных
    \"\"\"
    print("Предобработка данных")
    # Здесь код для предобработки данных
    
def save_data(data, output_filepath):
    \"\"\"
    Сохранение обработанных данных
    \"\"\"
    print(f"Сохранение данных в {output_filepath}")
    # Здесь код для сохранения данных

def main():
    \"\"\"
    Основная функция для загрузки и обработки данных
    \"\"\"
    input_filepath = "data/raw/sample.csv"
    output_filepath = "data/processed/processed_data.csv"
    
    # Загрузка данных
    data = load_data(input_filepath)
    
    # Предобработка данных
    processed_data = preprocess_data(data)
    
    # Сохранение обработанных данных
    save_data(processed_data, output_filepath)
    
if __name__ == "__main__":
    main()
""",
    
    "src/features/build_features.py": """
import pandas as pd
import numpy as np

def create_features(data):
    \"\"\"
    Создание признаков из данных
    \"\"\"
    print("Создание признаков")
    # Здесь код для создания признаков
    
def select_features(data, features):
    \"\"\"
    Отбор признаков
    \"\"\"
    print("Отбор признаков")
    # Здесь код для отбора признаков
    
def main():
    \"\"\"
    Основная функция для создания признаков
    \"\"\"
    input_filepath = "data/processed/processed_data.csv"
    output_filepath = "data/processed/features.csv"
    
    # Загрузка данных
    data = pd.read_csv(input_filepath)
    
    # Создание признаков
    data_with_features = create_features(data)
    
    # Отбор признаков
    selected_features = ["feature1", "feature2", "feature3"]
    final_data = select_features(data_with_features, selected_features)
    
    # Сохранение данных с признаками
    final_data.to_csv(output_filepath, index=False)
    print(f"Признаки сохранены в {output_filepath}")
    
if __name__ == "__main__":
    main()
""",
    
    "src/models/train_model.py": """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def load_data(filepath):
    \"\"\"
    Загрузка данных с признаками
    \"\"\"
    print(f"Загрузка данных из {filepath}")
    data = pd.read_csv(filepath)
    return data
    
def split_data(data, target_column):
    \"\"\"
    Разделение данных на признаки и целевую переменную
    \"\"\"
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
    
def train_model(X_train, y_train):
    \"\"\"
    Обучение модели
    \"\"\"
    print("Обучение модели")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
    
def save_model(model, output_filepath):
    \"\"\"
    Сохранение модели
    \"\"\"
    print(f"Сохранение модели в {output_filepath}")
    joblib.dump(model, output_filepath)
    
def main():
    \"\"\"
    Основная функция для обучения модели
    \"\"\"
    input_filepath = "data/processed/features.csv"
    output_filepath = "models/model.joblib"
    target_column = "target"
    
    # Загрузка данных
    data = load_data(input_filepath)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    
    # Обучение модели
    model = train_model(X_train, y_train)
    
    # Сохранение модели
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    save_model(model, output_filepath)
    
if __name__ == "__main__":
    main()
""",
    
    "src/models/evaluate_model.py": """
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_model(model_filepath):
    \"\"\"
    Загрузка обученной модели
    \"\"\"
    print(f"Загрузка модели из {model_filepath}")
    model = joblib.load(model_filepath)
    return model
    
def load_data(filepath):
    \"\"\"
    Загрузка тестовых данных
    \"\"\"
    print(f"Загрузка данных из {filepath}")
    data = pd.read_csv(filepath)
    return data
    
def evaluate_model(model, X_test, y_test):
    \"\"\"
    Оценка модели на тестовых данных
    \"\"\"
    print("Оценка модели")
    y_pred = model.predict(X_test)
    
    # Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }
    
def plot_confusion_matrix(cm, output_filepath):
    \"\"\"
    Визуализация матрицы ошибок
    \"\"\"
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Сохранение графика
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath)
    plt.close()
    
def main():
    \"\"\"
    Основная функция для оценки модели
    \"\"\"
    model_filepath = "models/model.joblib"
    data_filepath = "data/processed/features.csv"
    output_filepath = "reports/figures/confusion_matrix.png"
    target_column = "target"
    
    # Загрузка модели
    model = load_model(model_filepath)
    
    # Загрузка данных
    data = load_data(data_filepath)
    X_test = data.drop(target_column, axis=1)
    y_test = data[target_column]
    
    # Оценка модели
    results = evaluate_model(model, X_test, y_test)
    
    # Визуализация результатов
    plot_confusion_matrix(results["confusion_matrix"], output_filepath)
    print(f"Матрица ошибок сохранена в {output_filepath}")
    
if __name__ == "__main__":
    main()
""",
    
    "src/visualization/visualize.py": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    \"\"\"
    Загрузка данных
    \"\"\"
    print(f"Загрузка данных из {filepath}")
    data = pd.read_csv(filepath)
    return data
    
def plot_feature_importance(model, feature_names, output_filepath):
    \"\"\"
    Визуализация важности признаков
    \"\"\"
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Сохранение графика
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath)
    plt.close()
    
def plot_distributions(data, output_filepath):
    \"\"\"
    Визуализация распределений признаков
    \"\"\"
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(data[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    # Скрытие пустых графиков
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Сохранение графика
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath)
    plt.close()
    
def main():
    \"\"\"
    Основная функция для визуализации
    \"\"\"
    data_filepath = "data/processed/features.csv"
    model_filepath = "models/model.joblib"
    feature_importance_filepath = "reports/figures/feature_importance.png"
    distributions_filepath = "reports/figures/distributions.png"
    
    # Загрузка данных
    data = load_data(data_filepath)
    
    # Загрузка модели
    import joblib
    model = joblib.load(model_filepath)
    
    # Визуализация важности признаков
    feature_names = data.drop("target", axis=1).columns
    plot_feature_importance(model, feature_names, feature_importance_filepath)
    print(f"График важности признаков сохранен в {feature_importance_filepath}")
    
    # Визуализация распределений
    plot_distributions(data, distributions_filepath)
    print(f"Графики распределений сохранены в {distributions_filepath}")
    
if __name__ == "__main__":
    main()
"""
}

# Создание скриптов
for filepath, content in scripts.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Создан файл: {filepath}")

# Создание Jupyter notebook для исследования данных
with open("notebooks/1.0-data-exploration.ipynb", "w") as f:
    f.write("""
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Исследование данных"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn-whitegrid')\n",
    "sns.set(font_scale=1.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Загрузка данных"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Загрузка данных\n",
    "# data = pd.read_csv('../data/raw/sample.csv')\n",
    "# data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Исследовательский анализ данных"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Информация о структуре данных\n",
    "# data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Статистическое описание\n",
    "# data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Проверка пропущенных значений\n",
    "# data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Визуализация данных"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Гистограммы числовых признаков\n",
    "# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns\n",
    "# fig, axes = plt.subplots(3, 3, figsize=(15, 12))\n",
    "# axes = axes.flatten()\n",
    "# \n",
    "# for i, col in enumerate(numeric_cols[:9]):\n",
    "#     sns.histplot(data[col], kde=True, ax=axes[i])\n",
    "#     axes[i].set_title(f'Distribution of {col}')\n",
    "#     \n",
    "# plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Матрица корреляций\n",
    "# plt.figure(figsize=(12, 10))\n",
    "# correlation_matrix = data.corr()\n",
    "# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')\n",
    "# plt.title('Correlation Matrix')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Выводы"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Здесь будут выводы по результатам исследования данных."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
""")

print("\nСтруктура проекта создана успешно!")
print("Теперь вы можете начать работу над проектом, следуя инструкциям в README.md")
```

## Домашнее задание

### Задание 1: Продолжение работы над финальным проектом

1. Выберите тему для финального проекта из предложенных вариантов:
   - Система рекомендаций товаров на основе поведения пользователей
   - Классификация изображений с использованием трансферного обучения
   - Анализ тональности текста с применением современных NLP-моделей
   - Прогнозирование временных рядов для бизнес-показателей
   - Детектирование объектов на изображениях с использованием YOLO

2. Подготовьте данные для проекта:
   - Найдите подходящий набор данных или создайте синтетические данные
   - Выполните предварительную обработку данных
   - Проведите разведочный анализ данных
   - Подготовьте данные для моделирования

3. Разработайте архитектуру решения:
   - Определите основные компоненты системы
   - Выберите подходящие алгоритмы и модели
   - Спланируйте процесс обучения и оценки моделей
   - Определите метрики для оценки качества решения

4. Начните реализацию проекта:
   - Создайте структуру проекта
   - Реализуйте базовые компоненты
   - Подготовьте план дальнейшей работы

5. Подготовьте отчет о проделанной работе и план завершения проекта

## Ресурсы и материалы

### Основные ресурсы
- [Документация TensorFlow по генеративным моделям](https://www.tensorflow.org/tutorials/generative)
- [Документация Keras по VAE](https://keras.io/examples/generative/vae/)
- [Документация TensorFlow по GAN](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Документация MLflow](https://www.mlflow.org/docs/latest/index.html)
- [Документация Docker](https://docs.docker.com/)
- [Документация FastAPI](https://fastapi.tiangolo.com/)

### Дополнительные материалы
- [Книга "Generative Deep Learning" by David Foster](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/)
- [Курс "Reinforcement Learning Specialization" на Coursera](https://www.coursera.org/specializations/reinforcement-learning)
- [Статья "An Introduction to Variational Autoencoders"](https://arxiv.org/abs/1906.02691)
- [Статья "Generative Adversarial Networks: An Overview"](https://arxiv.org/abs/1710.07035)
- [Книга "Building Machine Learning Pipelines" by Hannes Hapke & Catherine Nelson](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/)
- [Статья "MLOps: Continuous delivery and automation pipelines in machine learning"](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Инструменты и платформы
- [MLflow](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Docker](https://www.docker.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Flask](https://flask.palletsprojects.com/)
- [Streamlit](https://streamlit.io/)

