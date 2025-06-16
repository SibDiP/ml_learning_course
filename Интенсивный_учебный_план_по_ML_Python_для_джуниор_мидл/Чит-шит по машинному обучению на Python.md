# Чит-шит по машинному обучению на Python

## Основы машинного обучения

- **Машинное обучение (ML)**: Обучение компьютеров на основе данных без явного программирования.
- **Типы ML**:
  - **С учителем (Supervised)**: Обучение на размеченных данных (вход -> выход).
    - **Задачи**: Классификация, Регрессия.
  - **Без учителя (Unsupervised)**: Обучение на неразмеченных данных (поиск структуры).
    - **Задачи**: Кластеризация, Снижение размерности, Обнаружение аномалий.
  - **С подкреплением (Reinforcement)**: Обучение агента через взаимодействие со средой (награды/штрафы).
- **Переобучение (Overfitting)**: Модель слишком хорошо подогнана под обучающие данные, плохо обобщает.
- **Недообучение (Underfitting)**: Модель слишком проста, не улавливает закономерности.
- **Борьба с переобучением**: Регуляризация (L1, L2), Dropout, Ранняя остановка, Кросс-валидация, Увеличение данных.
- **Кросс-валидация (Cross-Validation)**: Метод оценки обобщающей способности (например, k-fold).
- **Метрики качества**:
  - **Классификация**: Accuracy, Precision, Recall, F1-score, ROC-AUC.
  - **Регрессия**: MSE, RMSE, MAE, R².

## Обработка данных (Pandas & NumPy)

```python
import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv("data.csv")

# Информация о данных
df.info()
df.describe()

# Обработка пропусков
df.isnull().sum() # Проверка пропусков
df.dropna() # Удаление строк с пропусками
df.fillna(df.mean()) # Заполнение средним

# Кодирование категорий
df_encoded = pd.get_dummies(df, columns=["category_col"])

# Выбор признаков и цели
X = df.drop("target", axis=1)
y = df["target"]

# Масштабирование признаков
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Scikit-learn: Основные шаги

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Пример классификатора
from sklearn.metrics import accuracy_score, classification_report

# 1. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Масштабирование (если нужно)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Выбор и обучение модели
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Предсказания
y_pred = model.predict(X_test_scaled)

# 5. Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

## Классические алгоритмы (Scikit-learn)

- **Линейная регрессия**: `from sklearn.linear_model import LinearRegression`
- **Логистическая регрессия**: `from sklearn.linear_model import LogisticRegression`
- **k-ближайших соседей (k-NN)**: `from sklearn.neighbors import KNeighborsClassifier`
- **Дерево решений**: `from sklearn.tree import DecisionTreeClassifier`
- **Случайный лес**: `from sklearn.ensemble import RandomForestClassifier`
- **Метод опорных векторов (SVM)**: `from sklearn.svm import SVC`
- **Наивный Байес**: `from sklearn.naive_bayes import GaussianNB`
- **Градиентный бустинг**: `from sklearn.ensemble import GradientBoostingClassifier`
- **K-Means (кластеризация)**: `from sklearn.cluster import KMeans`
- **PCA (снижение размерности)**: `from sklearn.decomposition import PCA`

## Глубокое обучение (Keras/TensorFlow)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Создание Sequential модели
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(input_dim,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax") # Для многоклассовой классификации
])

# Компиляция модели
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy", # Для целочисленных меток
    metrics=["accuracy"]
)

# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)

# Предсказания
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Сохранение/Загрузка модели
model.save("model.h5")
loaded_model = keras.models.load_model("model.h5")
```

## Компьютерное зрение (CNN)

```python
# Пример CNN слоя
layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")

# Пример Pooling слоя
layers.MaxPooling2D(pool_size=(2, 2))

# Пример простой CNN архитектуры
model_cnn = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# Трансферное обучение
from tensorflow.keras.applications import VGG16
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Заморозка весов

model_tl = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
```

## Обработка естественного языка (NLP)

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Предобработка текста
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Векторизация текста
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(corpus)

# Embedding слой в Keras
layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)

# Пример RNN (LSTM)
model_rnn = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(1, activation="sigmoid") # Для бинарной классификации
])
```

## MLOps

- **MLflow**: Отслеживание экспериментов, логирование параметров, метрик, артефактов.
  ```python
  import mlflow
  with mlflow.start_run():
      mlflow.log_param("param_name", value)
      mlflow.log_metric("metric_name", value)
      mlflow.sklearn.log_model(model, "model_name")
  ```
- **Docker**: Контейнеризация модели и зависимостей для развертывания.
  - `Dockerfile`: Инструкции по сборке образа.
  - `docker build -t image_name .`
  - `docker run -p host_port:container_port image_name`
- **API (Flask/FastAPI)**: Создание интерфейса для взаимодействия с моделью.
  ```python
  # Flask
  from flask import Flask, request, jsonify
  app = Flask(__name__)
  @app.route("/predict", methods=["POST"])
  def predict(): ...

  # FastAPI
  from fastapi import FastAPI
  app = FastAPI()
  @app.post("/predict")
  async def predict(data: InputSchema): ...
  ```
- **Мониторинг**: Отслеживание производительности модели, дрейфа данных.

## Визуализация (Matplotlib & Seaborn)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Гистограмма
sns.histplot(data=df, x="feature", kde=True)
plt.title("Distribution")
plt.show()

# Диаграмма рассеяния
sns.scatterplot(data=df, x="feature1", y="feature2", hue="target")
plt.title("Scatter Plot")
plt.show()

# Матрица ошибок
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Кривая обучения
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

## Сохранение и загрузка моделей

```python
# Scikit-learn
import joblib
joblib.dump(model, "model.joblib")
loaded_model = joblib.load("model.joblib")

# Keras/TensorFlow
model.save("model.h5")
from tensorflow import keras
loaded_model = keras.models.load_model("model.h5")
```

