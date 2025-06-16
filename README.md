# 🧠 Diabetes Prediction with Random Forest


## 📌 Project Description | Описание проекта

**EN:**  
This is a machine learning project to predict the presence of diabetes based on medical data. The model was trained on the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) using RandomForestClassifier.  

**RU:**  
Проект машинного обучения для предсказания наличия диабета на основе медицинских показателей. Модель обучалась на [датасете диабета пима-индейцев](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) с использованием RandomForestClassifier.

---

## 📂 Dataset Features | Признаки в датасете

| Feature | Description (EN)                          | Описание (RU)                                      |
|---------|-------------------------------------------|----------------------------------------------------|
| Pregnancies | Number of times pregnant               | Количество беременностей                           |
| Glucose     | Plasma glucose concentration           | Концентрация глюкозы в крови                       |
| BloodPressure | Diastolic blood pressure             | Диастолическое давление                            |
| SkinThickness | Triceps skin fold thickness          | Толщина кожной складки                             |
| Insulin    | 2-hour serum insulin                    | Уровень инсулина через 2 часа                      |
| BMI        | Body Mass Index                         | Индекс массы тела                                  |
| DiabetesPedigreeFunction | Diabetes pedigree function | Наследственный фактор диабета                      |
| Age        | Age (in years)                          | Возраст (в годах)                                  |
| Outcome    | 1 = Diabetic, 0 = Healthy               | 1 = Диабет, 0 = Здоров                             |

---

## 🧠 Model | Модель

- Algorithm (Алогритм): **Random Forest Classifier**
- Accuracy (Точность): ~**77%**
- GridSearchCV used to optimize (GridSearchCV используется для оптимизации):
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`

---

## 📈 Visualizations | Визуализации

- Histograms of features | Гистограммы признаков
- Bar plots of pregnancies and age distribution | Столбчатые гистограммы беременностей и распределения по возрасту
- Scatter plots: insulin vs. glucose, insulin vs. heredity | Диаграммы рассеивания: зависимость инсулин к глюкозе, зависимость инсулина к наследственности
- Confusion matrix & classification report | Матрица ошибок и отчет о классификации
- Feature importance plot | График важности признаков модели
- A lot of comments (currently in Russian) | Много комментариев (на данный момент на русском)

---

## 🚀 Usage | Использование

```bash
# Install requirements | Установка зависимостей
pip install -r requirements.txt

# Run the model script | Запуск скрипта модели
python diabetes_model.py
```

To load the trained model later | После этого загрузите модель:

```python
import joblib
model = joblib.load('diabets_model.pkl')
prediction = model.predict([your_data])
```

---

## 🧠 Example Prediction | Пример предсказания

```python
sample = [[6,148,72,35,0,33.6,0.627,50]]  # Example input
prediction = model.predict(sample)
print(prediction)  # Output: [1] -> Diabetic
```
---
## 💾 Model Saving & Loading | Сохранение и загрузка модели

```python
# Save | Сохранение
joblib.dump(model, 'diabetes_model.pkl')

# Load | Загрузка
loaded_model = joblib.load('diabetes_model.pkl')
```

---

## 📚 Requirements | Зависимости

- pandas
- matplotlib
- seaborn
- plotly
- scikit-learn
- joblib
---

## 📄 License | Лицензия
EN: This project is for educational and demonstrational purposes.

RU: Этот проект создан для образовательных и демонстрационных целей.

---

## Get Feedback | Обратная связь
EN: If you have any requests or questions, you can contact in private messages in messenger [Telegram](t.me/denchicka213)

RU: Если у Вас есть пожелания или вопросы, Вы можете связаться в личных сообщениях в мессенджере [Telegram](t.me/denchicka213)
