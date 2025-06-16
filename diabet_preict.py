# Pregnancies: Количество беременностей
# Glucose: Концентрация глюкозы в плазме крови через 2 часа при проведении перорального теста на толерантность к глюкозе
# BloodPressure: Диастолическое артериальное давление (мм рт. ст.)
# SkinThickness: Толщина кожной складки трицепса (мм)
# Insulin: 2-часовой инсулин в сыворотке крови (мю Ед/мл)
# BMI: Индекс массы тела (вес в кг/(рост в м)^2)
# DiabetesPedigreeFunction: Функция родословной диабета
# Age: Возраст (годы)
# Outcome: Классовая переменная (0 или 1)


# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index (weight in kg/(height in m)^2)
# DiabetesPedigreeFunction: Diabetes pedigree function
# Age: Age (years)
# Outcome: Class variable (0 or 1)

# 1. Импортируем нужные библиотеки
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import sklearn.metrics as sk_metrics
import seaborn as sns
import joblib

# Настройки для PyCharm

# Отображать все строки
pd.set_option('display.max_rows', None)

# Отображать все столбцы
pd.set_option('display.max_columns', None)

# Отображать весь текст в ячейке
pd.set_option('display.max_colwidth', None)

# Если хотите, чтобы всё помещалось по ширине окна
pd.set_option('display.width', None)

FILLER = "-" * 50  # Заполнитель для красивого вывода

# 2. Загружаем dataset
df = pd.read_csv('diabetes.csv', encoding='UTF-8')

# 3. Исследуем dataset (EDA)

print(FILLER)
print(df.head())  # Выводим первые 5 значений из datasetа
print(FILLER)
print(df.describe())  # Выводим основные статистики по каждому признаку
print(FILLER)
print(df.isnull().sum())  # Проверяем наличие пропущенных значений
print(FILLER)
print(df.dtypes)  # Проверка типов данных в колонках
print(FILLER)
print(df.tail())  # Выводим последние 5 значений из datasetа
print(FILLER)

print(FILLER)
print("Количество 0-ых значений по колонкам:")
print(FILLER)
for index in df.columns:
    print(f"{index}: {(df[index] == 0).sum()}")
print(FILLER)

print(df['Outcome'].value_counts()) # Количество 0 - здоров, 1 - болен

# 4. Визуализируем данные

# Считаем, сколько раз встречается каждое значение 'Pregnancies'
pregnancy_counts = df['Pregnancies'].value_counts().sort_index()

# Строим график - распределения количества беременностей
fig_pregnancy = go.Figure(
    data=[
        go.Bar(
            x=pregnancy_counts.index,  # Количество беременностей
            y=pregnancy_counts.values,  # Количество женщин забеременевших это кол-во раз
            showlegend=False
        )
    ]
)

# Обновляем заголовки графика
fig_pregnancy.update_layout(
    xaxis_title='Количество беременностей',
    yaxis_title='Количество женщин',
    title='Распределение количества беременностей'
)

# Выводим график - распределения количества беременностей
# fig_pregnancy.show()

print(FILLER)
print(f"Соотношение количества беременностей и числа женщин с таким количеством")
print(FILLER)
for preg_count, fem_count in pregnancy_counts.items():
    print(f"Количество беременностей {preg_count} - {fem_count}")
print(FILLER)

# Подсчитываем количество женщин с более чем одной беременностью по возрастам
age_pregnancies = df[df['Pregnancies'] > 1]['Age'].value_counts()

print(FILLER)
print(f"Количество беременностей по возрастам: ")
print(FILLER)
for age, pregnancies in age_pregnancies.items():
    print(f"Возраст: {age} - {pregnancies}")
print(FILLER)

# Строим график - количество женщин с >1 беременностью по возрастам
fig_age = px.bar(age_pregnancies, x=age_pregnancies.index, y=age_pregnancies.values,
                 title='Количество беременностей по возрастам', barmode='stack')
fig_age.update_layout(
    xaxis_title='Возраст',
    yaxis_title='Количество беременностей'
)

# Выводим график - количество женщин с >1 беременностью по возрастам
fig_age.show()

# Подсчёт уникальных сочетаний возраста и ИМТ (BMI)
bmi_to_age = df[['Age', 'BMI']].value_counts().reset_index(name='Count')

# График - распределение ИМТ по возрастам
fig_bmi_age = px.bar(
    bmi_to_age,
    x='Age',
    y='Count',
    color='BMI',
    title='Соотношение ИМТ к возрасту',
    barmode='stack'
)

# Выводим график соотношения ИМТ к возрасту
fig_bmi_age.show()

# Удаляем строки с Insulin == 0, так как это нереалистичные значения
df_clean = df[df['Insulin'] != 0]

# Диаграмма рассеивания (Инсулин vs Наследственность)
fig_insulin = px.scatter(
    df_clean,
    x='Insulin',  # Ось x - 2-часовой инсулин в сыворотке крови (мю Ед/мл)
    y='DiabetesPedigreeFunction',  # Ось y - функция родословной диабета
    color='Outcome',  # Цветовая сегментация по метке
    title='Отношение инсулина к диабету'
)

# Выводим график отношения инсулина к диабету
fig_insulin.show()

# Диаграмма рассеивания (Инсулин vs Глюкоза)
fig_glucose = px.scatter(
    df_clean,
    x='Insulin',  # Ось x - 2-часовой инсулин в сыворотке крови (мю Ед/мл)
    y='Glucose',
    # Ось y - Концентрация глюкозы в плазме крови через 2 часа при проведении перорального теста на толерантность к глюкозе
    color='Outcome',  # Цветовая сегментация по метке
    title='Отношение инсулина к глюкозе'
)

# Выводим график рассеивания инсулина к глюкозе
fig_glucose.show()

# Подготавливаем dataset для обучения и тестирования модели
# X — все признаки (факторы), кроме целевой переменной
# Y — целевая переменная (наличие или отсутствие диабета)
X = df.drop(['Outcome'], axis=1)
Y = df['Outcome']

# Печатаем форму (размерность) признаков и целевой переменной
print(X.shape)  # (кол-во строк, кол-во признаков)
print(Y.shape)  # (кол-во строк,)

# 5. Разделяем dataset: 80% - обучение, 20% - тестирование
# X_train, y_train - обучающие признаки и метки
# X_test, y_test - тестовые признаки и метки

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 6. Создаем экземпляр модели

# Подбираем наилучшие параметры
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)

# n_estimators=100 — количество деревьев в лесу.
# Большее значение может повысить точность, но замедлит обучение и предсказание.

# max_depth=8 — максимальная глубина каждого дерева.
# Ограничивает переобучение, позволяет деревьям быть достаточно "умными", но не чрезмерно.

# random_state=42 — фиксирует случайность для воспроизводимости результата.
# Влияет на порядок обучения деревьев, разбиение данных.

# class_weight='balanced' — автоматически учитывает дисбаланс классов.
# Увеличивает вес меньшинства (например, больных диабетом), чтобы модель их не "игнорировала".

# max_features='log2' — число признаков, которое пробуется при каждом разбиении — логарифм по основанию 2 от общего количества признаков.
# Уменьшает корреляцию между деревьями и улучшает обобщающую способность.

# min_samples_split=2 — минимальное количество объектов, необходимое для разбиения узла.
# Значение 2 — классика, разрешает разбиение при любой возможности.

# min_samples_leaf=2 — минимальное количество объектов в листе (конечном узле дерева).
# Не дает деревьям создавать слишком "узкие" листья, уменьшает переобучение.

# Создаем модель
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced', max_features='log2', min_samples_split=2, min_samples_leaf=2)

# Обучаем модель
model.fit(X_train, y_train)

# Делаем прогноз
y_pred = model.predict(X_test)

# 7. Определяем точность модели
score = accuracy_score(y_test, y_pred)
print(FILLER)
print(f'Точность обученной модели: {round(score * 100, 2)}%')
print(FILLER)


# 8. Строим матрицу ошибок
# Строки — это фактические значения (истина).
# Столбцы — это предсказанные значения (модель).

def show_confusion_matrix(test_labels, test_classes):
    # Вычисление матрицы ошибок и ее нормализация
    plt.figure(figsize=(10, 10))
    confusion = sk_metrics.confusion_matrix(test_labels, test_classes)
    confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)
    axis_labels = [0, 1]
    ax = sns.heatmap(
        confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap='Blues', annot=True, fmt='.4f', square=True)
    plt.title(f"Матрица ошибок с точностью модели: {round(score * 100, 2)}%")
    plt.ylabel("Истинные метки")
    plt.xlabel("Предсказанные метки")
    print("Отчет о классификации\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))
    plt.show()


show_confusion_matrix(y_test, y_pred)

# 9. Создаем отчет о классификации
class_report = classification_report(y_test, y_pred)

class_report_plot = classification_report(y_test, y_pred, output_dict=True)  # Получаем отчет о классификации в виде словаря

df_report = pd.DataFrame(class_report_plot).transpose()  # Создаем датафрейм для графика
df_report = df_report.drop(columns='support')  # Удаляем вспомогательную колонку

# Визуализируем метрики precision, recall, f1-score по классам
sns.heatmap(df_report, annot=True, cmap='Blues')
plt.title("Отчет о классификации")
plt.xlabel("Метрика")
plt.ylabel("Класс / Среднее")
plt.show()

print(FILLER)
print(f"\nОтчет о классификации:\n {class_report}")
print(FILLER)

# 10. Строим график важности признаков модели

# Получаем важности признаков модели
importances = model.feature_importances_

# Собираем их в DataFrame
feature_names = X.columns  # Имена признаков

# Формируем DataFrame и сортируем признаки по убыванию важности
importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': importances }
).sort_values(by='Важность', ascending=False)

# Отображаем график важности признаков модели
plt.figure(figsize=(10, 6))
sns.barplot(x='Важность', y='Признак', data=importance_df, legend=False)
plt.title('Важность признаков по версии RandomForest')
plt.xlabel('Вклад в предсказание')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()

# 11. Сохраняем модель в файл
joblib.dump(model, 'diabetes_model.pkl')
print("Модель успешно сохранена в файл diabetes_model.pkl")

# 12. Загружаем сохраненную модель
loaded_model = joblib.load('diabetes_model.pkl')

# Используем загруженную модель для предсказания
y_pred_loaded = loaded_model.predict(X_test)

# 13. Сравниваем загруженную модель
print(f"Точность загруженной модели: {accuracy_score(y_test, y_pred_loaded) * 100:.2f}%")