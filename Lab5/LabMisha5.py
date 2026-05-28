import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА: Регрессия с DecisionTreeRegressor")
print("=" * 70)

print("\n" + "=" * 70)
print("1. СИНТЕТИЧЕСКИЙ ДАТАСЕТ (train.tsv / test.tsv)")
print("=" * 70)

try:
    train_syn = pd.read_csv('train.tsv', sep='\t', header=None)
    test_syn = pd.read_csv('test.tsv', sep='\t', header=None)
    print(f"Train shape: {train_syn.shape}, Test shape: {test_syn.shape}")

    X_syn = train_syn.iloc[:, :-1].values
    y_syn = train_syn.iloc[:, -1].values

    X_syn_train, X_syn_val, y_syn_train, y_syn_val = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)

    scaler_syn = StandardScaler()
    X_syn_train_scaled = scaler_syn.fit_transform(X_syn_train)
    X_syn_val_scaled = scaler_syn.transform(X_syn_val)

    dt_syn = DecisionTreeRegressor(random_state=42, max_depth=15)
    dt_syn.fit(X_syn_train_scaled, y_syn_train)
    y_pred_dt_syn = dt_syn.predict(X_syn_val_scaled)


    def print_metrics(y_true, y_pred, name):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{name}: R²={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}")


    print("\n=== Синтетический датасет ===")
    print_metrics(y_syn_val, y_pred_dt_syn, "DecisionTree")

except FileNotFoundError:
    print("Файлы train.tsv/test.tsv не найдены. Пропускаем синтетический датасет.")

print("\n" + "=" * 70)
print("2. РЕАЛЬНЫЙ ДАТАСЕТ (цены на квартиры в Москве)")
print("=" * 70)

df = pd.read_csv('ml_moscow_flats.csv', encoding='windows-1251')
print(f"\nРазмер датасета: {df.shape}")

target = 'price'
X = df.drop(columns=[target], errors='ignore')
y = df[target]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    df_clean = df.dropna()
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    print(f"Размер после удаления пропусков: {X.shape}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

print("\n" + "=" * 70)
print("ЭТАП 1: ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ")
print("=" * 70)

X_raw = X.copy()
preprocessor_raw = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_raw, y, test_size=0.2, random_state=42)

X_train_raw_processed = preprocessor_raw.fit_transform(X_train_raw)
X_val_raw_processed = preprocessor_raw.transform(X_val_raw)

dt_raw = DecisionTreeRegressor(random_state=42, max_depth=15)
dt_raw.fit(X_train_raw_processed, y_train_raw)
y_pred_dt_raw = dt_raw.predict(X_val_raw_processed)


def print_metrics(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")


print_metrics(y_val_raw, y_pred_dt_raw, "DecisionTree (Сырые данные)")

print("\n" + "=" * 70)
print("ЭТАП 2: ОБУЧЕНИЕ НА МАСШТАБИРОВАННЫХ ДАННЫХ")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline_full = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42, max_depth=15))
])

start_time = time.time()
pipeline_full.fit(X_train, y_train)
y_pred_dt = pipeline_full.predict(X_val)
dt_time = time.time() - start_time

print_metrics(y_val, y_pred_dt, "DecisionTree (Масштабированные данные)")
print(f"Время обучения и предсказания: {dt_time:.3f} сек")

print("\n" + "=" * 70)
print("ЭТАП 3: CROSS-VALIDATION (KFold)")
print("=" * 70)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_dt = cross_val_score(pipeline_full, X, y, cv=cv, scoring='r2')

print(f"DecisionTree CV R² scores: {cv_scores_dt}")
print(f"DecisionTree CV Mean R²: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

print("\n" + "=" * 70)
print("ЭТАП 4: КЛАСТЕРИЗАЦИЯ KMEANS (k=2)")
print("=" * 70)

X_processed = preprocessor.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

silhouette = silhouette_score(X_processed, clusters)
print(f"Коэффициент силуэта: {silhouette:.4f}")

print("\nРаспределение по кластерам:")
for i in range(2):
    print(f"Кластер {i}: {np.sum(clusters == i)} объектов")

print("\nСредняя цена по кластерам:")
for i in range(2):
    print(f"Кластер {i}: {y[clusters == i].mean():.0f} руб.")

feature_names = (numeric_features +
                 list(pipeline_full.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_features)))

y_bin = (y > y.median()).astype(int)
ari = adjusted_rand_score(y_bin, clusters)
ami = adjusted_mutual_info_score(y_bin, clusters)
print(f"\nARI (сравнение с медианой цены): {ari:.4f}")
print(f"AMI (сравнение с медианой цены): {ami:.4f}")

print("\n" + "=" * 70)
print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
print("=" * 70)

importances = pipeline_full.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10

print(f"\nТоп-{top_n} наиболее важных признаков:")
for i in range(top_n):
    print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_dt, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Истинная цена')
plt.ylabel('Предсказанная цена')
plt.title(f'DecisionTreeRegressor\nR² = {r2_score(y_val, y_pred_dt):.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_val - y_pred_dt
sns.histplot(residuals, bins=40, kde=True, color='skyblue')
plt.xlabel('Ошибка предсказания (остатки)')
plt.ylabel('Частота')
plt.title('Распределение остатков DecisionTree')
plt.axvline(x=0, color='red', linestyle='--', label='Нулевая ошибка')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
plt.xlabel('Важность признака')
plt.title('Топ-10 признаков, влияющих на цену квартиры (DecisionTree)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Кластер')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Визуализация кластеров KMeans (k=2)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_bin, cmap='coolwarm', alpha=0.5, s=10)
plt.colorbar(scatter, label='Категория цены (0 - ниже медианы, 1 - выше медианы)')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Истинные метки по медиане цены')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)

results_df = pd.DataFrame({
    'Модель': ['DecisionTree (сырые)', 'DecisionTree (масштабированные)'],
    'R² (val)': [r2_score(y_val_raw, y_pred_dt_raw), r2_score(y_val, y_pred_dt)],
    'RMSE (val)': [np.sqrt(mean_squared_error(y_val_raw, y_pred_dt_raw)),
                   np.sqrt(mean_squared_error(y_val, y_pred_dt))],
    'MAE (val)': [mean_absolute_error(y_val_raw, y_pred_dt_raw),
                  mean_absolute_error(y_val, y_pred_dt)],
    'CV R² (среднее)': ['-', f"{cv_scores_dt.mean():.4f}±{cv_scores_dt.std():.4f}"]
})

print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ (k=2)")
print("=" * 70)

clust_df = pd.DataFrame({
    'Метрика': ['Silhouette', 'ARI', 'AMI'],
    'Значение': [f"{silhouette:.4f}", f"{ari:.4f}", f"{ami:.4f}"]
})
print(clust_df.to_string(index=False))