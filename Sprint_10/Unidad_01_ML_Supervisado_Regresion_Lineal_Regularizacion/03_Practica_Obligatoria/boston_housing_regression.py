# %% [markdown]
# # PRÁCTICA: PREDICCIÓN DE PRECIOS DE CASAS EN BOSTON
# ## Modelado completo con Regresión Lineal y Regularización

# %% [markdown]
# ## IMPORTS

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("ANÁLISIS DE REGRESIÓN - PREDICCIÓN DE PRECIOS DE CASAS EN BOSTON")
print("="*70)

# %% [markdown]
# ## 1. CARGA DEL DATASET

# %%
print("\n1. CARGANDO DATASET...")

try:
    df = pd.read_csv('boston_housing.csv', sep='|')
except:
    print("Creando dataset de ejemplo...")
    np.random.seed(42)
    n = 506
    df = pd.DataFrame({
        'CRIM': np.random.exponential(3, n),
        'ZN': np.random.choice([0, 12.5, 25, 50], n),
        'INDUS': np.random.uniform(0, 28, n),
        'CHAS': np.random.choice([0, 1], n, p=[0.93, 0.07]),
        'NOX': np.random.uniform(0.3, 0.9, n),
        'RM': np.random.normal(6.3, 0.7, n),
        'AGE': np.random.uniform(0, 100, n),
        'DIS': np.random.uniform(1, 12, n),
        'RAD': np.random.choice(range(1, 25), n),
        'TAX': np.random.uniform(180, 720, n),
        'PTRATIO': np.random.uniform(12, 22, n),
        'LSTAT': np.random.uniform(1, 38, n),
        'MEDV': np.random.uniform(5, 50, n)
    })

print(f"✓ Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
display(df.head())
display(df.describe())

# %% [markdown]
# ## 2. ANÁLISIS DE LA VARIABLE TARGET

# %%
target_col = 'MEDV'

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df[target_col], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Precio Mediano (miles $)')
axes[0].axvline(df[target_col].mean(), color='red', linestyle='--')
axes[1].boxplot(df[target_col])
stats.probplot(df[target_col], dist="norm", plot=axes[2])
plt.tight_layout()
plt.savefig('01_analisis_target.png', dpi=300)
plt.show()

print(f"Media: ${df[target_col].mean():.2f}k | Std: ${df[target_col].std():.2f}k")

# %% [markdown]
# ## 3. SPLIT EN TRAIN Y TEST

# %%
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {train_df.shape[0]} | Test: {test_df.shape[0]}")

# %% [markdown]
# ## 4. CONSTRUCCIÓN DE X, y

# %%
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# %% [markdown]
# ## 5. MINI-EDA Y SELECCIÓN DE FEATURES

# %%
correlation_matrix = train_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
plt.tight_layout()
plt.savefig('02_matriz_correlacion.png', dpi=300)
plt.show()

correlations = correlation_matrix[target_col].sort_values(ascending=False)
print("\nCorrelaciones con MEDV:")
print(correlations)

# %%
top_features = correlations[1:].abs().sort_values(ascending=False).head(4).index
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    axes[idx].scatter(train_df[feature], train_df[target_col], alpha=0.5)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('MEDV')
    z = np.polyfit(train_df[feature], train_df[target_col], 1)
    p = np.poly1d(z)
    axes[idx].plot(train_df[feature], p(train_df[feature]), "r--")
    
plt.tight_layout()
plt.savefig('04_top_features.png', dpi=300)
plt.show()

# %% [markdown]
# ## 6. NORMALIZACIÓN

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
print("✓ Variables estandarizadas")

# %% [markdown]
# ## 7. REGRESIÓN LINEAL BASE

# %%
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)
print("✓ Modelo entrenado")

# %% [markdown]
# ## 8. IMPORTANCIA DE FEATURES

# %%
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coeficiente': lr_model.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

print(coef_df)

fig, ax = plt.subplots(figsize=(10, 6))
coef_df.set_index('Feature')['Coeficiente'].plot(kind='barh', ax=ax)
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('05_importancia_features.png', dpi=300)
plt.show()

# %% [markdown]
# ## 9. EVALUACIÓN DEL MODELO

# %%
def evaluate_model(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}: MAE=${mae:.4f}k | RMSE=${rmse:.4f}k | R²={r2:.4f}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

train_metrics = evaluate_model(y_train, y_train_pred, "Train")
test_metrics = evaluate_model(y_test, y_test_pred, "Test")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_train, y_train_pred, alpha=0.5)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
axes[0].set_title(f'Train (R²={train_metrics["R2"]:.3f})')
axes[1].scatter(y_test, y_test_pred, alpha=0.5, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title(f'Test (R²={test_metrics["R2"]:.3f})')
plt.savefig('06_predicciones_vs_reales.png', dpi=300)
plt.show()

# %% [markdown]
# ## 10. REGULARIZACIÓN

# %%
alphas = [0.1, 1.0, 10.0]
results = {'Linear Regression': test_metrics}

print("\n--- RIDGE ---")
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    results[f'Ridge (α={alpha})'] = evaluate_model(y_test, y_pred, f"Ridge α={alpha}")

print("\n--- LASSO ---")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    results[f'Lasso (α={alpha})'] = evaluate_model(y_test, y_pred, f"Lasso α={alpha}")

# %%
results_df = pd.DataFrame(results).T
print("\n" + "="*70)
print("COMPARACIÓN DE MODELOS")
print("="*70)
display(results_df)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, metric in enumerate(['MAE', 'MSE', 'RMSE', 'R2']):
    ax = axes[idx // 2, idx % 2]
    results_df[metric].plot(kind='barh', ax=ax)
    ax.set_title(f'{metric}')
plt.tight_layout()
plt.savefig('08_comparacion_modelos.png', dpi=300)
plt.show()

# %% [markdown]
# ## 11. CONCLUSIONES

# %%
best_model = results_df['R2'].idxmax()
print(f"\n✓ MEJOR MODELO: {best_model}")
print(f"  R²: {results_df.loc[best_model, 'R2']:.4f}")
print(f"  RMSE: ${results_df.loc[best_model, 'RMSE']:.4f}k")
print(f"  Features clave: {', '.join(coef_df.head(3)['Feature'].values)}")
