import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configurar matplotlib para mostrar texto en español
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8')

class PredictorMermas:
    def __init__(self):
        self.modelos = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.mejor_modelo = None
        self.metricas_df = None

    def conectar_bd(self):
        """Conexión a la base de datos"""
        return pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='mermas'
        )

    def calcular_metricas(self, y_true, y_pred, nombre_modelo):
        """Calcular métricas de evaluación con manejo de errores"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            # MAPE con manejo de divisiones por cero
            with np.errstate(divide='ignore', invalid='ignore'):
                ape = np.abs((y_true - y_pred) / np.where(np.abs(y_true) > 1e-8, y_true, 1e-8))
                mape = np.mean(ape[np.isfinite(ape)]) * 100

            # Porcentaje de error medio
            error_porcentual = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 0.1)) * 100

            return {
                'Modelo': nombre_modelo,
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE (%)': mape,
                'Error % Medio': error_porcentual
            }
        except Exception as e:
            print(f"Error calculando métricas para {nombre_modelo}: {e}")
            return None

    def cargar_datos(self):
        """Cargar y procesar datos desde la base de datos"""
        connection = self.conectar_bd()

        # Query optimizada para manejar diferentes tipos de unidades
        query = """
        SELECT 
            fecha,
            linea,
            categoria,
            seccion,
            motivo,
            negocio,
            comuna,
            region,
            descripcion,
            merma_unidad,
            merma_monto,
            mes,
            año,
            semestre
        FROM mermasdb 
        WHERE merma_unidad IS NOT NULL 
        AND merma_monto IS NOT NULL
        AND fecha IS NOT NULL
        AND linea IS NOT NULL
        AND categoria IS NOT NULL
        ORDER BY fecha
        """

        data = pd.read_sql(query, connection)
        connection.close()

        print(f"Datos cargados: {len(data)} registros")
        return data

    def preprocesar_datos(self, data):
        """Preprocesamiento avanzado de datos"""
        print("🔧 Preprocesando datos...")

        # Manejar valores de merma_unidad (pueden ser negativos, decimales, etc.)
        data['merma_unidad_abs'] = np.abs(data['merma_unidad'])
        data['merma_monto_abs'] = np.abs(data['merma_monto'])

        # Filtrar valores extremos (outliers)
        q1_unidad = data['merma_unidad_abs'].quantile(0.25)
        q3_unidad = data['merma_unidad_abs'].quantile(0.75)
        iqr_unidad = q3_unidad - q1_unidad

        # Mantener valores dentro de 1.5 * IQR
        data = data[
            (data['merma_unidad_abs'] >= q1_unidad - 1.5 * iqr_unidad) &
            (data['merma_unidad_abs'] <= q3_unidad + 1.5 * iqr_unidad)
            ]

        # Convertir fecha
        data['fecha'] = pd.to_datetime(data['fecha'])

        # Características temporales
        data['año'] = data['fecha'].dt.year
        data['mes'] = data['fecha'].dt.month
        data['dia_semana'] = data['fecha'].dt.dayofweek
        data['dia_mes'] = data['fecha'].dt.day
        data['trimestre'] = data['fecha'].dt.quarter
        data['semana_año'] = data['fecha'].dt.isocalendar().week

        # Características de estacionalidad
        data['es_fin_semana'] = (data['dia_semana'] >= 5).astype(int)
        data['es_inicio_mes'] = (data['dia_mes'] <= 5).astype(int)
        data['es_fin_mes'] = (data['dia_mes'] >= 25).astype(int)

        # Codificar variables categóricas
        categorical_columns = ['linea', 'categoria', 'seccion', 'motivo', 'negocio', 'comuna', 'region']

        for col in categorical_columns:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            self.encoders[col] = le

        # Agregar por línea y categoría para reducir dimensionalidad
        data_agg = data.groupby(['fecha', 'linea', 'categoria']).agg({
            'merma_unidad_abs': 'sum',
            'merma_monto_abs': 'sum',
            'año': 'first',
            'mes': 'first',
            'dia_semana': 'first',
            'dia_mes': 'first',
            'trimestre': 'first',
            'semana_año': 'first',
            'es_fin_semana': 'first',
            'es_inicio_mes': 'first',
            'es_fin_mes': 'first',
            'linea_encoded': 'first',
            'categoria_encoded': 'first',
            'seccion_encoded': 'first',
            'motivo_encoded': 'first',
            'negocio_encoded': 'first'
        }).reset_index()

        # Normalizar variable objetivo con transformación robusta
        # Usar log1p para valores positivos y manejar la escala
        data_agg['merma_unidad_normalizada'] = np.log1p(data_agg['merma_unidad_abs'])

        # Crear características adicionales
        data_agg['merma_por_monto'] = data_agg['merma_unidad_abs'] / np.maximum(data_agg['merma_monto_abs'], 1)

        print(f"Datos después del preprocesamiento: {len(data_agg)} registros")
        return data_agg

    def entrenar_modelos(self, X_train, X_test, y_train, y_test):
        """Entrenar múltiples modelos de ML"""
        print("Entrenando modelos de Machine Learning...")
        resultados = []

        # 1. Random Forest (optimizado)
        print("- Entrenando Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        self.modelos['Random Forest'] = rf
        metricas_rf = self.calcular_metricas(y_test, y_pred_rf, 'Random Forest')
        if metricas_rf:
            resultados.append(metricas_rf)

        # 2. Gradient Boosting (optimizado)
        print("- Entrenando Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        self.modelos['Gradient Boosting'] = gb
        metricas_gb = self.calcular_metricas(y_test, y_pred_gb, 'Gradient Boosting')
        if metricas_gb:
            resultados.append(metricas_gb)

        # 3. SVR (Support Vector Regression) - Nuevo modelo
        print("- Entrenando Support Vector Regression...")
        scaler_svr = RobustScaler()
        X_train_scaled = scaler_svr.fit_transform(X_train)
        X_test_scaled = scaler_svr.transform(X_test)

        svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        y_pred_svr = svr.predict(X_test_scaled)
        self.modelos['SVR'] = svr
        self.scalers['SVR'] = scaler_svr
        metricas_svr = self.calcular_metricas(y_test, y_pred_svr, 'SVR')
        if metricas_svr:
            resultados.append(metricas_svr)

        # 4. Regresión Lineal con regularización
        print("- Entrenando Regresión Lineal...")
        scaler_lr = StandardScaler()
        X_train_scaled_lr = scaler_lr.fit_transform(X_train)
        X_test_scaled_lr = scaler_lr.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_scaled_lr, y_train)
        y_pred_lr = lr.predict(X_test_scaled_lr)
        self.modelos['Regresión Lineal'] = lr
        self.scalers['Regresión Lineal'] = scaler_lr
        metricas_lr = self.calcular_metricas(y_test, y_pred_lr, 'Regresión Lineal')
        if metricas_lr:
            resultados.append(metricas_lr)

        # 5. CatBoost
        print("- Entrenando CatBoost...")
        catboost = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False
        )
        catboost.fit(X_train, y_train)
        y_pred_catboost = catboost.predict(X_test)
        self.modelos['CatBoost'] = catboost
        metricas_catboost = self.calcular_metricas(y_test, y_pred_catboost, 'CatBoost')
        if metricas_catboost:
            resultados.append(metricas_catboost)

        # 6. XGBoost
        print("- Entrenando XGBoost...")
        xgboost = XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        xgboost.fit(X_train, y_train)
        y_pred_xgboost = xgboost.predict(X_test)
        self.modelos['XGBoost'] = xgboost
        metricas_xgboost = self.calcular_metricas(y_test, y_pred_xgboost, 'XGBoost')
        if metricas_xgboost:
            resultados.append(metricas_xgboost)

        return resultados

    def visualizar_resultados(self, X_test, y_test, resultados):
        """Visualizar resultados con métricas claras"""
        if not resultados:
            print("No hay resultados para visualizar")
            return None, None

        # Crear DataFrame de métricas
        self.metricas_df = pd.DataFrame(resultados)

        # Mostrar tabla de métricas
        print("\n""")
        print("MÉTRICAS DE EVALUACIÓN DE MODELOS")
        print("")
        print(self.metricas_df.round(4).to_string(index=False))

        # Encontrar el mejor modelo (mayor R²)
        if len(self.metricas_df) > 0:
            mejor_idx = self.metricas_df['R²'].idxmax()
            self.mejor_modelo = self.metricas_df.loc[mejor_idx, 'Modelo']
            mejor_r2 = self.metricas_df.loc[mejor_idx, 'R²']
            mejor_error = self.metricas_df.loc[mejor_idx, 'Error % Medio']

            print(f"\nMEJOR MODELO: {self.mejor_modelo}")
            print(f"   R² = {mejor_r2:.4f}")
            print(f"   Error % Medio = {mejor_error:.2f}%")
            print(f"   Precisión = {100 - mejor_error:.2f}%")

        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Predicción de Mermas en Supermercados', fontsize=16, fontweight='bold')

        # Gráfico 1: Comparación R² y Error %
        ax1 = axes[0, 0]
        x_pos = np.arange(len(self.metricas_df))

        # Barras para R²
        bars1 = ax1.bar(x_pos - 0.2, self.metricas_df['R²'], 0.4,
                        label='R² Score', color='skyblue', alpha=0.8)

        # Eje secundario para Error %
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x_pos + 0.2, self.metricas_df['Error % Medio'], 0.4,
                             label='Error % Medio', color='lightcoral', alpha=0.8)

        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('R² Score', color='blue')
        ax1_twin.set_ylabel('Error % Medio', color='red')
        ax1.set_title('Comparación de Modelos: R² vs Error %', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.metricas_df['Modelo'], rotation=45, ha='right')

        # Añadir valores en las barras
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax1.text(bar1.get_x() + bar1.get_width() / 2., height1 + 0.01,
                     f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
            ax1_twin.text(bar2.get_x() + bar2.get_width() / 2., height2 + 1,
                          f'{height2:.1f}%', ha='center', va='bottom', fontsize=9)

        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # Gráfico 2: Predicciones vs Reales (mejor modelo)
        ax2 = axes[0, 1]
        if self.mejor_modelo:
            # Obtener predicciones del mejor modelo
            if self.mejor_modelo in self.scalers:
                X_test_scaled = self.scalers[self.mejor_modelo].transform(X_test)
                y_pred_mejor = self.modelos[self.mejor_modelo].predict(X_test_scaled)
            else:
                y_pred_mejor = self.modelos[self.mejor_modelo].predict(X_test)

            # Scatter plot
            ax2.scatter(y_test, y_pred_mejor, alpha=0.6, color='blue', s=30)

            # Línea de predicción perfecta
            min_val = min(y_test.min(), y_pred_mejor.min())
            max_val = max(y_test.max(), y_pred_mejor.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')

            # Calcular y mostrar R²
            r2_mejor = r2_score(y_test, y_pred_mejor)
            ax2.text(0.05, 0.95, f'R² = {r2_mejor:.4f}', transform=ax2.transAxes,
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax2.set_xlabel('Valores Reales (log)')
            ax2.set_ylabel('Predicciones (log)')
            ax2.set_title(f'Predicciones vs Reales - {self.mejor_modelo}', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Gráfico 3: Importancia de características (mejor modelo disponible)
        ax3 = axes[1, 0]
        modelo_importancia = None

        # Buscar un modelo que tenga feature_importances_
        if self.mejor_modelo in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'CatBoost']:
            modelo_importancia = self.modelos[self.mejor_modelo]
        elif 'Random Forest' in self.modelos:
            modelo_importancia = self.modelos['Random Forest']
        elif 'XGBoost' in self.modelos:
            modelo_importancia = self.modelos['XGBoost']
        elif 'CatBoost' in self.modelos:
            modelo_importancia = self.modelos['CatBoost']

        if modelo_importancia and hasattr(modelo_importancia, 'feature_importances_'):
            importances = modelo_importancia.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10

            ax3.bar(range(len(indices)), importances[indices], color='green', alpha=0.7)
            ax3.set_xlabel('Características')
            ax3.set_ylabel('Importancia')

            # Determinar qué modelo se está usando para el título
            modelo_usado = self.mejor_modelo if self.mejor_modelo in ['Random Forest', 'Gradient Boosting', 'XGBoost',
                                                                      'CatBoost'] else 'Random Forest'
            ax3.set_title(f'Top 10 - Importancia de Características ({modelo_usado})', fontweight='bold')

            ax3.set_xticks(range(len(indices)))
            ax3.set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Importancia de características\nno disponible para este modelo',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Importancia de Características', fontweight='bold')

        # Gráfico 4: Distribución de errores con estadísticas
        ax4 = axes[1, 1]
        if self.mejor_modelo:
            errores = y_test - y_pred_mejor

            # Histograma
            n, bins, patches = ax4.hist(errores, bins=30, alpha=0.7, color='green',
                                        edgecolor='black', density=True)

            # Estadísticas
            error_mean = errores.mean()
            error_std = errores.std()

            # Líneas de estadísticas
            ax4.axvline(error_mean, color='red', linestyle='--', linewidth=2,
                        label=f'Error Medio: {error_mean:.3f}')
            ax4.axvline(error_mean + error_std, color='orange', linestyle=':', linewidth=2,
                        label=f'+1 Std: {error_mean + error_std:.3f}')
            ax4.axvline(error_mean - error_std, color='orange', linestyle=':', linewidth=2,
                        label=f'-1 Std: {error_mean - error_std:.3f}')

            ax4.set_xlabel('Error de Predicción (log)')
            ax4.set_ylabel('Densidad')
            ax4.set_title('Distribución de Errores de Predicción', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return self.mejor_modelo, self.metricas_df

    def generar_reporte_final(self):
        """Generar reporte final con recomendaciones"""
        print("\n" + "=" * 80)
        print("REPORTE FINAL - PREDICCIÓN DE MERMAS")
        print("=" * 80)

        if self.metricas_df is not None and len(self.metricas_df) > 0:
            mejor_modelo_data = self.metricas_df[self.metricas_df['Modelo'] == self.mejor_modelo].iloc[0]

            print(f"\nRESUMEN EJECUTIVO:")
            print(f"   • Mejor Modelo: {self.mejor_modelo}")
            print(f"   • Precisión: {100 - mejor_modelo_data['Error % Medio']:.1f}%")
            print(f"   • R² Score: {mejor_modelo_data['R²']:.4f}")
            print(f"   • Error Medio: {mejor_modelo_data['Error % Medio']:.2f}%")

            print(f"\nRECOMENDACIONES:")
            if mejor_modelo_data['R²'] > 0.7:
                print("   Modelo con buen desempeño predictivo")
                print("   Recomendado para implementación en producción")
            elif mejor_modelo_data['R²'] > 0.5:
                print("   Modelo con desempeño moderado")
                print("   Requiere más datos o ajuste de parámetros")
            else:
                print("   Modelo con bajo desempeño")
                print("   Requiere revisión de características y datos")

            print(f"\nANÁLISIS DE TODOS LOS MODELOS:")
            for _, row in self.metricas_df.iterrows():
                print(f"   {row['Modelo']:20} | R²: {row['R²']:6.4f} | Error: {row['Error % Medio']:6.2f}%")

            print(f"\nCOMPARACIÓN DE FAMILIAS DE MODELOS:")

            # Agrupar por tipo de modelo
            tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'CatBoost']
            other_models = ['SVR', 'Regresión Lineal']

            tree_results = self.metricas_df[self.metricas_df['Modelo'].isin(tree_models)]
            other_results = self.metricas_df[self.metricas_df['Modelo'].isin(other_models)]

            if len(tree_results) > 0:
                print("   Modelos basados en árboles:")
                best_tree = tree_results.loc[tree_results['R²'].idxmax()]
                print(f"     Mejor: {best_tree['Modelo']} (R²: {best_tree['R²']:.4f})")

            if len(other_results) > 0:
                print("   Otros modelos:")
                best_other = other_results.loc[other_results['R²'].idxmax()]
                print(f"     Mejor: {best_other['Modelo']} (R²: {best_other['R²']:.4f})")

            # Análisis específico de los nuevos modelos
            if 'CatBoost' in self.metricas_df['Modelo'].values:
                catboost_metrics = self.metricas_df[self.metricas_df['Modelo'] == 'CatBoost'].iloc[0]
                print(f"\n   CatBoost Performance:")
                print(f"     R²: {catboost_metrics['R²']:.4f} | Error: {catboost_metrics['Error % Medio']:.2f}%")

            if 'XGBoost' in self.metricas_df['Modelo'].values:
                xgboost_metrics = self.metricas_df[self.metricas_df['Modelo'] == 'XGBoost'].iloc[0]
                print(f"   XGBoost Performance:")
                print(f"     R²: {xgboost_metrics['R²']:.4f} | Error: {xgboost_metrics['Error % Medio']:.2f}%")

    def ejecutar_analisis_completo(self):
        """Ejecutar análisis completo de principio a fin"""
        print("")
        print("SISTEMA DE PREDICCIÓN DE MERMAS EN SUPERMERCADOS")
        print("")

        try:
            # 1. Cargar datos
            data = self.cargar_datos()

            # 2. Preprocesar
            data_processed = self.preprocesar_datos(data)

            # 3. Preparar características
            self.feature_names = [
                'año', 'mes', 'dia_semana', 'dia_mes', 'trimestre', 'semana_año',
                'es_fin_semana', 'es_inicio_mes', 'es_fin_mes',
                'linea_encoded', 'categoria_encoded', 'seccion_encoded',
                'motivo_encoded', 'negocio_encoded', 'merma_por_monto'
            ]

            X = data_processed[self.feature_names]
            y = data_processed['merma_unidad_normalizada']

            print(f"Características utilizadas: {len(self.feature_names)}")
            print(f"Registros para entrenamiento: {len(X)}")

            # 4. Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )

            # 5. Entrenar modelos
            resultados = self.entrenar_modelos(X_train, X_test, y_train, y_test)

            # 6. Visualizar resultados
            self.visualizar_resultados(X_test, y_test, resultados)

            # 7. Generar reporte final
            self.generar_reporte_final()

            print("\nAnálisis completado exitosamente!")

        except Exception as e:
            print(f"Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()


# Ejecutar análisis
if __name__ == "__main__":
    predictor = PredictorMermas()
    predictor.ejecutar_analisis_completo()