import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Evaluador de modelos de cáncer de piel usando dataset real ISIC
    """
    
    def __init__(self, test_data_path="data/ISIC_dataset_test", cache_dir="app/cache"):
        """
        Inicializa el evaluador
        
        Args:
            test_data_path: Ruta al dataset de test
            cache_dir: Directorio para guardar métricas
        """
        self.test_data_path = Path(test_data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.metrics_file = self.cache_dir / "model_metrics.json"
        self.comparison_file = self.cache_dir / "model_comparisons.json"
        
        # Verificar que existe el dataset de test
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Dataset de test no encontrado en: {self.test_data_path}")
        
        # Cargar datos de test
        self.test_generator = self._load_test_data()
        
    def _load_test_data(self):
        """Carga los datos de test usando ImageDataGenerator"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            str(self.test_data_path),
            target_size=(300, 300),
            batch_size=32,
            class_mode='binary',
            shuffle=False,  # Importante para mantener orden
            seed=42
        )
        
        return test_generator
    
    def _load_models(self):
        """Carga todos los modelos disponibles"""
        models = {}
        models_dir = Path("models")
        
        if models_dir.exists():
            for model_path in models_dir.glob("*.h5"):
                try:
                    model_name = model_path.stem
                    model = tf.keras.models.load_model(str(model_path))
                    models[model_name] = model
                    print(f"✅ Modelo cargado: {model_name}")
                except Exception as e:
                    print(f"❌ Error cargando {model_path}: {e}")
        
        return models
    
    def evaluate_single_model(self, model, model_name):
        """
        Evalúa un modelo individual
        
        Args:
            model: Modelo de TensorFlow/Keras
            model_name: Nombre del modelo
            
        Returns:
            dict: Métricas del modelo
        """
        print(f"📊 Evaluando modelo: {model_name}")
        
        # Resetear el generador para asegurar orden consistente
        self.test_generator.reset()
        
        # Obtener predicciones
        predictions = model.predict(self.test_generator, verbose=0)
        
        # Obtener etiquetas verdaderas
        true_labels = self.test_generator.classes
        
        # Convertir predicciones a clases binarias
        pred_classes = (predictions > 0.5).astype(int).flatten()
        pred_probs = predictions.flatten()
        
        # Calcular métricas
        accuracy = accuracy_score(true_labels, pred_classes)
        precision = precision_score(true_labels, pred_classes, zero_division=0)
        recall = recall_score(true_labels, pred_classes, zero_division=0)
        f1 = f1_score(true_labels, pred_classes, zero_division=0)
        
        # Métricas adicionales
        cm = confusion_matrix(true_labels, pred_classes)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(true_labels, pred_classes)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(true_labels, pred_probs)
        except:
            auc_roc = 0.5
        
        # Distribución de clases
        unique, counts = np.unique(true_labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        # Compilar métricas
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'mcc': float(mcc),
            'auc_roc': float(auc_roc),
            'confusion_matrix': cm.tolist(),
            'test_samples': len(true_labels),
            'class_distribution': [int(class_distribution.get(0, 0)), int(class_distribution.get(1, 0))],
            'predictions': pred_classes.tolist(),
            'prediction_probabilities': pred_probs.tolist(),
            'true_labels': true_labels.tolist(),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ {model_name} - Accuracy: {accuracy:.3f}, MCC: {mcc:.3f}")
        
        return metrics
    
    def evaluate_all_models(self, force_refresh=False):
        """
        Evalúa todos los modelos disponibles
        
        Args:
            force_refresh: Si True, recalcula todas las métricas
            
        Returns:
            dict: Métricas de todos los modelos
        """
        # Verificar si ya existen métricas y no se fuerza el refresh
        if not force_refresh and self.metrics_file.exists():
            print("📋 Cargando métricas existentes...")
            return self.load_metrics()
        
        print("🚀 Iniciando evaluación completa de modelos...")
        
        # Cargar modelos
        models = self._load_models()
        
        if not models:
            print("❌ No se encontraron modelos para evaluar")
            return {}
        
        all_metrics = {}
        
        # Evaluar cada modelo
        for model_name, model in models.items():
            try:
                metrics = self.evaluate_single_model(model, model_name)
                all_metrics[model_name] = metrics
            except Exception as e:
                print(f"❌ Error evaluando {model_name}: {e}")
                continue
        
        # Guardar métricas
        self.save_metrics(all_metrics)
        
        # Realizar comparaciones estadísticas
        self._perform_model_comparisons(all_metrics)
        
        print(f"✅ Evaluación completada: {len(all_metrics)} modelos")
        
        return all_metrics
    
    def _perform_model_comparisons(self, all_metrics):
        """
        Realiza comparaciones estadísticas entre modelos
        
        Args:
            all_metrics: Diccionario con métricas de todos los modelos
        """
        print("📊 Realizando comparaciones estadísticas...")
        
        model_names = list(all_metrics.keys())
        comparisons = {}
        
        # Comparar cada par de modelos
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Evitar duplicados
                    continue
                    
                comparison_key = f"{model1} vs {model2}"
                
                try:
                    # Obtener predicciones
                    pred1 = np.array(all_metrics[model1]['predictions'])
                    pred2 = np.array(all_metrics[model2]['predictions'])
                    true_labels = np.array(all_metrics[model1]['true_labels'])
                    
                    # Crear tabla de contingencia para McNemar
                    # Casos donde ambos aciertan o fallan
                    both_correct = (pred1 == true_labels) & (pred2 == true_labels)
                    both_wrong = (pred1 != true_labels) & (pred2 != true_labels)
                    
                    # Casos donde uno acierta y otro falla
                    model1_correct_model2_wrong = (pred1 == true_labels) & (pred2 != true_labels)
                    model1_wrong_model2_correct = (pred1 != true_labels) & (pred2 == true_labels)
                    
                    # Tabla de contingencia 2x2
                    n_01 = np.sum(model1_correct_model2_wrong)
                    n_10 = np.sum(model1_wrong_model2_correct)
                    
                    # Test de McNemar
                    if n_01 + n_10 > 0:
                        try:
                            # Usar corrección de continuidad
                            chi2 = (abs(n_01 - n_10) - 1)**2 / (n_01 + n_10)
                            p_value = 1 - stats.chi2.cdf(chi2, 1)
                            
                            # Determinar si es significativo
                            significant = p_value < 0.05
                            
                            # Determinar cuál modelo es mejor
                            model1_better = n_01 > n_10
                            
                            comparisons[comparison_key] = {
                                'model1': model1,
                                'model2': model2,
                                'model1_correct_model2_wrong': int(n_01),
                                'model1_wrong_model2_correct': int(n_10),
                                'chi2_statistic': float(chi2),
                                'p_value': float(p_value),
                                'significant': significant,
                                'model1_better': model1_better,
                                'comparison_timestamp': datetime.now().isoformat()
                            }
                            
                        except Exception as e:
                            print(f"❌ Error en test McNemar para {comparison_key}: {e}")
                            comparisons[comparison_key] = None
                    else:
                        comparisons[comparison_key] = None
                        
                except Exception as e:
                    print(f"❌ Error comparando {comparison_key}: {e}")
                    comparisons[comparison_key] = None
        
        # Guardar comparaciones
        self.save_comparisons(comparisons)
        
        print(f"✅ Comparaciones completadas: {len(comparisons)} pares")
    
    def save_metrics(self, metrics):
        """Guarda las métricas en archivo JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Métricas guardadas en: {self.metrics_file}")
    
    def load_metrics(self):
        """Carga las métricas desde archivo JSON"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_comparisons(self, comparisons):
        """Guarda las comparaciones en archivo JSON"""
        with open(self.comparison_file, 'w') as f:
            json.dump(comparisons, f, indent=2)
        print(f"💾 Comparaciones guardadas en: {self.comparison_file}")
    
    def load_comparisons(self):
        """Carga las comparaciones desde archivo JSON"""
        if self.comparison_file.exists():
            with open(self.comparison_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_best_model(self, metrics=None):
        """
        Obtiene el mejor modelo basado en MCC
        
        Args:
            metrics: Métricas de los modelos (opcional)
            
        Returns:
            tuple: (nombre_modelo, métricas)
        """
        if metrics is None:
            metrics = self.load_metrics()
        
        if not metrics:
            return None, None
        
        best_model = max(metrics.items(), key=lambda x: x[1]['mcc'])
        return best_model[0], best_model[1]
    
    def generate_summary_report(self):
        """
        Genera un reporte resumen de la evaluación
        
        Returns:
            dict: Reporte resumen
        """
        metrics = self.load_metrics()
        comparisons = self.load_comparisons()
        
        if not metrics:
            return {}
        
        # Mejor modelo
        best_model_name, best_metrics = self.get_best_model(metrics)
        
        # Estadísticas generales
        summary = {
            'total_models': len(metrics),
            'test_samples': metrics[list(metrics.keys())[0]]['test_samples'],
            'class_distribution': metrics[list(metrics.keys())[0]]['class_distribution'],
            'best_model': {
                'name': best_model_name,
                'mcc': best_metrics['mcc'],
                'accuracy': best_metrics['accuracy'],
                'f1_score': best_metrics['f1_score']
            },
            'model_rankings': sorted(
                [(name, data['mcc']) for name, data in metrics.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            'evaluation_timestamp': datetime.now().isoformat(),
            'significant_comparisons': len([
                comp for comp in comparisons.values() 
                if comp and comp.get('significant', False)
            ])
        }
        
    def load_comparison_results(self):
        """Carga las comparaciones desde archivo JSON (alias para compatibilidad)"""
        return self.load_comparisons()