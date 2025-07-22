# data_processor.py
"""
Procesador de datos para comparaciones entre modelos y análisis
"""

import time
import pandas as pd
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from model_utils import predict_image
from config import REAL_TRAINING_METRICS


def compare_all_models(models, processed_image):
    """
    Realiza predicciones con todos los modelos disponibles
    
    Args:
        models (dict): Diccionario con todos los modelos cargados
        processed_image (np.array): Imagen procesada para predicción
    
    Returns:
        list: Lista de diccionarios con resultados de comparación
    """
    comparison_results = []
    model_names = list(models.keys())
    
    for model_name in model_names:
        if model_name in models:
            start_time = time.time()
            model = models[model_name]
            diagnosis, confidence_percent, raw_confidence = predict_image(model, processed_image)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convertir a milisegundos
            
            comparison_results.append({
                'Modelo': model_name,
                'Diagnostico': diagnosis,
                'Confianza (%)': round(confidence_percent, 1),
                'Valor Raw': round(raw_confidence, 3),
                'Tiempo (ms)': round(inference_time, 1)
            })
    
    return comparison_results


def analyze_consistency(comparison_results):
    """
    Analiza la consistencia entre predicciones de diferentes modelos
    
    Args:
        comparison_results (list): Lista de resultados de comparación
    
    Returns:
        tuple: (es_consistente, diagnósticos_únicos, mensaje)
    """
    if not comparison_results:
        return False, [], "No hay resultados para analizar"
    
    diagnoses = [result['Diagnostico'] for result in comparison_results]
    unique_diagnoses = list(set(diagnoses))
    
    is_consistent = len(unique_diagnoses) == 1
    
    if is_consistent:
        message = f"✅ **Consistencia perfecta**: Todos los modelos coinciden en el diagnóstico: {unique_diagnoses[0]}"
    else:
        message = f"⚠️ **Inconsistencia detectada**: Los modelos no coinciden en el diagnóstico"
    
    return is_consistent, unique_diagnoses, message


def get_model_metrics(selected_model):
    """
    Obtiene las métricas del modelo seleccionado (reales o simuladas)
    
    Args:
        selected_model (str): Nombre del modelo seleccionado
    
    Returns:
        tuple: (metrics_data, is_real_data)
    """
    if selected_model in REAL_TRAINING_METRICS:
        metrics_data = REAL_TRAINING_METRICS[selected_model].copy()
        # Convertir matriz de confusión a numpy array
        metrics_data['confusion_matrix'] = np.array(metrics_data['confusion_matrix'])
        return metrics_data, True
    else:
        # Generar datos simulados
        from metrics_calculator import generate_simulated_metrics
        metrics_data = generate_simulated_metrics(selected_model)
        return metrics_data, False


def generate_activation_map(model, image):
    """
    Función obsoleta: usar generate_gradcam() del módulo activation_maps en su lugar
    
    Args:
        model: Modelo de TensorFlow/Keras
        image (np.array): Imagen de entrada
    
    Returns:
        np.array: Imagen con mapa de calor superpuesto o None si hay error
    """
    from activation_maps import generate_gradcam
    return generate_gradcam(model, image)


def create_mcc_comparison_dataframe(mcc_data):
    """
    Crea un DataFrame para la comparación de MCC
    
    Args:
        mcc_data (dict): Datos de MCC por modelo
    
    Returns:
        pd.DataFrame: DataFrame formateado para mostrar
    """
    df_mcc = pd.DataFrame({
        'Modelo': list(mcc_data.keys()),
        'MCC': [data['MCC'] for data in mcc_data.values()],
        'Accuracy': [data['Accuracy'] for data in mcc_data.values()],
        'Sensitivity': [data['Sensitivity'] for data in mcc_data.values()],
        'Specificity': [data['Specificity'] for data in mcc_data.values()],
        'Interpretación': [data['Interpretacion'] for data in mcc_data.values()]
    })
    
    # Aplicar formato a la tabla
    df_display = df_mcc.copy()
    df_display['MCC'] = df_display['MCC'].apply(lambda x: f"{x:.4f}")
    df_display['Accuracy'] = df_display['Accuracy'].apply(lambda x: f"{x:.4f}")
    df_display['Sensitivity'] = df_display['Sensitivity'].apply(lambda x: f"{x:.4f}")
    df_display['Specificity'] = df_display['Specificity'].apply(lambda x: f"{x:.4f}")
    
    return df_display


def format_metrics_for_display(metrics_data):
    """
    Formatea las métricas para mostrar en la interfaz
    
    Args:
        metrics_data (dict): Diccionario con métricas
    
    Returns:
        dict: Métricas formateadas para mostrar
    """
    formatted_metrics = {}
    
    for key, value in metrics_data.items():
        if key in ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', 'mcc']:
            formatted_metrics[key] = {
                'value': value,
                'percentage': f"{value * 100:.1f}%",
                'formatted': f"{value:.3f}"
            }
        else:
            formatted_metrics[key] = value
    
    return formatted_metrics


def prepare_interpretation_text(metrics_data):
    """
    Prepara texto de interpretación de métricas
    
    Args:
        metrics_data (dict): Diccionario con métricas
    
    Returns:
        str: Texto de interpretación formateado
    """
    interpretation = f"""
    - **Accuracy**: {metrics_data['accuracy']*100:.1f}% de las predicciones son correctas
    - **Sensitivity**: {metrics_data['sensitivity']*100:.1f}% de los casos malignos son detectados
    - **Specificity**: {metrics_data['specificity']*100:.1f}% de los casos benignos son correctamente identificados
    - **Precision**: {metrics_data['precision']*100:.1f}% de los casos clasificados como malignos son realmente malignos
    - **F1-Score**: {metrics_data['f1_score']*100:.1f}% es el balance entre precisión y sensibilidad
    - **MCC**: {metrics_data['mcc']:.3f} (Coeficiente de Matthews - balanceado para clases desequilibradas)
    """
    return interpretation


def get_recommendation_based_on_confidence(confidence_percent, threshold_percent, diagnosis):
    """
    Genera recomendación basada en la confianza y diagnóstico
    
    Args:
        confidence_percent (float): Porcentaje de confianza
        threshold_percent (float): Umbral de confianza en porcentaje
        diagnosis (str): Diagnóstico (Benigno/Maligno)
    
    Returns:
        tuple: (tipo_mensaje, mensaje)
    """
    if confidence_percent < threshold_percent:
        return "warning", "⚠️ **Confianza baja**: La confianza en el diagnóstico es menor al umbral establecido. Se recomienda consultar a un especialista."
    else:
        if diagnosis == "Benigno":
            return "success", "✅ **Resultado favorable**: La lesión parece ser benigna según el análisis del modelo entrenado. Sin embargo, se recomienda seguimiento con un dermatólogo para confirmación."
        else:
            return "error", "🚨 **Atención requerida**: El sistema ha detectado características que sugieren una lesión maligna. Se recomienda consultar **urgentemente** con un especialista."
