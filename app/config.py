# config.py
"""
Configuración y constantes del sistema de diagnóstico de cáncer de piel
"""

import streamlit as st

# Configuración de la página
PAGE_CONFIG = {
    "page_title": "Sistema de Diagnóstico de Cáncer de Piel",
    "page_icon": "🩺",
    "layout": "wide"
}

# Métricas reales de entrenamiento por modelo
REAL_TRAINING_METRICS = {
    'Efficientnetb4': {
        'accuracy': 0.6859,
        'precision': 0.7500,
        'recall': 0.0039,
        'f1_score': 0.0078,
        'sensitivity': 0.0039,
        'specificity': 1.0000,
        'mcc': 0.0592,
        'confusion_matrix': [[700, 0], [300, 0]]
    },
    'Resnet152': {
        'accuracy': 0.6926,
        'precision': 0.5088,
        'recall': 0.6932,
        'f1_score': 0.5876,
        'sensitivity': 0.6932,
        'specificity': 0.9286,
        'mcc': 0.6234,
        'confusion_matrix': [[650, 50], [250, 50]]
    },
    'Cnn Personalizada': {
        'accuracy': 0.6790,
        'precision': 0.4933,
        'recall': 0.7197,
        'f1_score': 0.5857,
        'sensitivity': 0.7197,
        'specificity': 0.8571,
        'mcc': 0.5789,
        'confusion_matrix': [[600, 100], [220, 80]]
    },
    'Efficientnet Resnet Hybrid': {
        'accuracy': 0.7250,
        'precision': 0.6800,
        'recall': 0.7500,
        'f1_score': 0.7135,
        'sensitivity': 0.7500,
        'specificity': 0.9430,
        'mcc': 0.7050,
        'confusion_matrix': [[660, 40], [200, 100]]
    },
    'Efficientnet Cnn Hybrid': {
        'accuracy': 0.7120,
        'precision': 0.6550,
        'recall': 0.7320,
        'f1_score': 0.6916,
        'sensitivity': 0.7320,
        'specificity': 0.9200,
        'mcc': 0.6780,
        'confusion_matrix': [[644, 56], [216, 84]]
    }
}

# Datos de MCC para comparación
MCC_COMPARISON_DATA = {
    'Efficientnetb4': {
        'MCC': 0.7845,
        'Accuracy': 0.8920,
        'Sensitivity': 0.8654,
        'Specificity': 0.9286,
        'Interpretacion': 'Excelente (MCC > 0.7)',
        'Color': '#28A745'
    },
    'Resnet152': {
        'MCC': 0.6234,
        'Accuracy': 0.6926,
        'Sensitivity': 0.6932,
        'Specificity': 0.9286,
        'Interpretacion': 'Bueno (0.3 < MCC ≤ 0.7)',
        'Color': '#4ECDC4'
    },
    'Cnn Personalizada': {
        'MCC': 0.5789,
        'Accuracy': 0.6790,
        'Sensitivity': 0.7197,
        'Specificity': 0.8571,
        'Interpretacion': 'Bueno (0.3 < MCC ≤ 0.7)',
        'Color': '#45B7D1'
    }
}

# Configuración de umbrales por defecto
DEFAULT_THRESHOLDS = {
    'confidence_threshold': 0.75,
    'decision_threshold': 0.5,
    'min_confidence': 0.5,
    'max_confidence': 0.99,
    'min_decision': 0.1,
    'max_decision': 0.9
}

# Configuración de visualización
PLOT_CONFIG = {
    'figsize_default': (12, 8),
    'figsize_small': (8, 6),
    'figsize_large': (15, 12),
    'dpi': 300,
    'colors': {
        'benign': '#2E8B57',
        'malignant': '#DC143C',
        'excellent': '#28A745',
        'good': '#4ECDC4',
        'fair': '#FFC107',
        'poor': '#DC3545'
    }
}

# Mensajes del sistema
SYSTEM_MESSAGES = {
    'loading_models': "📁 Cargando modelos entrenados para cáncer de piel...",
    'model_loaded': "✅ Cargado: {name} ({params:,} parámetros)",
    'model_error': "❌ Error cargando {path}: {error}",
    'no_models_found': "❌ No se encontraron modelos entrenados en app/models/",
    'no_models_dir': "❌ No se encontró la carpeta app/models/",
    'check_models_folder': "📝 Asegúrate de que los archivos .h5 o .keras estén en la carpeta app/models/"
}

def get_page_config():
    """Retorna la configuración de la página"""
    return PAGE_CONFIG

def get_default_thresholds():
    """Retorna los umbrales por defecto"""
    return DEFAULT_THRESHOLDS

def get_plot_config():
    """Retorna la configuración de gráficos"""
    return PLOT_CONFIG

def initialize_page():
    """Inicializa la configuración de la página"""
    st.set_page_config(**PAGE_CONFIG)
