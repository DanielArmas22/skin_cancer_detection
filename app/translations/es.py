translations = {
    # Títulos y encabezados
    'app_title': 'Sistema Inteligente de Diagnóstico de Cáncer de Piel',
    'app_description': 'Este sistema utiliza **modelos entrenados específicamente para cáncer de piel** con el dataset ISIC 2019 para analizar imágenes dermatológicas y proporcionar un diagnóstico preliminar de lesiones cutáneas.',
    'settings': '⚙️ Configuración',
    'settings_description': 'Selecciona los parámetros para el análisis',
    'image_upload': '📸 Carga de Imagen',
    
    # Opciones de configuración
    'debug_mode': '🐛 Modo Debug',
    'debug_help': 'Activa información detallada de debug para diagnosticar problemas',
    'select_model': '🤖 Selecciona el modelo a utilizar',
    'select_model_help': 'Cada modelo tiene diferentes características de rendimiento y precisión',
    'model_info': '📊 **Información del Modelo:**',
    'parameters': '**Parámetros:**',
    'layers': '**Capas:**',
    'confidence_threshold': '🎯 Umbral de confianza para diagnóstico',
    'confidence_help': 'Valores más altos requieren mayor confianza para el diagnóstico',
    'decision_threshold': '⚖️ Umbral de decisión Maligno/Benigno',
    'decision_help': 'Valores más bajos hacen el modelo más sensible a casos malignos',
    'threshold_note': '💡 **Nota**: Un umbral de decisión más bajo (ej: 0.3) hará que el modelo sea más sensible a detectar casos malignos, pero también aumentará los falsos positivos.',
    
    # Upload de imágenes
    'upload_prompt': 'Sube una imagen de la lesión cutánea (JPG, JPEG, PNG)',
    'upload_help': 'La imagen debe ser clara y mostrar bien la lesión',
    
    # Resultados y análisis
    'processing_image': 'Procesando imagen...',
    'benign': 'Benigno',
    'malignant': 'Maligno',
    'confidence': 'Confianza',
    'prediction': 'Diagnóstico',
    'advanced_analysis': 'Análisis Avanzado',
    'metrics_title': 'Métricas de Rendimiento',
    'confusion_matrix': 'Matriz de Confusión',
    'statistical_summary': 'Resumen Estadístico Completo',
    'low_confidence_warning': '⚠️ **Confianza baja**: La confianza en el diagnóstico es menor al umbral establecido. Se recomienda consultar a un especialista.',
    'favorable_result': '✅ **Resultado favorable**: La lesión parece ser benigna según el análisis del modelo entrenado. Sin embargo, se recomienda seguimiento con un dermatólogo para confirmación.',
    'attention_required': '🚨 **Atención requerida**: El sistema ha detectado características que sugieren una lesión maligna. Se recomienda consultar **urgentemente** con un especialista.',
    
    # Elementos de métricas
    'accuracy': 'Accuracy',
    'sensitivity': 'Sensibilidad',
    'specificity': 'Specificidad',
    'precision': 'Precision',
    'f1_score': 'F1-Score',
    'mcc': 'MCC',
    'tp': 'Verdaderos Positivos',
    'tn': 'Verdaderos Negativos',
    'fp': 'Falsos Positivos',
    'fn': 'Falsos Negativos',
    
    # Idiomas
    'language': 'Idioma',
    
    # Errores y mensajes
    'models_load_error': 'No se pudieron cargar los modelos entrenados.',
    'models_folder_check': 'Asegúrate de que los archivos .h5 estén en la carpeta app/models/',
    'model_load_exception': 'Error al cargar los modelos',
    'no_models_available': 'No hay modelos disponibles. Verifica que los modelos entrenados estén en app/models/',
    
    # Procesamiento de imágenes
    'original_image': 'Imagen Original',
    'processed_image': 'Imagen Procesada (300x300)',
    'diagnosis_results': 'Resultados del Diagnóstico',
    
    # Secciones adicionales
    'results_interpretation': 'Interpretación de Resultados',
    'model_comparison': 'Comparación de Todos los Modelos',
    'consistency_analysis': 'Análisis de Consistencia',
    'pdf_success': 'Reporte PDF generado exitosamente',
    
    # Advertencias y mensajes
    'low_confidence_warning': '⚠️ **Confianza baja**: La confianza en el diagnóstico es menor al umbral establecido. Se recomienda consultar a un especialista.',
    'favorable_result': '✅ **Resultado favorable**: La lesión parece ser benigna según el análisis. Se recomienda seguimiento médico.',
    'attention_required': '🚨 **Atención requerida**: Se detectaron características que sugieren una lesión maligna. Consulte urgentemente a un especialista.',
    
    # Debug
    'debug_info': '**Información de Debug:**',
    'processed_image_title': 'Imagen procesada',
    'range': 'Rango',
    'mean': 'Media',
    'std_dev': 'Desv. estándar',
    'model_title': 'Modelo',
    'decision_threshold_title': 'Umbral de decisión',
    
    # PDF y reportes
    'download_pdf': 'Descargar Reporte PDF',
    'pdf_success': 'Reporte PDF generado exitosamente',
}
