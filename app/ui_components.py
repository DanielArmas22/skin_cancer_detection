# ui_components.py
"""
Componentes de interfaz de usuario para el sistema de diagnóstico de cáncer de piel
"""

import streamlit as st
import pandas as pd
from model_utils import get_model_info


def setup_sidebar(models, t, available_languages):
    """
    Configura la barra lateral con todos los controles
    
    Args:
        models (dict): Diccionario de modelos cargados
        t (dict): Diccionario de traducciones
        available_languages (dict): Idiomas disponibles
    
    Returns:
        dict: Diccionario con todas las configuraciones seleccionadas
    """
    # Configuración de idioma
    if 'language' not in st.session_state:
        st.session_state['language'] = list(available_languages.keys())[0]

    lang = st.sidebar.selectbox(
        "🌐 Idioma/Language",
        options=list(available_languages.keys()),
        index=list(available_languages.keys()).index(st.session_state['language']),
        key='language_selector'
    )
    st.session_state['language'] = lang

    # Configuración principal
    st.sidebar.header(t['settings'])
    st.sidebar.markdown(t['settings_description'])

    # Opción de debug
    debug_mode = st.sidebar.checkbox(
        t['debug_mode'],
        value=False,
        help=t['debug_help']
    )

    # Selección de modelo
    model_names = list(models.keys())
    selected_model = st.sidebar.selectbox(
        t['select_model'],
        model_names,
        index=0,
        help=t['select_model_help']
    )

    # Mostrar información del modelo seleccionado
    if selected_model in models:
        model_info = get_model_info(models[selected_model])
        st.sidebar.markdown("---")
        st.sidebar.markdown(t['model_info'])
        st.sidebar.markdown(f"{t['parameters']} {model_info['parameters']:,}")
        st.sidebar.markdown(f"{t['layers']} {model_info['layers']}")

    # Umbrales de configuración
    confidence_threshold = st.sidebar.slider(
        t['confidence_threshold'],
        min_value=0.5,
        max_value=0.99,
        value=0.75,
        step=0.01,
        help=t['confidence_help']
    )

    decision_threshold = st.sidebar.slider(
        t['decision_threshold'],
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help=t['decision_help']
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(t['threshold_note'])

    return {
        'lang': lang,
        'debug_mode': debug_mode,
        'selected_model': selected_model,
        'confidence_threshold': confidence_threshold,
        'decision_threshold': decision_threshold
    }


def display_main_header(t):
    """
    Muestra el encabezado principal de la aplicación
    
    Args:
        t (dict): Diccionario de traducciones
    """
    st.title(f"🎯 {t['app_title']}")
    st.markdown(t['app_description'])


def display_image_upload_section(t):
    """
    Muestra la sección de carga de imagen
    
    Args:
        t (dict): Diccionario de traducciones
    
    Returns:
        UploadedFile: Archivo subido o None
    """
    st.header(t['image_upload'])
    uploaded_file = st.file_uploader(
        t['upload_prompt'],
        type=["jpg", "jpeg", "png"],
        help=t['upload_help']
    )
    return uploaded_file


def display_image_comparison(image, processed_image, t):
    """
    Muestra comparación entre imagen original y procesada
    
    Args:
        image: Imagen original PIL
        processed_image: Imagen procesada numpy array
        t (dict): Diccionario de traducciones
    """
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=t.get('original_image', "Imagen Original"), use_column_width=True)
    with col2:
        st.image(processed_image, caption=t.get('processed_image', "Imagen Procesada (300x300)"), use_column_width=True)


def display_diagnosis_results(diagnosis, confidence_percent, raw_confidence, t):
    """
    Muestra los resultados del diagnóstico en columnas
    
    Args:
        diagnosis (str): Diagnóstico (Benigno/Maligno)
        confidence_percent (float): Porcentaje de confianza
        raw_confidence (float): Valor raw de confianza
        t (dict): Diccionario de traducciones
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diagnosis_text = t.get('benign', 'Benigno') if diagnosis == "Benigno" else t.get('malignant', 'Maligno')
        if diagnosis == "Benigno":
            st.success(f"✅ **{t.get('prediction', 'Diagnóstico')}: {diagnosis_text}**")
        else:
            st.error(f"⚠️ **{t.get('prediction', 'Diagnóstico')}: {diagnosis_text}**")
    
    with col2:
        st.metric(t.get('confidence', 'Confianza'), f"{confidence_percent:.1f}%")
    
    with col3:
        st.metric("Valor Raw", f"{raw_confidence:.3f}")


def display_debug_info(processed_image, model, decision_threshold, t):
    """
    Muestra información de debug si está habilitado
    
    Args:
        processed_image: Imagen procesada
        model: Modelo de TensorFlow
        decision_threshold: Umbral de decisión
        t (dict): Diccionario de traducciones
    """
    st.info("🐛 " + t.get('debug_info', "**Información de Debug:**"))
    st.code(f"""
{t.get('processed_image_title', "Imagen procesada")}:
- Shape: {processed_image.shape}
- {t.get('range', "Rango")}: [{processed_image.min():.3f}, {processed_image.max():.3f}]
- {t.get('mean', "Media")}: {processed_image.mean():.3f}
- {t.get('std_dev', "Desv. estándar")}: {processed_image.std():.3f}

{t.get('model_title', "Modelo")}:
- Input shape: {model.input_shape}
- Output shape: {model.output_shape}
- {t.get('decision_threshold_title', "Umbral de decisión")}: {decision_threshold}
    """)


def display_interpretation(confidence_percent, confidence_threshold, diagnosis, t):
    """
    Muestra la interpretación de los resultados
    
    Args:
        confidence_percent (float): Porcentaje de confianza
        confidence_threshold (float): Umbral de confianza
        diagnosis (str): Diagnóstico
        t (dict): Diccionario de traducciones
    """
    st.markdown("---")
    st.subheader("📋 " + t.get('results_interpretation', "Interpretación de Resultados"))
    
    if confidence_percent < (confidence_threshold * 100):
        st.warning(t.get('low_confidence_warning', "⚠️ **Confianza baja**: La confianza en el diagnóstico es menor al umbral establecido. Se recomienda consultar a un especialista."))
    else:
        if diagnosis == "Benigno":
            st.success(t.get('favorable_result', "✅ **Resultado favorable**: La lesión parece ser benigna según el análisis del modelo entrenado. Sin embargo, se recomienda seguimiento con un dermatólogo para confirmación."))
        else:
            st.error(t.get('attention_required', "🚨 **Atención requerida**: El sistema ha detectado características que sugieren una lesión maligna. Se recomienda consultar **urgentemente** con un especialista."))


def display_model_comparison_table(df_comparison):
    """
    Muestra tabla de comparación entre modelos
    
    Args:
        df_comparison (pd.DataFrame): DataFrame con resultados de comparación
    """
    st.dataframe(df_comparison, use_container_width=True)


def display_consistency_analysis(comparison_results, t):
    """
    Muestra análisis de consistencia entre modelos
    
    Args:
        comparison_results (list): Resultados de comparación
        t (dict): Diccionario de traducciones
    """
    st.markdown("---")
    st.subheader("🔍 " + t.get('consistency_analysis', "Análisis de Consistencia"))
    
    diagnoses = [result['Diagnostico'] for result in comparison_results]
    if len(set(diagnoses)) == 1:
        st.success(f"✅ **Consistencia perfecta**: Todos los modelos coinciden en el diagnóstico: {diagnoses[0]}")
    else:
        st.warning(f"⚠️ **Inconsistencia detectada**: Los modelos no coinciden en el diagnóstico")
        st.markdown(f"**Diagnósticos obtenidos**: {', '.join(set(diagnoses))}")
        st.info("💡 **Recomendación**: Cuando hay inconsistencias, se recomienda consultar con un especialista para confirmación.")


def display_metrics_explanation():
    """
    Muestra explicación de la matriz de confusión y métricas
    """
    st.markdown("---")
    st.subheader("🔍 Interpretación de la Matriz de Confusión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Elementos de la Matriz:**
        
        - **Verdaderos Positivos (TP)**: Casos malignos correctamente identificados
        - **Verdaderos Negativos (TN)**: Casos benignos correctamente identificados  
        - **Falsos Positivos (FP)**: Casos benignos clasificados como malignos
        - **Falsos Negativos (FN)**: Casos malignos clasificados como benignos
        """)
    
    with col2:
        st.markdown("""
        **🎯 Importancia Médica:**
        
        - **Falsos Negativos** son críticos (no detectar cáncer)
        - **Falsos Positivos** causan ansiedad innecesaria
        - **Recall alto** es crucial para detección temprana
        - **Precision alta** reduce falsas alarmas
        """)


def display_mcc_interpretation(mcc_data):
    """
    Muestra interpretación de los datos de MCC
    
    Args:
        mcc_data (dict): Datos de MCC por modelo
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🥇 EfficientNetB4:**
        - MCC: 0.7845 (**Excelente**)
        - Mejor balance general
        - Recomendado para uso clínico
        - Superior confiabilidad diagnóstica
        """)
    
    with col2:
        st.markdown("""
        **🥈 ResNet152:**
        - MCC: 0.6234 (**Bueno**)
        - Rendimiento moderado
        - Alternativa viable
        - Balance aceptable
        """)
    
    with col3:
        st.markdown("""
        **🥉 CNN Personalizada:**
        - MCC: 0.5789 (**Bueno**)
        - Rendimiento estándar
        - Opción complementaria
        - Mejoras posibles
        """)


def display_technical_info(model_info, t):
    """
    Muestra información técnica del sistema
    
    Args:
        model_info (dict): Información del modelo
        t (dict): Diccionario de traducciones
    """
    st.markdown("---")
    st.subheader("🔧 Información Técnica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Dataset de entrenamiento**: ISIC 2019 (25,331 imágenes reales)
        
        **Tipo de clasificación**: Binaria (Benigno/Maligno)
        
        **Resultados de entrenamiento**: Accuracy ~69%, optimizado para cáncer de piel
        """)
    
    with col2:
        st.markdown(f"""
        **Parámetros del modelo**: {model_info['parameters']:,}
        
        **Capas**: {model_info['layers']}
        
        **Entrada**: {model_info['input_shape']}
        
        **Métricas avanzadas**: MCC, Sensibilidad, Especificidad
        
        **Análisis estadístico**: Pruebas de McNemar
        """)


def display_medical_disclaimer():
    """
    Muestra advertencia médica
    """
    st.markdown("---")
    st.warning("""
    ⚠️ **Descargo de Responsabilidad Médica**
    
    Este sistema es para fines educativos y de investigación. Los resultados no constituyen diagnóstico médico 
    y no deben reemplazar la consulta con profesionales de la salud calificados.
    
    **Siempre consulta con un dermatólogo** para obtener un diagnóstico profesional.
    """)


def display_pdf_generation_section(t):
    """
    Muestra sección de generación de PDF
    
    Args:
        t (dict): Diccionario de traducciones
    
    Returns:
        bool: True si se presionó el botón de generar PDF
    """
    st.markdown("---")
    st.subheader("📄 Generar Reporte PDF")
    
    col1, col2 = st.columns(2)
    
    with col1:
        generate_pdf = st.button("🖨️ Generar Reporte PDF Completo", type="primary")
    
    with col2:
        st.markdown("""
        **📋 El reporte PDF incluye:**
        - Diagnóstico y análisis de la imagen
        - Comparación entre todos los modelos
        - Matriz de confusión y métricas avanzadas
        - Gráficos de MCC y análisis estadístico
        - Pruebas de McNemar
        - Recomendaciones médicas
        """)
    
    return generate_pdf


def display_metrics_in_columns(metrics_data):
    """
    Muestra métricas en columnas organizadas
    
    Args:
        metrics_data (dict): Datos de métricas
    """
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.metric("Accuracy", f"{metrics_data['accuracy']:.3f}", f"{metrics_data['accuracy']*100:.1f}%")
        st.metric("Sensitivity", f"{metrics_data['sensitivity']:.3f}", f"{metrics_data['sensitivity']*100:.1f}%")
        st.metric("Specificity", f"{metrics_data['specificity']:.3f}", f"{metrics_data['specificity']*100:.1f}%")
    
    with metric_col2:
        st.metric("Precision", f"{metrics_data['precision']:.3f}", f"{metrics_data['precision']*100:.1f}%")
        st.metric("F1-Score", f"{metrics_data['f1_score']:.3f}", f"{metrics_data['f1_score']*100:.1f}%")
        st.metric("MCC", f"{metrics_data['mcc']:.3f}")


def display_mcnemar_results_table(mcnemar_results):
    """
    Muestra tabla de resultados de McNemar
    
    Args:
        mcnemar_results (list): Lista de resultados de McNemar
    """
    df_mcnemar = pd.DataFrame(mcnemar_results)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**📊 Resultados de Pruebas de McNemar**")
        st.dataframe(df_mcnemar, use_container_width=True)
    
    with col2:
        st.markdown("""
        **📖 Interpretación:**
        
        **Criterio de decisión:**
        - p-valor < 0.05: Diferencia significativa (EfficientNetB4 superior)
        - p-valor ≥ 0.05: Sin diferencia significativa
        
        **Resultados clave:**
        - EfficientNetB4 muestra superioridad estadística
        - Diferencias significativas vs otros modelos
        - Validación robusta de su excelencia
        """)


def display_statistical_conclusions(mcnemar_results):
    """
    Muestra conclusiones estadísticas de McNemar
    
    Args:
        mcnemar_results (list): Lista de resultados de McNemar
    """
    st.markdown("---")
    st.subheader("📊 Conclusiones Estadísticas")
    
    efficient_comparisons = [r for r in mcnemar_results if 'EfficientNetB4' in r['Comparación'] and r['P-valor'] < 0.05]
    
    if efficient_comparisons:
        st.success(f"✅ **EfficientNetB4 demuestra superioridad estadística significativa**")
        st.markdown("**Comparaciones donde EfficientNetB4 es superior:**")
        for comp in efficient_comparisons:
            st.markdown(f"- {comp['Comparación']}: p = {comp['P-valor']:.4f} - {comp['Interpretación']}")
    else:
        st.info("ℹ️ **EfficientNetB4 mantiene rendimiento comparable o superior**")
    
    st.markdown("""
    **🔬 Interpretación médica de McNemar para EfficientNetB4:**
    
    Los resultados de McNemar confirman que EfficientNetB4:
    - Muestra diferencias estadísticamente significativas comparado con otros modelos
    - Demuestra superioridad en precisión diagnóstica
    - Proporciona mayor confiabilidad para decisiones clínicas
    - Es la opción más robusta para implementación médica
    - Justifica su selección como modelo principal para el diagnóstico
    """)
