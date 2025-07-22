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
    
    # Usamos una key específica para el file_uploader para preservar el estado
    # entre cambios de idioma
    uploaded_file = st.file_uploader(
        t['upload_prompt'],
        type=["jpg", "jpeg", "png"],
        help=t['upload_help'],
        key="skin_image_uploader"
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
        st.metric(t.get('raw_value', "Valor Raw"), f"{raw_confidence:.3f}")


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
        st.success(f"{t.get('perfect_consistency', '✅ **Consistencia perfecta**: Todos los modelos coinciden en el diagnóstico:')} {diagnoses[0]}")
    else:
        st.warning(t.get('inconsistency_detected', '⚠️ **Inconsistencia detectada**: Los modelos no coinciden en el diagnóstico'))
        st.markdown(f"{t.get('diagnoses_obtained', '**Diagnósticos obtenidos**:')} {', '.join(set(diagnoses))}")
        st.info(f"{t.get('recommendation_title', '💡 **Recomendación**:')} {t.get('inconsistency_recommendation', 'Cuando hay inconsistencias, se recomienda consultar con un especialista para confirmación.')}")


def display_metrics_explanation(t):
    """
    Muestra explicación de la matriz de confusión y métricas
    
    Args:
        t (dict): Diccionario de traducciones
    """
    st.markdown("---")
    st.subheader("🔍 " + t.get('confusion_matrix_interpretation', 'Interpretación de la Matriz de Confusión'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        {t.get('matrix_elements', '**📊 Elementos de la Matriz:**')}
        
        - {t.get('true_positives', '**Verdaderos Positivos (TP)**: Casos malignos correctamente identificados')}
        - {t.get('true_negatives', '**Verdaderos Negativos (TN)**: Casos benignos correctamente identificados')}
        - {t.get('false_positives', '**Falsos Positivos (FP)**: Casos benignos clasificados como malignos')}
        - {t.get('false_negatives', '**Falsos Negativos (FN)**: Casos malignos clasificados como benignos')}
        """)
    
    with col2:
        st.markdown(f"""
        {t.get('medical_importance', '**🎯 Importancia Médica:**')}
        
        - {t.get('fn_critical', '**Falsos Negativos** son críticos (no detectar cáncer)')}
        - {t.get('fp_anxiety', '**Falsos Positivos** causan ansiedad innecesaria')}
        - {t.get('recall_importance', '**Recall alto** es crucial para detección temprana')}
        - {t.get('precision_importance', '**Precision alta** reduce falsas alarmas')}
        """)


def display_mcc_interpretation(mcc_data, t):
    """
    Muestra interpretación de los datos de MCC
    
    Args:
        mcc_data (dict): Datos de MCC por modelo
        t (dict): Diccionario de traducciones
    """
    col1, col2, col3 = st.columns(3)
    
    # Obtener los modelos y sus datos
    models = list(mcc_data.keys())
    
    if len(models) >= 3:
        with col1:
            model = models[0]
            data = mcc_data[model]
            st.markdown(f"""
            **🥇 {model}:**
            - MCC: {data['MCC']:.4f} (**{data['Interpretacion']}**)
            - {t.get('best_balance', 'Mejor balance general')}
            - {t.get('recommended_clinical', 'Recomendado para uso clínico')}
            - {t.get('superior_reliability', 'Superior confiabilidad diagnóstica')}
            """)
        
        with col2:
            model = models[1]
            data = mcc_data[model]
            st.markdown(f"""
            **🥈 {model}:**
            - MCC: {data['MCC']:.4f} (**{data['Interpretacion']}**)
            - {t.get('moderate_performance', 'Rendimiento moderado')}
            - {t.get('viable_alternative', 'Alternativa viable')}
            - {t.get('acceptable_balance', 'Balance aceptable')}
            """)
        
        with col3:
            model = models[2]
            data = mcc_data[model]
            st.markdown(f"""
            **🥉 {model}:**
            - MCC: {data['MCC']:.4f} (**{data['Interpretacion']}**)
            - {t.get('standard_performance', 'Rendimiento estándar')}
            - {t.get('complementary_option', 'Opción complementaria')}
            - {t.get('possible_improvements', 'Mejoras posibles')}
            """)
    else:
        # En caso de que haya menos de 3 modelos, mostrar los que hay
        for i, model in enumerate(models):
            data = mcc_data[model]
            with col1 if i == 0 else col2 if i == 1 else col3:
                st.markdown(f"""
                **{model}:**
                - MCC: {data['MCC']:.4f} (**{data['Interpretacion']}**)
                - {t.get('accuracy', 'Precisión')}: {data['Accuracy']:.4f}
                - {t.get('sensitivity', 'Sensibilidad')}: {data['Sensitivity']:.4f}
                - {t.get('specificity', 'Especificidad')}: {data['Specificity']:.4f}
                """)


def display_technical_info(model_info, t, config=None):
    """
    Muestra información técnica del sistema
    
    Args:
        model_info (dict): Información del modelo
        t (dict): Diccionario de traducciones
        config (dict, optional): Configuración del sistema
    """
    st.markdown("---")
    
    # Título usando traducción
    technical_title = t.get('technical_info', "Información Técnica")
    if technical_title == technical_title.upper():
        technical_title = technical_title.title()
    st.subheader(f"🔧 {technical_title}")
    
    # Obtener traducciones para la información técnica
    dataset_info = t.get('technical_dataset', "Dataset: ISIC 2019 (25,331 imágenes reales)")
    type_info = t.get('technical_type', "Tipo: Clasificación Binaria (Benigno/Maligno)")
    accuracy_info = t.get('technical_accuracy', "Precisión: ~69% (optimizado para cáncer de piel)")
    
    # Extraer solo las etiquetas sin el contenido
    dataset_label = dataset_info.split(":")[0] + ":"
    type_label = type_info.split(":")[0] + ":"
    accuracy_label = accuracy_info.split(":")[0] + ":"
    
    # Usar configuración si está disponible, de lo contrario valores predeterminados
    if config:
        dataset_name = config.get("DATASET_NAME", "ISIC 2019")
        dataset_size = config.get("DATASET_SIZE", "25,331")
        accuracy_value = config.get("MODEL_ACCURACY", "~69%")
        optimization_target = config.get("OPTIMIZATION_TARGET", "cáncer de piel")
    else:
        dataset_name = "ISIC 2019"
        dataset_size = "25,331"
        accuracy_value = "~69%"
        optimization_target = "cáncer de piel"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **{dataset_label}** {dataset_name} ({dataset_size} imágenes reales)
        
        **{type_label}** Binaria (Benigno/Maligno)
        
        **{accuracy_label}** {accuracy_value}, optimizado para {optimization_target}
        """)
    
    with col2:
        parameters_label = t.get('parameters', "Parámetros")
        layers_label = t.get('layers', "Capas")
        input_label = t.get('technical_input', "Entrada").split(":")[0]
        
        st.markdown(f"""
        **{parameters_label}:** {model_info['parameters']:,}
        
        **{layers_label}:** {model_info['layers']}
        
        **{input_label}:** {model_info['input_shape']}
        
        **Métricas avanzadas:** MCC, {t.get('sensitivity', 'Sensibilidad')}, {t.get('specificity', 'Especificidad')}
        
        **Análisis estadístico:** Pruebas de McNemar
        """)


def display_medical_disclaimer(t=None):
    """
    Muestra advertencia médica
    
    Args:
        t (dict, optional): Diccionario de traducciones
    """
    st.markdown("---")
    
    # Si no hay traducciones, usar texto por defecto en inglés
    if not t:
        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This system is for educational and research purposes only. Results DO NOT constitute medical diagnosis 
        and should not replace consultation with qualified healthcare professionals.
        
        **Always consult with a dermatologist** for professional diagnosis.
        """)
    else:
        disclaimer_title = t.get('medical_disclaimer_title', "Descargo de Responsabilidad Médica").replace("DESCARGO", "Descargo")
        disclaimer_1 = t.get('medical_disclaimer_1', "Este sistema es para fines educativos y de investigación.")
        disclaimer_2 = t.get('medical_disclaimer_2', "Los resultados NO constituyen diagnóstico médico.")
        disclaimer_3 = t.get('medical_disclaimer_3', "SIEMPRE consulte con un dermatólogo para diagnóstico profesional.")
        
        st.warning(f"""
        ⚠️ **{disclaimer_title}**
        
        {disclaimer_1} {disclaimer_2}
        
        **{disclaimer_3.capitalize()}**
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
    
    # Título para la sección PDF (creamos clave específica si no existe)
    pdf_section_title = t.get('pdf_section_title', "Generar Reporte PDF")
    st.subheader(f"📄 {pdf_section_title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Texto para el botón de PDF
        pdf_button_text = t.get('generate_pdf_button', "Generar Reporte PDF Completo")
        generate_pdf = st.button(f"🖨️ {pdf_button_text}", type="primary")
    
    with col2:
        # Contenido del PDF, traducido si existe o texto por defecto
        pdf_includes = t.get('pdf_includes', "El reporte PDF incluye")
        pdf_content_diagnosis = t.get('pdf_content_diagnosis', "Diagnóstico y análisis de la imagen")
        pdf_content_comparison = t.get('pdf_content_comparison', "Comparación entre todos los modelos")
        pdf_content_matrix = t.get('pdf_content_matrix', "Matriz de confusión y métricas avanzadas")
        pdf_content_charts = t.get('pdf_content_charts', "Gráficos de MCC y análisis estadístico")
        pdf_content_mcnemar = t.get('pdf_content_mcnemar', "Pruebas de McNemar")
        pdf_content_recommendations = t.get('pdf_content_recommendations', "Recomendaciones médicas")
        
        st.markdown(f"""
        **📋 {pdf_includes}:**
        - {pdf_content_diagnosis}
        - {pdf_content_comparison}
        - {pdf_content_matrix}
        - {pdf_content_charts}
        - {pdf_content_mcnemar}
        - {pdf_content_recommendations}
        """)
    
    return generate_pdf


def display_metrics_in_columns(metrics_data, t=None):
    """
    Muestra métricas en columnas organizadas
    
    Args:
        metrics_data (dict): Datos de métricas
        t (dict, optional): Diccionario de traducciones
    """
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.metric(t.get('accuracy', 'Accuracy') if t else "Accuracy", f"{metrics_data['accuracy']:.3f}", f"{metrics_data['accuracy']*100:.1f}%")
        st.metric(t.get('sensitivity', 'Sensitivity') if t else "Sensitivity", f"{metrics_data['sensitivity']:.3f}", f"{metrics_data['sensitivity']*100:.1f}%")
        st.metric(t.get('specificity', 'Specificity') if t else "Specificity", f"{metrics_data['specificity']:.3f}", f"{metrics_data['specificity']*100:.1f}%")
    
    with metric_col2:
        st.metric(t.get('precision', 'Precision') if t else "Precision", f"{metrics_data['precision']:.3f}", f"{metrics_data['precision']*100:.1f}%")
        st.metric(t.get('f1_score', 'F1-Score') if t else "F1-Score", f"{metrics_data['f1_score']:.3f}", f"{metrics_data['f1_score']*100:.1f}%")
        st.metric(t.get('mcc', 'MCC') if t else "MCC", f"{metrics_data['mcc']:.3f}")


def display_mcnemar_results_table(mcnemar_results, t=None):
    """
    Muestra tabla de resultados de McNemar
    
    Args:
        mcnemar_results (list): Lista de resultados de McNemar
        t (dict, optional): Diccionario de traducciones
    """
    df_mcnemar = pd.DataFrame(mcnemar_results)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**📊 " + (t.get('mcnemar_test_results', "Resultados de Pruebas de McNemar") if t else "Resultados de Pruebas de McNemar") + "**")
        st.dataframe(df_mcnemar, use_container_width=True)
    
    with col2:
        if t:
            st.markdown(f"""
            **📖 {t.get('interpretation', 'Interpretación')}:**
            
            **{t.get('decision_criteria', 'Criterio de decisión')}:**
            - p-valor < 0.05: {t.get('significant_difference', 'Diferencia significativa')}
            - p-valor ≥ 0.05: {t.get('no_significant_difference', 'Sin diferencia significativa')}
            
            **{t.get('key_results', 'Resultados clave')}:**
            - {t.get('statistical_superiority_shown', 'Muestra superioridad estadística')}
            - {t.get('significant_diff_vs_models', 'Diferencias significativas vs otros modelos')}
            - {t.get('robust_validation', 'Validación robusta de su excelencia')}
            """)
        else:
            st.markdown("""
            **📖 Interpretation:**
            
            **Decision criteria:**
            - p-value < 0.05: Significant difference
            - p-value ≥ 0.05: No significant difference
            
            **Key results:**
            - Shows statistical superiority
            - Significant differences vs other models
            - Robust validation of its excellence
            """)


def display_statistical_conclusions(mcnemar_results, t=None):
    """
    Muestra conclusiones estadísticas de McNemar
    
    Args:
        mcnemar_results (list): Lista de resultados de McNemar
        t (dict, optional): Diccionario de traducciones
    """
    st.markdown("---")
    st.subheader("📊 " + (t.get('statistical_conclusions', "Conclusiones Estadísticas") if t else "Conclusiones Estadísticas"))
    
    # Encontrar el modelo con más resultados significativos
    model_stats = {}
    
    # Contabilizar resultados significativos por modelo
    for result in mcnemar_results:
        comp = result['Comparación']
        p_value = result['P-valor']
        
        # Si el p-valor es significativo
        if p_value < 0.05:
            # Extraer nombres de modelos de la comparación (formato: "Modelo1 vs Modelo2")
            models = comp.split(' vs ')
            
            # Verificar cuál modelo aparece en la interpretación como "mejor"
            interpretation = result['Interpretación']
            for model in models:
                if f"{model} significativamente mejor" in interpretation:
                    if model not in model_stats:
                        model_stats[model] = 0
                    model_stats[model] += 1
    
    # Encontrar el modelo con más victorias estadísticamente significativas
    best_model = None
    max_wins = 0
    for model, wins in model_stats.items():
        if wins > max_wins:
            max_wins = wins
            best_model = model
    
    # Si se encontró un modelo con victorias significativas
    if best_model:
        significant_results = [r for r in mcnemar_results if best_model in r['Comparación'] and r['P-valor'] < 0.05 and best_model in r['Interpretación']]
        
        st.success(f"✅ **{best_model} " + (t.get('statistical_superiority', "demuestra superioridad estadística significativa") if t else "demuestra superioridad estadística significativa") + "**")
        
        superior_comparisons = t.get('superior_comparisons', "**Comparaciones donde {model} es superior:**") if t else "**Comparaciones donde {model} es superior:**"
        st.markdown(superior_comparisons.format(model=best_model))
        
        for comp in significant_results:
            st.markdown(f"- {comp['Comparación']}: p = {comp['P-valor']:.4f} - {comp['Interpretación']}")
    else:
        st.info("ℹ️ **" + (t.get('no_statistical_diff', "No hay diferencias estadísticamente significativas entre los modelos") if t else "No hay diferencias estadísticamente significativas entre los modelos") + "**")
    
    # Conclusiones generales
    if best_model:
        medical_interpretation = t.get('medical_interpretation', 'Interpretación médica') if t else 'Interpretación médica'
        for_model = t.get('for_model', 'para') if t else 'para'
        mcnemar_confirm = t.get('mcnemar_confirm', 'Los resultados de McNemar confirman que') if t else 'Los resultados de McNemar confirman que'
        stat_diff = t.get('stat_diff', 'Muestra diferencias estadísticamente significativas comparado con otros modelos') if t else 'Muestra diferencias estadísticamente significativas comparado con otros modelos'
        diagnostic_superiority = t.get('diagnostic_superiority', 'Demuestra superioridad en precisión diagnóstica') if t else 'Demuestra superioridad en precisión diagnóstica'
        clinical_reliability = t.get('clinical_reliability', 'Proporciona mayor confiabilidad para decisiones clínicas') if t else 'Proporciona mayor confiabilidad para decisiones clínicas'
        robust_option = t.get('robust_option', 'Es la opción más robusta para implementación médica') if t else 'Es la opción más robusta para implementación médica'
        justified_selection = t.get('justified_selection', 'Justifica su selección como modelo principal para el diagnóstico') if t else 'Justifica su selección como modelo principal para el diagnóstico'
        
        st.markdown(f"""
        **🔬 {medical_interpretation} {for_model} {best_model}:**
        
        {mcnemar_confirm} {best_model}:
        - {stat_diff}
        - {diagnostic_superiority}
        - {clinical_reliability}
        - {robust_option}
        - {justified_selection}
        """)
