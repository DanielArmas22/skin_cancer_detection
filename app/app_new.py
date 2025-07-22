# app.py
"""
Sistema de Diagnóstico de Cáncer de Piel usando técnicas de IA
Aplicación principal refactorizada y modularizada
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Importaciones de módulos propios
from config import initialize_page, MCC_COMPARISON_DATA, REAL_TRAINING_METRICS, get_default_thresholds
from model_utils import load_models, predict_image, predict_image_with_debug, predict_image_with_custom_threshold, get_model_info
from preprocessing import preprocess_image
from translations import get_available_languages, load_translations
from activation_maps import generate_gradcam
from ui_components import (
    setup_sidebar, display_main_header, display_image_upload_section,
    display_image_comparison, display_diagnosis_results, display_debug_info,
    display_interpretation, display_model_comparison_table, 
    display_consistency_analysis, display_metrics_explanation,
    display_mcc_interpretation, display_technical_info,
    display_medical_disclaimer, display_pdf_generation_section,
    display_metrics_in_columns, display_mcnemar_results_table,
    display_statistical_conclusions
)
from metrics_calculator import (
    perform_mcnemar_comparisons, generate_simulated_metrics,
    calculate_advanced_metrics, matthews_correlation_coefficient,
    mcnemar_test, get_mcc_interpretation
)
from data_processor import (
    compare_all_models, get_model_metrics, analyze_consistency,
    create_mcc_comparison_dataframe, format_metrics_for_display,
    prepare_interpretation_text, get_recommendation_based_on_confidence
)
from visualization import (
    plot_confusion_matrix, create_metrics_dashboard,
    create_advanced_metrics_dashboard, create_model_comparison_plots,
    create_mcc_comparison_chart, create_mcnemar_plot
)
from pdf_generator_optimized import generate_pdf_report


@st.cache_resource
def load_models_cached():
    """
    Carga los modelos desde caché para mejorar rendimiento
    
    Returns:
        dict: Diccionario con los modelos cargados
    """
    try:
        models = load_models()
        if not models:
            st.error("❌ No se pudieron cargar los modelos entrenados.")
            st.error("📝 Asegúrate de que los archivos .h5 estén en la carpeta app/models/")
            return {}
        return models
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {str(e)}")
        return {}


def main():
    """
    Función principal de la aplicación
    """
    # Inicializar la aplicación
    initialize_page()
    
    # Cargar modelos entrenados
    models = load_models_cached()
    if not models:
        st.error("❌ No hay modelos disponibles. Verifica que los modelos entrenados estén en app/models/")
        st.stop()
    
    # Configuración de idioma
    available_languages = get_available_languages()
    
    # Configurar sidebar y obtener opciones
    sidebar_config = setup_sidebar(models, load_translations('es'), available_languages)
    
    # Cargar traducciones para el idioma seleccionado
    t = load_translations(sidebar_config['lang'])
    
    # Extraer configuraciones del sidebar
    selected_model = sidebar_config['selected_model']
    confidence_threshold = sidebar_config['confidence_threshold']
    decision_threshold = sidebar_config['decision_threshold']
    debug_mode = sidebar_config['debug_mode']
    
    # Mostrar encabezado principal
    display_main_header(t)
    
    # Crear pestañas principales
    tab1, tab2, tab3 = st.tabs([
        t.get('tab_diagnostic', 'Diagnóstico'), 
        t.get('tab_model_comparison', 'Comparación de Modelos'),
        t.get('tab_documentation', 'Documentación')
    ])
    
    # PESTAÑA 1: DIAGNÓSTICO
    with tab1:
        # Sección de carga de imagen
        uploaded_file = display_image_upload_section(t)
        
        if uploaded_file is not None:
            # Cargar y mostrar imagen original
            image = Image.open(uploaded_file)
            
            # Preprocesamiento
            processed_image = preprocess_image(np.array(image))
            
            # Mostrar comparación de imágenes
            display_image_comparison(image, processed_image, t)
            
            # Sección de diagnóstico
            st.header("🔍 " + t.get('diagnosis_results', "Resultados del Diagnóstico"))
            
            with st.spinner(t.get('processing_image', "Analizando imagen...")):
                model = models[selected_model]
                
                # Realizar predicción con el modelo seleccionado
                if debug_mode:
                    # Usar función de debug
                    diagnosis, confidence_percent, raw_confidence = predict_image_with_debug(model, processed_image)
                    
                    # Mostrar información de debug
                    display_debug_info(processed_image, model, decision_threshold, t)
                else:
                    # Usar función con umbral personalizado
                    diagnosis, confidence_percent, raw_confidence = predict_image_with_custom_threshold(
                        model, processed_image, threshold=decision_threshold
                    )
            
            # Mostrar resultados del diagnóstico
            display_diagnosis_results(diagnosis, confidence_percent, raw_confidence, t)
            
            # Interpretación de resultados
            display_interpretation(confidence_percent, confidence_threshold, diagnosis, t)
            
            # COMPARACIÓN DE TODOS LOS MODELOS
            st.markdown("---")
            st.subheader("📊 " + t.get('model_comparison', "Comparación de Todos los Modelos"))
            st.markdown(t.get('model_comparison_desc', "Resultados de análisis de la misma imagen con diferentes modelos:"))
            
            # Realizar predicciones con todos los modelos
            with st.spinner(t.get('comparing_models', "Comparando todos los modelos...")):
                comparison_results = compare_all_models(models, processed_image)
            
            # Mostrar tabla de comparación
            if comparison_results:
                display_model_comparison_table(pd.DataFrame(comparison_results))
                
                # Análisis de consistencia entre modelos
                display_consistency_analysis(comparison_results, t)
                
                # Gráficos de comparación
                fig1, fig2 = create_model_comparison_plots(pd.DataFrame(comparison_results))
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.pyplot(fig2)
            
            # SECCIÓN MATRIZ DE CONFUSIÓN Y MÉTRICAS
            st.markdown("---")
            st.subheader("📊 " + t.get('confusion_matrix', "Matriz de Confusión y Métricas"))
            st.markdown(t.get('metrics_desc', "Análisis detallado del rendimiento del modelo seleccionado:"))
            
            # Obtener métricas del modelo seleccionado
            metrics_data, is_real_data = get_model_metrics(selected_model)
            
            # Mostrar visualización de matriz de confusión
            if metrics_data:
                fig_cm = plot_confusion_matrix(metrics_data['confusion_matrix'], selected_model)
                if fig_cm:
                    st.pyplot(fig_cm)
                
                # Métricas en columnas para mejor visualización
                display_metrics_in_columns(metrics_data)
                
                # Dashboard de métricas
                fig_dashboard = create_metrics_dashboard(metrics_data, selected_model)
                if fig_dashboard:
                    st.pyplot(fig_dashboard)
                
                # Dashboard avanzado con MCC y otras métricas
                fig_advanced = create_advanced_metrics_dashboard(metrics_data, selected_model)
                if fig_advanced:
                    st.pyplot(fig_advanced)
                
                # Generar mapa de activación (GRADCAM) para explicabilidad
                st.markdown("---")
                st.subheader("🔍 " + t.get('activation_map', "Mapa de Activación (Grad-CAM)"))
                st.markdown(t.get('activation_map_desc', "Visualización de las regiones que el modelo consideró importantes para su decisión:"))
                
                with st.spinner(t.get('generating_map', "Generando mapa de activación...")):
                    activation_map = generate_gradcam(models[selected_model], processed_image)
                    if activation_map is not None:
                        st.image(activation_map, caption=t.get('activation_map_caption', "Mapa de calor de activación"), use_column_width=True)
                    else:
                        st.warning(t.get('activation_map_error', "No se pudo generar el mapa de activación para este modelo."))
                
                # Generar PDF
                generate_pdf = display_pdf_generation_section(t)
                if generate_pdf:
                    with st.spinner(t.get('generating_pdf', "Generando reporte PDF...")):
                        # Preparar datos para el PDF
                        plots_data = {
                            'confusion_matrix': fig_cm if 'fig_cm' in locals() else None,
                            'metrics_dashboard': fig_dashboard if 'fig_dashboard' in locals() else None,
                            'advanced_dashboard': fig_advanced if 'fig_advanced' in locals() else None,
                            'comparison_plots': {
                                'Comparacion de Confianza': fig1 if 'fig1' in locals() else None,
                                'Velocidad de Inferencia': fig2 if 'fig2' in locals() else None
                            }
                        }
                        
                        # Generar PDF
                        generate_pdf_report(
                            image=image,
                            diagnosis=diagnosis,
                            confidence_percent=confidence_percent,
                            raw_confidence=raw_confidence,
                            model_name=selected_model,
                            model_info=get_model_info(models[selected_model]),
                            comparison_results=comparison_results,
                            translations=t,
                            confidence_threshold=confidence_threshold,
                            metrics_data=metrics_data,
                            plots_data=plots_data
                        )
    
    # PESTAÑA 2: COMPARACIÓN DE MODELOS
    with tab2:
        st.header(t.get('model_comparison', "Comparación de Modelos"))
        
        # Mostrar tabla con información de todos los modelos
        model_info = {name: get_model_info(model) for name, model in models.items()}
        
        # Tabla de información de modelos
        st.subheader(t.get('model_info_title', "Información de los Modelos"))
        model_info_df = pd.DataFrame({
            'Modelo': list(model_info.keys()),
            'Parámetros': [info['parameters'] for info in model_info.values()],
            'Capas': [info['layers'] for info in model_info.values()],
            'Forma de Entrada': [str(info['input_shape']) for info in model_info.values()]
        })
        st.dataframe(model_info_df, use_container_width=True)
        
        # Comparación de métricas
        st.markdown("---")
        st.subheader(t.get('metrics_comparison', "Comparación de Métricas"))
        
        # Crear DataFrame con todas las métricas
        metrics_df = pd.DataFrame()
        for model_name in models.keys():
            metrics_data, _ = get_model_metrics(model_name)
            if metrics_data:
                metrics_row = {
                    'Modelo': model_name,
                    'Accuracy': f"{metrics_data['accuracy']:.3f}",
                    'Precision': f"{metrics_data['precision']:.3f}",
                    'Recall': f"{metrics_data['sensitivity']:.3f}",
                    'F1-Score': f"{metrics_data['f1_score']:.3f}",
                    'MCC': f"{metrics_data['mcc']:.3f}",
                    'Especificidad': f"{metrics_data['specificity']:.3f}"
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
        
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True)
            
            # Comparación de MCC (Matthews Correlation Coefficient)
            st.markdown("---")
            st.subheader(t.get('mcc_comparison', "Comparación de MCC"))
            
            # Crear DataFrame para MCC
            mcc_df = create_mcc_comparison_dataframe(MCC_COMPARISON_DATA)
            st.dataframe(mcc_df, use_container_width=True)
            
            # Gráfico de comparación MCC
            fig_mcc = create_mcc_comparison_chart(MCC_COMPARISON_DATA)
            if fig_mcc:
                st.pyplot(fig_mcc)
            
            # Interpretación de MCC
            display_mcc_interpretation(MCC_COMPARISON_DATA)
            
            # Análisis de McNemar para significancia estadística
            st.markdown("---")
            st.subheader(t.get('mcnemar_test', "Prueba de McNemar"))
            st.markdown(t.get('mcnemar_desc', "Análisis estadístico para determinar si existe diferencia significativa entre modelos:"))
            
            # Ejecutar pruebas de McNemar
            mcnemar_results = perform_mcnemar_comparisons()
            
            # Mostrar tabla de resultados
            display_mcnemar_results_table(mcnemar_results)
            
            # Gráfico de p-valores
            fig_mcnemar = create_mcnemar_plot(mcnemar_results)
            if fig_mcnemar:
                st.pyplot(fig_mcnemar)
            
            # Conclusiones estadísticas
            display_statistical_conclusions(mcnemar_results)
    
    # PESTAÑA 3: DOCUMENTACIÓN
    with tab3:
        st.header(t.get('documentation', "Documentación"))
        
        # Información técnica
        display_technical_info(get_model_info(models[selected_model]), t)
        
        # Explicación de métricas
        display_metrics_explanation()
        
        # Descargo de responsabilidad médica
        display_medical_disclaimer()


if __name__ == "__main__":
    main()
