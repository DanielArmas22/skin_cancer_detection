# app.py
"""
Aplicación principal para el sistema de diagnóstico de cáncer de piel
"""

import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Importar módulos propios
from config import initialize_page, MCC_COMPARISON_DATA
from model_utils import load_models, predict_image, predict_image_with_debug, predict_image_with_custom_threshold, get_model_info
from preprocessing import preprocess_image
from ui_components import (
    setup_sidebar, display_main_header, display_image_upload_section, display_image_comparison,
    display_diagnosis_results, display_debug_info, display_interpretation, display_model_comparison_table,
    display_consistency_analysis, display_metrics_explanation, display_metrics_in_columns,
    display_mcc_interpretation, display_technical_info, display_medical_disclaimer,
    display_pdf_generation_section, display_mcnemar_results_table, display_statistical_conclusions
)
from data_processor import (
    compare_all_models, analyze_consistency, get_model_metrics,
    create_mcc_comparison_dataframe, format_metrics_for_display
)
from activation_maps import generate_gradcam
from visualization import (
    plot_confusion_matrix, create_metrics_dashboard, create_advanced_metrics_dashboard,
    create_model_comparison_plots, create_mcc_comparison_chart, create_mcnemar_plot
)
from metrics_calculator import perform_mcnemar_comparisons
from pdf_generator_optimized import generate_pdf_report
from translations import get_available_languages, load_translations


def main():
    """Función principal de la aplicación"""
    # Inicializar la configuración de la página
    initialize_page()
    
    # Configuración inicial para traducciones (español por defecto)
    available_languages = get_available_languages()
    t = load_translations('es')
    
    # Cargar modelos entrenados
    @st.cache_resource
    def load_models_cached():
        try:
            models = load_models()
            if not models:
                st.error("❌ " + t.get('models_load_error', "No se pudieron cargar los modelos entrenados."))
                st.error("📝 " + t.get('models_folder_check', "Asegúrate de que los archivos .h5 estén en la carpeta app/models/"))
                return {}
            return models
        except Exception as e:
            st.error(f"❌ {t.get('model_load_exception', 'Error al cargar los modelos')}: {str(e)}")
            return {}

    models = load_models_cached()
    model_names = list(models.keys())

    if not model_names:
        st.error("❌ " + t.get('no_models_available', "No hay modelos disponibles. Verifica que los modelos entrenados estén en app/models/"))
        st.stop()
    
    # Configurar la barra lateral y obtener la configuración del usuario
    sidebar_config = setup_sidebar(models, t, available_languages)
    
    # Actualizar las traducciones según el idioma seleccionado
    current_lang_code = available_languages[sidebar_config['lang']]
    t = load_translations(current_lang_code)
    
    # Mostrar encabezado principal
    display_main_header(t)
    
    # Sección de carga de imagen
    uploaded_file = display_image_upload_section(t)
    
    # Procesamiento de la imagen si se sube un archivo
    if uploaded_file is not None:
        # Cargar y mostrar imagen original
        image = Image.open(uploaded_file)
        
        # Preprocesamiento
        processed_image = preprocess_image(np.array(image))
        
        # Mostrar comparación de imágenes
        display_image_comparison(image, processed_image, t)
        
        # Realizar predicción con el modelo seleccionado
        st.header("🔍 " + t.get('diagnosis_results', "Resultados del Diagnóstico"))
        
        with st.spinner(t.get('processing_image', "Analizando imagen...")):
            model = models[sidebar_config['selected_model']]
            
            if sidebar_config['debug_mode']:
                # Usar función de debug
                diagnosis, confidence_percent, raw_confidence = predict_image_with_debug(model, processed_image)
                
                # Mostrar información de debug
                display_debug_info(processed_image, model, sidebar_config['decision_threshold'], t)
            else:
                # Usar función con umbral personalizado
                diagnosis, confidence_percent, raw_confidence = predict_image_with_custom_threshold(
                    model, processed_image, threshold=sidebar_config['decision_threshold']
                )
        
        # Mostrar resultados con mejor diseño
        display_diagnosis_results(diagnosis, confidence_percent, raw_confidence, t)
        
        # Interpretación de resultados
        display_interpretation(confidence_percent, sidebar_config['confidence_threshold'], diagnosis, t)
        
        # COMPARACIÓN DE TODOS LOS MODELOS
        st.markdown("---")
        st.subheader("📊 " + t.get('model_comparison', "Comparación de Todos los Modelos"))
        st.markdown("Resultados de análisis de la misma imagen con diferentes modelos:")
        
        # Realizar predicciones con todos los modelos
        with st.spinner("Comparando todos los modelos..."):
            comparison_results = compare_all_models(models, processed_image)
        
        # Mostrar tabla de comparación
        if comparison_results:
            df_comparison = pd.DataFrame(comparison_results)
            display_model_comparison_table(df_comparison)
            
            # Gráficos de comparación
            fig_confianza, fig_tiempo = create_model_comparison_plots(df_comparison)
            if fig_confianza:
                st.pyplot(fig_confianza)
            if fig_tiempo:
                st.pyplot(fig_tiempo)
            
            # Análisis de consistencia
            display_consistency_analysis(comparison_results, t)
        
        # MATRIZ DE CONFUSIÓN Y MÉTRICAS
        st.markdown("---")
        st.subheader("📊 " + t.get('confusion_matrix', "Matriz de Confusión y Métricas"))
        st.markdown("Análisis detallado del rendimiento del modelo seleccionado:")
        
        # Obtener métricas del modelo seleccionado
        metrics_data, is_real_data = get_model_metrics(sidebar_config['selected_model'])
        
        if is_real_data:
            st.success(f"✅ **Datos Reales de Entrenamiento**: Mostrando métricas reales del modelo {sidebar_config['selected_model']} en el dataset ISIC 2019")
        else:
            st.warning("⚠️ **Datos Simulados**: Usando datos de ejemplo para demostración")
        
        # Mostrar matriz de confusión
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Matriz de Confusión**")
            fig_cm = plot_confusion_matrix(metrics_data['confusion_matrix'], sidebar_config['selected_model'])
            if fig_cm:
                st.pyplot(fig_cm)
        
        with col2:
            st.markdown("**📈 Métricas de Rendimiento Avanzadas**")
            display_metrics_in_columns(metrics_data)
            
            # Interpretación de métricas
            st.markdown("**📋 Interpretación:**")
            formatted_metrics = format_metrics_for_display(metrics_data)
            st.markdown(f"""
            - **Accuracy**: {metrics_data['accuracy']*100:.1f}% de las predicciones son correctas
            - **Sensitivity**: {metrics_data['sensitivity']*100:.1f}% de los casos malignos son detectados
            - **Specificity**: {metrics_data['specificity']*100:.1f}% de los casos benignos son correctamente identificados
            - **Precision**: {metrics_data['precision']*100:.1f}% de los casos clasificados como malignos son realmente malignos
            - **F1-Score**: {metrics_data['f1_score']*100:.1f}% es el balance entre precisión y sensibilidad
            - **MCC**: {metrics_data['mcc']:.3f} (Coeficiente de Matthews - balanceado para clases desequilibradas)
            """)
        
        # Dashboard completo de métricas
        st.markdown("---")
        st.subheader("📊 Dashboard Completo de Métricas")
        
        fig_dashboard = create_metrics_dashboard(metrics_data, sidebar_config['selected_model'])
        if fig_dashboard:
            st.pyplot(fig_dashboard)
        
        # Visualización de mapas de activación
        st.markdown("---")
        st.subheader("🔍 Visualización de Mapas de Activación")
        st.markdown("Visualización de las regiones de la imagen que más influyeron en el diagnóstico:")
        
        with st.spinner("Generando mapa de activación..."):
            activation_map = generate_gradcam(model, processed_image)
            if activation_map is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(processed_image, caption="Imagen Procesada", use_column_width=True)
                with col2:
                    st.image(activation_map, caption="Mapa de Activación (Grad-CAM)", use_column_width=True)
                st.info("El mapa de calor muestra las regiones que más influyeron en el diagnóstico del modelo. Las áreas rojas y amarillas son las más relevantes.")
            else:
                st.error("No se pudo generar el mapa de activación para este modelo.")
        
        # Explicación de la matriz de confusión
        display_metrics_explanation()
        
        # Análisis Estadístico Avanzado
        st.markdown("---")
        st.subheader("🔬 Análisis Estadístico Avanzado")
        st.markdown("Incluyendo Coeficiente de Matthews y Pruebas de McNemar:")
        
        # Crear dashboard avanzado con MCC
        fig_advanced = create_advanced_metrics_dashboard(metrics_data, sidebar_config['selected_model'])
        if fig_advanced:
            st.pyplot(fig_advanced)
        
        # Tabla de Resumen MCC y Gráfico Comparativo
        st.markdown("---")
        st.subheader("📊 Resumen Comparativo de Coeficientes de Matthews (MCC)")
        st.markdown("Comparación de todos los modelos basada en el Coeficiente de Matthews:")
        
        # Crear DataFrame para la tabla
        df_mcc = create_mcc_comparison_dataframe(MCC_COMPARISON_DATA)
        
        # Mostrar tabla con formato mejorado
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**📋 Tabla de Resumen - Coeficientes de Matthews**")
            st.dataframe(df_mcc, use_container_width=True)
        
        with col2:
            display_mcc_interpretation(MCC_COMPARISON_DATA)
        
        # Gráfico MCC
        fig_mcc = create_mcc_comparison_chart(MCC_COMPARISON_DATA)
        if fig_mcc:
            st.pyplot(fig_mcc)
        
        # Análisis Estadístico - Pruebas de McNemar
        st.markdown("---")
        st.subheader("🔬 Pruebas Estadísticas de McNemar")
        st.markdown("Comparación estadística entre modelos:")
        
        # Generar resultados de McNemar
        mcnemar_results = perform_mcnemar_comparisons()
        
        # Mostrar tabla de resultados
        display_mcnemar_results_table(mcnemar_results)
        
        # Gráfico de p-valores
        fig_mcnemar = create_mcnemar_plot(mcnemar_results)
        if fig_mcnemar:
            st.pyplot(fig_mcnemar)
        
        # Conclusiones estadísticas
        display_statistical_conclusions(mcnemar_results)
        
        # Generar reporte PDF
        generate_pdf = display_pdf_generation_section(t)
        
        if generate_pdf:
            with st.spinner("Generando reporte PDF..."):
                # Preparar datos para el PDF
                plots_data = {
                    'confusion_matrix': fig_cm if 'fig_cm' in locals() else None,
                    'metrics_dashboard': fig_dashboard if 'fig_dashboard' in locals() else None,
                    'advanced_dashboard': fig_advanced if 'fig_advanced' in locals() else None,
                    'comparison_plots': {
                        'Comparacion de Confianza': fig_confianza if 'fig_confianza' in locals() else None,
                        'Velocidad de Inferencia': fig_tiempo if 'fig_tiempo' in locals() else None,
                        'MCC Comparativo': fig_mcc if 'fig_mcc' in locals() else None,
                        'McNemar P-valores': fig_mcnemar if 'fig_mcnemar' in locals() else None
                    }
                }
                
                # Generar PDF
                generate_pdf_report(
                    image=image,
                    diagnosis=diagnosis,
                    confidence_percent=confidence_percent,
                    raw_confidence=raw_confidence,
                    model_name=sidebar_config['selected_model'],
                    model_info=get_model_info(models[sidebar_config['selected_model']]),
                    comparison_results=comparison_results,
                    translations=t,
                    confidence_threshold=sidebar_config['confidence_threshold'],
                    metrics_data=metrics_data,
                    plots_data=plots_data
                )
        
        # Información técnica
        display_technical_info(get_model_info(models[sidebar_config['selected_model']]), t)
        
        # Advertencia médica
        display_medical_disclaimer()


if __name__ == "__main__":
    main()
