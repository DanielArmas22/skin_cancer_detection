import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from system_manager import SkinCancerDiagnosisSystem
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Diagnóstico de Cáncer de Piel - Evaluación Real",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metrics-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🎯 Sistema de Diagnóstico de Cáncer de Piel</h1>', unsafe_allow_html=True)
st.markdown('<div class="success-box">🔬 <strong>Evaluación con Dataset Real ISIC</strong> - Sin datos simulados o hardcodeados</div>', unsafe_allow_html=True)

# Configurar sistema
@st.cache_resource
def initialize_system():
    """Inicializa el sistema de diagnóstico"""
    try:
        system = SkinCancerDiagnosisSystem()
        if system.initialize_components():
            return system
        else:
            st.error("❌ Error inicializando el sistema")
            return None
    except Exception as e:
        st.error(f"❌ Error al inicializar el sistema: {str(e)}")
        return None

@st.cache_data
def get_system_status():
    """Obtiene el estado del sistema"""
    system = initialize_system()
    if system:
        return system.get_system_status()
    return {}

# Sidebar
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["🏠 Inicio", "🔍 Diagnóstico", "📊 Evaluación de Modelos", "📈 Visualizaciones", "📄 Generar Reporte"]
)

# Inicializar sistema
system = initialize_system()

if page == "🏠 Inicio":
    st.markdown("## 🏠 Bienvenido al Sistema de Diagnóstico de Cáncer de Piel")
    
    # Obtener estado del sistema
    status = get_system_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metrics-card">
            <h3>🤖 Modelos Disponibles</h3>
            <p><strong>Total:</strong> {status.get('models_loaded', 0)}</p>
            <ul>
        """, unsafe_allow_html=True)
        
        for model_name in status.get('models_available', []):
            st.markdown(f"<li>{model_name}</li>", unsafe_allow_html=True)
        
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metrics-card">
            <h3>📊 Dataset de Evaluación</h3>
            <p><strong>ISIC Dataset de Test</strong></p>
            <p>• Imágenes reales de lesiones cutáneas</p>
            <p>• Casos benignos y malignos</p>
            <p>• Evaluación objetiva sin datos simulados</p>
            <p>• Dataset disponible: {'✅' if status.get('test_dataset_available') else '❌'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metrics-card">
            <h3>⚡ Características</h3>
            <p>• Métricas calculadas en tiempo real</p>
            <p>• Análisis estadístico avanzado</p>
            <p>• Visualizaciones interactivas</p>
            <p>• Reportes PDF completos</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información del sistema
    st.markdown("## 🔧 Estado del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Modelos Cargados")
        if models:
            for model_name, model in models.items():
                model_info = get_model_info(model)
                st.success(f"✅ {model_name}: {model_info['parameters']:,} parámetros")
        else:
            st.error("❌ No se cargaron modelos")
    
    with col2:
        st.markdown("### 📊 Cache de Métricas")
        metrics = load_model_metrics()
        if metrics:
            st.success(f"✅ Métricas disponibles para {len(metrics)} modelos")
            for model_name, model_metrics in metrics.items():
                st.info(f"📈 {model_name}: MCC = {model_metrics['mcc']:.3f}")
        else:
            st.warning("⚠️ No hay métricas en cache. Ejecuta la evaluación primero.")

elif page == "🔍 Diagnóstico":
    st.markdown("## 🔍 Diagnóstico de Imagen")
    
    if not models:
        st.error("❌ No hay modelos disponibles para diagnóstico")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Cargar Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de lesión cutánea",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos soportados: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Mostrar imagen cargada
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Selector de modelo
            st.markdown("### 🤖 Seleccionar Modelo")
            selected_model = st.selectbox(
                "Elige el modelo para diagnóstico:",
                list(models.keys()),
                help="Selecciona el modelo que deseas usar para el diagnóstico"
            )
            
            # Botón de diagnóstico
            if st.button("🔍 Realizar Diagnóstico", type="primary"):
                with st.spinner("Procesando imagen..."):
                    # Preprocesar imagen
                    processed_image = preprocess_image(image)
                    
                    # Realizar predicción
                    start_time = time.time()
                    diagnosis, confidence_percent, raw_confidence = predict_image(
                        models[selected_model], processed_image
                    )
                    processing_time = time.time() - start_time
                    
                    # Mostrar resultados en col2
                    with col2:
                        st.markdown("### 📊 Resultados del Diagnóstico")
                        
                        # Mostrar diagnóstico
                        if diagnosis == "Maligno":
                            st.error(f"🚨 **Diagnóstico: {diagnosis}**")
                            st.error(f"⚠️ **Confianza: {confidence_percent:.1f}%**")
                        else:
                            st.success(f"✅ **Diagnóstico: {diagnosis}**")
                            st.success(f"✅ **Confianza: {confidence_percent:.1f}%**")
                        
                        # Información adicional
                        st.info(f"🤖 **Modelo usado:** {selected_model}")
                        st.info(f"⏱️ **Tiempo de procesamiento:** {processing_time:.2f} segundos")
                        st.info(f"📊 **Valor raw:** {raw_confidence:.3f}")
                        
                        # Interpretación
                        st.markdown("### 🔬 Interpretación")
                        if confidence_percent > 75:
                            st.success("🎯 **Alta confianza** en el diagnóstico")
                        elif confidence_percent > 50:
                            st.warning("⚠️ **Confianza moderada** - considerar segunda opinión")
                        else:
                            st.error("❌ **Baja confianza** - se requiere evaluación médica")
                        
                        # Disclaimer médico
                        st.markdown("""
                        <div class="warning-box">
                            <strong>⚠️ IMPORTANTE:</strong><br>
                            Este sistema es solo para fines educativos y de investigación.<br>
                            <strong>NO reemplaza el diagnóstico médico profesional.</strong><br>
                            Consulta siempre con un dermatólogo certificado.
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is None:
            st.markdown("### 📋 Instrucciones")
            st.markdown("""
            1. **Carga una imagen** de lesión cutánea
            2. **Selecciona el modelo** que deseas usar
            3. **Haz clic en "Realizar Diagnóstico"**
            4. **Revisa los resultados** y la interpretación
            
            **Recomendaciones para mejores resultados:**
            - Usa imágenes de alta calidad
            - Asegúrate de que la lesión sea visible
            - Evita imágenes borrosas o con mala iluminación
            """)

elif page == "📊 Evaluación de Modelos":
    st.markdown("## 📊 Evaluación Completa de Modelos")
    
    # Inicializar evaluador
    evaluator = initialize_evaluator()
    
    if not evaluator:
        st.error("❌ No se pudo inicializar el evaluador")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuración de Evaluación")
        
        # Opción para forzar recálculo
        force_refresh = st.checkbox(
            "🔄 Forzar recálculo de métricas",
            help="Marca esto para recalcular todas las métricas desde cero"
        )
        
        # Botón de evaluación
        if st.button("🚀 Evaluar Todos los Modelos", type="primary"):
            with st.spinner("Evaluando modelos en dataset de test..."):
                try:
                    # Ejecutar evaluación completa
                    all_metrics = evaluator.evaluate_all_models(force_refresh=force_refresh)
                    
                    if all_metrics:
                        st.success(f"✅ Evaluación completada: {len(all_metrics)} modelos")
                        
                        # Mostrar resumen
                        st.markdown("### 📈 Resumen de Evaluación")
                        
                        # Encontrar mejor modelo
                        best_model = max(all_metrics.items(), key=lambda x: x[1]['mcc'])
                        st.success(f"🏆 **Mejor modelo:** {best_model[0]} (MCC: {best_model[1]['mcc']:.3f})")
                        
                        # Mostrar métricas básicas
                        for model_name, metrics in all_metrics.items():
                            with st.expander(f"📊 {model_name}", expanded=False):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                                    st.metric("Precision", f"{metrics['precision']:.3f}")
                                    st.metric("Recall", f"{metrics['recall']:.3f}")
                                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                                
                                with col_b:
                                    st.metric("MCC", f"{metrics['mcc']:.3f}")
                                    st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
                                    st.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
                                    st.metric("Specificity", f"{metrics['specificity']:.3f}")
                        
                        # Guardar métricas en session state
                        st.session_state.model_metrics = all_metrics
                        
                    else:
                        st.error("❌ No se pudieron obtener métricas")
                        
                except Exception as e:
                    st.error(f"❌ Error durante la evaluación: {str(e)}")
    
    with col2:
        # Mostrar métricas existentes
        metrics = load_model_metrics()
        
        if metrics:
            st.markdown("### 📊 Métricas Actuales")
            
            # Crear DataFrame para mostrar tabla
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Seleccionar columnas importantes
            display_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc_roc']
            df_display = df_metrics[display_cols].round(3)
            
            # Mostrar tabla
            st.dataframe(df_display, use_container_width=True)
            
            # Información adicional
            st.markdown("### 📋 Información del Dataset")
            first_model = list(metrics.values())[0]
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Total de Muestras", f"{first_model['test_samples']:,}")
                st.metric("Casos Benignos", f"{first_model['class_distribution'][0]:,}")
            
            with col_b:
                st.metric("Casos Malignos", f"{first_model['class_distribution'][1]:,}")
                benign_pct = first_model['class_distribution'][0] / first_model['test_samples'] * 100
                st.metric("% Benignos", f"{benign_pct:.1f}%")
            
            # Comparación estadística
            st.markdown("### 📊 Comparación Estadística")
            comparison_results = evaluator.load_comparison_results()
            
            if comparison_results:
                st.markdown("**Pruebas de McNemar:**")
                
                for comparison, result in comparison_results.items():
                    if result:
                        models = comparison.split(' vs ')
                        better_model = models[0] if result.get('model1_better', False) else models[1]
                        
                        if result['significant']:
                            st.success(f"✅ {comparison}: **{better_model}** es significativamente mejor (p={result['p_value']:.3f})")
                        else:
                            st.info(f"ℹ️ {comparison}: No hay diferencia significativa (p={result['p_value']:.3f})")
            
        else:
            st.warning("⚠️ No hay métricas disponibles. Ejecuta la evaluación primero.")

elif page == "📈 Visualizaciones":
    st.markdown("## 📈 Visualizaciones Avanzadas")
    
    # Cargar métricas
    metrics = load_model_metrics()
    
    if not metrics:
        st.warning("⚠️ No hay métricas disponibles. Ejecuta la evaluación primero.")
        st.stop()
    
    # Inicializar visualizador
    visualizer = MetricsVisualizer(metrics)
    
    # Sidebar para seleccionar visualizaciones
    st.sidebar.markdown("### 📊 Seleccionar Visualizaciones")
    
    show_confusion_matrices = st.sidebar.checkbox("🎯 Matrices de Confusión", True)
    show_metrics_comparison = st.sidebar.checkbox("📊 Comparación de Métricas", True)
    show_roc_curves = st.sidebar.checkbox("📈 Curvas ROC", True)
    show_radar_chart = st.sidebar.checkbox("🎪 Gráfico de Radar", True)
    show_mcc_comparison = st.sidebar.checkbox("⚡ Comparación MCC", True)
    
    # Mostrar visualizaciones seleccionadas
    if show_confusion_matrices:
        st.markdown("### 🎯 Matrices de Confusión")
        
        # Selector de modelo para matriz individual
        selected_model = st.selectbox(
            "Selecciona un modelo para ver su matriz de confusión:",
            list(metrics.keys())
        )
        
        # Mostrar matriz seleccionada
        fig_cm = visualizer.plot_confusion_matrix(selected_model)
        if fig_cm:
            st.pyplot(fig_cm)
        
        # Mostrar comparación de todas las matrices
        st.markdown("#### 📊 Comparación de Todas las Matrices")
        fig_cm_comp = visualizer.plot_confusion_matrices_comparison()
        if fig_cm_comp:
            st.pyplot(fig_cm_comp)
    
    if show_metrics_comparison:
        st.markdown("### 📊 Comparación de Métricas")
        fig_metrics = visualizer.plot_metrics_comparison()
        if fig_metrics:
            st.pyplot(fig_metrics)
    
    if show_roc_curves:
        st.markdown("### 📈 Curvas ROC")
        fig_roc = visualizer.plot_roc_curves()
        if fig_roc:
            st.pyplot(fig_roc)
    
    if show_radar_chart:
        st.markdown("### 🎪 Gráfico de Radar - Rendimiento")
        fig_radar = visualizer.plot_performance_radar()
        if fig_radar:
            st.pyplot(fig_radar)
    
    if show_mcc_comparison:
        st.markdown("### ⚡ Comparación MCC")
        fig_mcc = visualizer.plot_mcc_comparison()
        if fig_mcc:
            st.pyplot(fig_mcc)
    
    # Tabla de métricas detalladas
    st.markdown("### 📋 Tabla de Métricas Detalladas")
    df_detailed = visualizer.create_metrics_summary_table()
    st.dataframe(df_detailed, use_container_width=True)
    
    # Insights automáticos
    st.markdown("### 🔍 Insights Automáticos")
    insights = visualizer.generate_insights()
    
    for insight in insights:
        st.info(insight)
    
    # Botón para guardar todos los gráficos
    if st.button("💾 Guardar Todas las Visualizaciones"):
        with st.spinner("Guardando visualizaciones..."):
            saved_plots = visualizer.save_all_plots()
            if saved_plots:
                st.success(f"✅ Se guardaron {len(saved_plots)} visualizaciones en app/plots/")
            else:
                st.error("❌ Error al guardar visualizaciones")

elif page == "📄 Generar Reporte":
    st.markdown("## 📄 Generar Reporte PDF Completo")
    
    # Cargar métricas
    metrics = load_model_metrics()
    
    if not metrics:
        st.warning("⚠️ No hay métricas disponibles. Ejecuta la evaluación primero.")
        st.stop()
    
    # Configuración del reporte
    st.markdown("### ⚙️ Configuración del Reporte")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input(
            "Título del reporte:",
            "Reporte de Evaluacion de Modelos de Cancer de Piel"
        )
        
        include_plots = st.checkbox("📊 Incluir visualizaciones", True)
        include_methodology = st.checkbox("📋 Incluir metodología", True)
        include_disclaimer = st.checkbox("⚠️ Incluir disclaimer médico", True)
    
    with col2:
        output_filename = st.text_input(
            "Nombre del archivo:",
            "reporte_evaluacion_completo.pdf"
        )
        
        # Información del reporte
        st.markdown("#### 📊 Contenido del Reporte")
        st.info(f"""
        **Modelos incluidos:** {len(metrics)}
        **Métricas por modelo:** 8+
        **Análisis estadístico:** Pruebas McNemar
        **Visualizaciones:** {6 if include_plots else 0}
        """)
    
    # Botón para generar reporte
    if st.button("📄 Generar Reporte PDF", type="primary"):
        with st.spinner("Generando reporte completo..."):
            try:
                # Inicializar evaluador y visualizador
                evaluator = initialize_evaluator()
                visualizer = MetricsVisualizer(metrics)
                
                # Generar visualizaciones si se solicita
                plot_paths = None
                if include_plots:
                    plot_paths = visualizer.save_all_plots()
                
                # Cargar resultados de comparación
                comparison_results = None
                if evaluator:
                    comparison_results = evaluator.load_comparisons()
                
                # Generar PDF
                pdf_generator = PDFReportGenerator()
                
                output_path = pdf_generator.generate_complete_report(
                    metrics_data=metrics,
                    comparison_data=comparison_results,
                    title=report_title,
                    include_images=include_plots,
                    output_filename=output_filename
                )
                
                if output_path and os.path.exists(output_path):
                    st.success(f"✅ Reporte generado exitosamente: {output_path}")
                    
                    # Mostrar información del archivo
                    file_size = os.path.getsize(output_path) / 1024  # KB
                    st.info(f"📄 Tamaño del archivo: {file_size:.1f} KB")
                    
                    # Botón de descarga
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Descargar Reporte PDF",
                            data=file.read(),
                            file_name=output_filename,
                            mime="application/pdf"
                        )
                    
                    # Mostrar preview del contenido
                    st.markdown("### 📋 Contenido del Reporte Generado")
                    st.markdown("""
                    1. **Página de Título** - Información general
                    2. **Resumen Ejecutivo** - Métricas principales
                    3. **Información del Dataset** - Datos ISIC utilizados
                    4. **Metodología** - Proceso de evaluación
                    5. **Análisis Detallado** - Por cada modelo
                    6. **Comparación Estadística** - Pruebas McNemar
                    7. **Visualizaciones** - Gráficos y matrices
                    8. **Conclusiones** - Recomendaciones finales
                    9. **Disclaimer Médico** - Limitaciones y responsabilidades
                    """)
                    
                else:
                    st.error("❌ Error al generar el reporte PDF")
                    
            except Exception as e:
                st.error(f"❌ Error durante la generación del reporte: {str(e)}")
    
    # Información adicional
    st.markdown("---")
    st.markdown("### ℹ️ Información sobre el Reporte")
    st.markdown("""
    **Características del reporte:**
    - ✅ Datos reales del dataset ISIC
    - ✅ Métricas calculadas objetivamente
    - ✅ Análisis estadístico completo
    - ✅ Visualizaciones profesionales
    - ✅ Interpretación médica
    - ✅ Recomendaciones basadas en evidencia
    
    **Uso recomendado:**
    - 📚 Documentación académica
    - 📊 Presentaciones profesionales
    - 📋 Análisis de rendimiento
    - 🔬 Investigación científica
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🩺 Sistema de Diagnóstico de Cáncer de Piel - Versión Mejorada</p>
    <p>📊 Evaluación con Dataset Real ISIC | 🚫 Sin Datos Simulados</p>
    <p>⚠️ Solo para fines educativos y de investigación</p>
</div>
""", unsafe_allow_html=True)
