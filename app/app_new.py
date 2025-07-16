import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar sistema manager
from system_manager import SkinCancerDiagnosisSystem

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Diagnóstico de Cáncer de Piel - Versión 2.0",
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
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🩺 Sistema de Diagnóstico de Cáncer de Piel v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="success-box">🔬 <strong>Sistema Completamente Automatizado</strong> - Evaluación con Dataset Real ISIC</div>', unsafe_allow_html=True)

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
            <p>• Sistema completamente automatizado</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información del sistema
    st.markdown("## 🔧 Estado del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Modelos Cargados")
        if status.get('models_loaded', 0) > 0:
            for model_name in status.get('models_available', []):
                st.success(f"✅ {model_name}")
                
            # Mostrar mejor modelo si está disponible
            if 'best_model' in status:
                best_model = status['best_model']
                st.info(f"🏆 **Mejor modelo:** {best_model['name']} (MCC: {best_model['mcc']:.3f})")
        else:
            st.error("❌ No se cargaron modelos")
    
    with col2:
        st.markdown("### 📊 Cache de Métricas")
        if status.get('metrics_cached', 0) > 0:
            st.success(f"✅ Métricas disponibles para {status['metrics_cached']} modelos")
            
            if status.get('last_evaluation'):
                st.info(f"📅 Última evaluación: {status['last_evaluation'][:19].replace('T', ' ')}")
        else:
            st.warning("⚠️ No hay métricas en cache. Ejecuta la evaluación primero.")
        
        st.markdown("### 🔧 Componentes del Sistema")
        components = [
            ("Evaluador", status.get('evaluator_ready', False)),
            ("Visualizador", status.get('visualizer_ready', False)),
            ("Generador PDF", status.get('pdf_generator_ready', False))
        ]
        
        for name, ready in components:
            if ready:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")
    
    # Información de uso
    st.markdown("## 📖 Cómo usar el sistema")
    st.markdown("""
    ### 🔍 Para Diagnóstico:
    1. **Ve a la sección "Diagnóstico"**
    2. **Sube una imagen** de lesión cutánea
    3. **El sistema seleccionará automáticamente el mejor modelo**
    4. **Haz clic en "Realizar Diagnóstico"**
    5. **Revisa los resultados** y la interpretación
    
    ### 📊 Para Evaluación:
    1. **Ve a "Evaluación de Modelos"**
    2. **Haz clic en "Evaluar Todos los Modelos"**
    3. **Revisa las métricas** y comparaciones estadísticas
    
    ### 📈 Para Visualizaciones:
    1. **Asegúrate de tener métricas evaluadas**
    2. **Ve a "Visualizaciones"**
    3. **Selecciona los gráficos** que deseas ver
    
    ### 📄 Para Reportes:
    1. **Ve a "Generar Reporte"**
    2. **Configura las opciones** del reporte
    3. **Genera el PDF** con todos los resultados
    """)
    
    st.markdown("---")
    
    # Versión del sistema
    st.markdown(f"**Versión del Sistema:** {status.get('system_version', 'N/A')}")
    
    # Disclaimer médico
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ IMPORTANTE:</strong><br>
        Este sistema es solo para fines educativos y de investigación.<br>
        <strong>NO reemplaza el diagnóstico médico profesional.</strong><br>
        Siempre consulte a un profesional médico calificado para diagnósticos reales.
    </div>
    """, unsafe_allow_html=True)

elif page == "🔍 Diagnóstico":
    st.markdown("## 🔍 Diagnóstico de Imagen")
    
    if not system:
        st.error("❌ Sistema no disponible")
        st.stop()
    
    # Obtener información de modelos
    models_info = system.get_models_info()
    
    if not models_info:
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
            
            # Mostrar información del mejor modelo
            status = get_system_status()
            if 'best_model' in status:
                best_model = status['best_model']
                st.info(f"🏆 Usando el mejor modelo: **{best_model['name']}** (MCC: {best_model['mcc']:.3f})")
            
            # Selector de modelo (opcional)
            st.markdown("### 🤖 Seleccionar Modelo (Opcional)")
            model_options = ["Automático (Mejor modelo)"] + list(models_info.keys())
            selected_model = st.selectbox(
                "Elige el modelo para diagnóstico:",
                model_options,
                help="Selecciona 'Automático' para usar el mejor modelo según MCC"
            )
            
            # Botón de diagnóstico
            if st.button("🔍 Realizar Diagnóstico", type="primary"):
                with st.spinner("Procesando imagen..."):
                    start_time = time.time()
                    
                    # Realizar predicción
                    model_name = None if selected_model == "Automático (Mejor modelo)" else selected_model
                    result = system.predict_single_image(image, model_name)
                    
                    processing_time = time.time() - start_time
                    
                    # Mostrar resultados en col2
                    with col2:
                        if 'error' in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            st.markdown("### 📊 Resultados del Diagnóstico")
                            
                            # Mostrar diagnóstico
                            diagnosis = result['diagnosis']
                            confidence_percent = result['confidence_percent']
                            
                            if diagnosis == "Maligno":
                                st.error(f"🚨 **Diagnóstico: {diagnosis}**")
                                st.error(f"⚠️ **Confianza: {confidence_percent:.1f}%**")
                            else:
                                st.success(f"✅ **Diagnóstico: {diagnosis}**")
                                st.success(f"✅ **Confianza: {confidence_percent:.1f}%**")
                            
                            # Información del modelo
                            st.info(f"🤖 **Modelo usado:** {result['model_used']}")
                            st.info(f"⏱️ **Tiempo de procesamiento:** {processing_time:.2f} segundos")
                            st.info(f"📊 **Valor raw:** {result['raw_confidence']:.3f}")
                            
                            # Métricas del modelo
                            st.markdown("### 📈 Métricas del Modelo")
                            model_metrics = result['model_metrics']
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if model_metrics['accuracy'] != 'N/A':
                                    st.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
                                if model_metrics['precision'] != 'N/A':
                                    st.metric("Precision", f"{model_metrics['precision']:.3f}")
                            
                            with col_b:
                                if model_metrics['recall'] != 'N/A':
                                    st.metric("Recall", f"{model_metrics['recall']:.3f}")
                                if model_metrics['mcc'] != 'N/A':
                                    st.metric("MCC", f"{model_metrics['mcc']:.3f}")
                            
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
                                Siempre consulte a un profesional médico calificado.
                            </div>
                            """, unsafe_allow_html=True)

elif page == "📊 Evaluación de Modelos":
    st.markdown("## 📊 Evaluación Completa de Modelos")
    
    if not system:
        st.error("❌ Sistema no disponible")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuración de Evaluación")
        
        # Opción para forzar recálculo
        force_refresh = st.checkbox(
            "🔄 Forzar recálculo de métricas",
            value=False,
            help="Marca esto para recalcular todas las métricas desde cero"
        )
        
        # Botón de evaluación
        if st.button("🚀 Evaluar Todos los Modelos", type="primary"):
            with st.spinner("Evaluando modelos en dataset de test..."):
                try:
                    # Ejecutar evaluación completa
                    metrics = system.evaluate_models(force_refresh=force_refresh)
                    
                    if metrics:
                        st.success(f"✅ Evaluación completada: {len(metrics)} modelos")
                        
                        # Recargar datos del sistema
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("❌ No se pudieron obtener métricas")
                        
                except Exception as e:
                    st.error(f"❌ Error durante la evaluación: {str(e)}")
    
    with col2:
        # Mostrar métricas existentes
        summary = system.get_evaluation_summary()
        
        if 'error' not in summary:
            st.markdown("### 📊 Resumen de Evaluación")
            
            # Métricas principales
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Modelos Evaluados", summary['total_models'])
                st.metric("Muestras de Test", f"{summary['test_samples']:,}")
            
            with col_b:
                st.metric("Accuracy Promedio", f"{summary['average_accuracy']:.3f}")
                st.metric("MCC Promedio", f"{summary['average_mcc']:.3f}")
            
            with col_c:
                best_model = summary['best_model']
                st.metric("Mejor Modelo", best_model['name'])
                st.metric("MCC del Mejor", f"{best_model['mcc']:.3f}")
            
            # Tabla de métricas
            st.markdown("### 📋 Tabla de Métricas Detalladas")
            
            # Crear DataFrame para mostrar
            metrics_data = []
            for model_name, mcc in summary['model_rankings']:
                metrics_data.append({
                    'Modelo': model_name,
                    'MCC': f"{mcc:.3f}",
                    'Ranking': f"#{summary['model_rankings'].index((model_name, mcc)) + 1}"
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            # Información del dataset
            st.markdown("### 📋 Información del Dataset")
            class_dist = summary['class_distribution']
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Casos Benignos", f"{class_dist[0]:,}")
                benign_pct = class_dist[0] / summary['test_samples'] * 100
                st.metric("% Benignos", f"{benign_pct:.1f}%")
            
            with col_b:
                st.metric("Casos Malignos", f"{class_dist[1]:,}")
                malign_pct = class_dist[1] / summary['test_samples'] * 100
                st.metric("% Malignos", f"{malign_pct:.1f}%")
            
            # Última evaluación
            if summary.get('last_evaluation'):
                st.info(f"📅 Última evaluación: {summary['last_evaluation'][:19].replace('T', ' ')}")
        
        else:
            st.warning("⚠️ No hay métricas disponibles. Ejecuta la evaluación primero.")

elif page == "📈 Visualizaciones":
    st.markdown("## 📈 Visualizaciones Avanzadas")
    
    if not system:
        st.error("❌ Sistema no disponible")
        st.stop()
    
    # Verificar si hay métricas
    summary = system.get_evaluation_summary()
    
    if 'error' in summary:
        st.warning("⚠️ No hay métricas disponibles. Ejecuta la evaluación primero.")
        st.stop()
    
    # Sidebar para seleccionar visualizaciones
    st.sidebar.markdown("### 📊 Seleccionar Visualizaciones")
    
    show_confusion_matrices = st.sidebar.checkbox("🎯 Matrices de Confusión", True)
    show_metrics_comparison = st.sidebar.checkbox("📊 Comparación de Métricas", True)
    show_roc_curves = st.sidebar.checkbox("📈 Curvas ROC", True)
    show_radar_chart = st.sidebar.checkbox("🎪 Gráfico de Radar", True)
    show_mcc_comparison = st.sidebar.checkbox("⚡ Comparación MCC", True)
    
    # Generar visualizaciones
    if st.button("📊 Generar Visualizaciones", type="primary"):
        with st.spinner("Generando visualizaciones..."):
            result = system.generate_visualizations(save_plots=True)
            
            if result.get('success'):
                st.success(f"✅ Visualizaciones generadas: {len(result.get('files_saved', []))} archivos")
            else:
                st.error(f"❌ Error: {result.get('error', 'Error desconocido')}")
    
    # Mostrar visualizaciones seleccionadas
    if system.visualizer:
        if show_confusion_matrices:
            st.markdown("### 🎯 Matrices de Confusión")
            
            # Selector de modelo para matriz individual
            models_available = list(system.metrics_cache.keys())
            selected_model = st.selectbox(
                "Selecciona un modelo para ver su matriz de confusión:",
                models_available
            )
            
            # Mostrar matriz seleccionada
            fig_cm = system.visualizer.plot_confusion_matrix(selected_model, save=False)
            if fig_cm:
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            
            # Mostrar comparación de todas las matrices
            st.markdown("#### 📊 Comparación de Todas las Matrices")
            fig_cm_comp = system.visualizer.plot_confusion_matrices_comparison(save=False)
            if fig_cm_comp:
                st.pyplot(fig_cm_comp)
                plt.close(fig_cm_comp)
        
        if show_metrics_comparison:
            st.markdown("### 📊 Comparación de Métricas")
            fig_metrics = system.visualizer.plot_metrics_comparison(save=False)
            if fig_metrics:
                st.pyplot(fig_metrics)
                plt.close(fig_metrics)
        
        if show_roc_curves:
            st.markdown("### 📈 Curvas ROC")
            fig_roc = system.visualizer.plot_roc_curves(save=False)
            if fig_roc:
                st.pyplot(fig_roc)
                plt.close(fig_roc)
        
        if show_radar_chart:
            st.markdown("### 🎪 Gráfico de Radar - Rendimiento")
            fig_radar = system.visualizer.plot_performance_radar(save=False)
            if fig_radar:
                st.pyplot(fig_radar)
                plt.close(fig_radar)
        
        if show_mcc_comparison:
            st.markdown("### ⚡ Comparación MCC")
            fig_mcc = system.visualizer.plot_mcc_comparison(save=False)
            if fig_mcc:
                st.pyplot(fig_mcc)
                plt.close(fig_mcc)
        
        # Tabla de métricas detalladas
        st.markdown("### 📋 Tabla de Métricas Detalladas")
        df_detailed = system.visualizer.create_metrics_summary_table()
        st.dataframe(df_detailed, use_container_width=True)
        
        # Insights automáticos
        st.markdown("### 💡 Insights Automáticos")
        insights = system.visualizer.generate_insights()
        for insight in insights:
            st.info(insight)
    
    else:
        st.error("❌ Visualizador no disponible")

elif page == "📄 Generar Reporte":
    st.markdown("## 📄 Generador de Reportes PDF")
    
    if not system:
        st.error("❌ Sistema no disponible")
        st.stop()
    
    # Verificar si hay métricas
    summary = system.get_evaluation_summary()
    
    if 'error' in summary:
        st.warning("⚠️ No hay métricas disponibles. Ejecuta la evaluación primero.")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuración del Reporte")
        
        # Título del reporte
        report_title = st.text_input(
            "Título del Reporte",
            value="Reporte de Evaluación - Sistema de Diagnóstico de Cáncer de Piel",
            help="Título que aparecerá en el reporte PDF"
        )
        
        # Nombre del archivo
        output_filename = st.text_input(
            "Nombre del Archivo",
            value="reporte_evaluacion.pdf",
            help="Nombre del archivo PDF generado"
        )
        
        # Opciones del reporte
        include_plots = st.checkbox(
            "🖼️ Incluir Visualizaciones",
            value=True,
            help="Incluir gráficos y visualizaciones en el reporte"
        )
        
        # Información del reporte
        st.markdown("### 📋 Contenido del Reporte")
        st.markdown(f"""
        **Modelos evaluados:** {summary['total_models']}
        **Muestras de test:** {summary['test_samples']:,}
        **Mejor modelo:** {summary['best_model']['name']}
        **Análisis estadístico:** Pruebas McNemar
        **Visualizaciones:** {6 if include_plots else 0}
        """)
    
    with col2:
        # Botón para generar reporte
        if st.button("📄 Generar Reporte PDF", type="primary"):
            with st.spinner("Generando reporte completo..."):
                try:
                    result = system.generate_complete_report(
                        title=report_title,
                        include_images=include_plots,
                        filename=output_filename
                    )
                    
                    if result.get('success'):
                        st.success("✅ Reporte generado exitosamente")
                        
                        # Información del archivo
                        st.markdown("### 📄 Información del Archivo")
                        st.info(f"📁 **Archivo:** {result['filename']}")
                        st.info(f"📍 **Ubicación:** {result['output_path']}")
                        st.info(f"🕐 **Generado:** {result['report_timestamp'][:19].replace('T', ' ')}")
                        
                        # Botón de descarga (si es posible)
                        if Path(result['output_path']).exists():
                            with open(result['output_path'], "rb") as pdf_file:
                                st.download_button(
                                    label="📥 Descargar Reporte PDF",
                                    data=pdf_file.read(),
                                    file_name=result['filename'],
                                    mime="application/pdf"
                                )
                    else:
                        st.error(f"❌ Error generando reporte: {result.get('error', 'Error desconocido')}")
                
                except Exception as e:
                    st.error(f"❌ Error durante la generación: {str(e)}")
        
        # Mostrar reportes anteriores
        st.markdown("### 📚 Reportes Anteriores")
        reports_dir = Path("app/reports")
        if reports_dir.exists():
            pdf_files = list(reports_dir.glob("*.pdf"))
            if pdf_files:
                for pdf_file in sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    file_info = pdf_file.stat()
                    size_mb = file_info.st_size / (1024 * 1024)
                    mod_time = time.ctime(file_info.st_mtime)
                    
                    st.info(f"📄 **{pdf_file.name}** ({size_mb:.1f} MB) - {mod_time}")
            else:
                st.info("📁 No hay reportes anteriores")
        else:
            st.info("📁 Directorio de reportes no encontrado")

# Footer
st.markdown("---")
st.markdown("### 🔧 Herramientas del Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Limpiar Cache", help="Limpia el cache del sistema"):
        if system:
            system.cleanup_cache()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache limpiado")
            st.rerun()

with col2:
    if st.button("📊 Estado del Sistema", help="Muestra información del sistema"):
        status = get_system_status()
        st.json(status)

with col3:
    if st.button("ℹ️ Información", help="Información del sistema"):
        st.info("Sistema de Diagnóstico de Cáncer de Piel v2.0")
        st.info("Desarrollado para evaluación automática de modelos de IA")
        st.info("Dataset: ISIC (International Skin Imaging Collaboration)")
