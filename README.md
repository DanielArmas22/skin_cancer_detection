# Sistema de Diagnóstico de Cáncer de Piel v2.0

## Descripción General

Sistema completo para el diagnóstico automático de cáncer de piel basado en análisis de imágenes utilizando modelos de aprendizaje automático. La versión 2.0 incluye una arquitectura completamente reestructurada, eliminación de datos hardcodeados y un sistema de gestión centralizado.

## Características Principales

### ✅ Funcionalidades Implementadas

1. **Evaluación Automática de Modelos**
   - Evaluación con dataset real ISIC
   - Cálculo de métricas completas (Accuracy, Precision, Recall, F1-Score, MCC, AUC-ROC)
   - Comparaciones estadísticas usando pruebas McNemar
   - Cache persistente de métricas

2. **Visualizaciones Avanzadas**
   - Matrices de confusión individuales y comparativas
   - Gráficos de comparación de métricas
   - Curvas ROC para todos los modelos
   - Gráficos de radar de rendimiento
   - Comparación específica de MCC

3. **Generación de Reportes PDF**
   - Reportes completos con análisis estadístico
   - Inclusión de visualizaciones
   - Metodología detallada
   - Conclusiones y recomendaciones

4. **Diagnóstico de Imágenes**
   - Selección automática del mejor modelo
   - Predicción en tiempo real
   - Interpretación de resultados
   - Métricas del modelo utilizado

5. **Sistema de Gestión Centralizado**
   - Configuración automática
   - Manejo de errores robusto
   - Logging detallado
   - Cache inteligente

## Estructura del Proyecto

```
skin_cancer_detection/
├── app/
│   ├── system_manager.py          # Gestor principal del sistema
│   ├── app.py                     # Interfaz Streamlit actualizada
│   ├── model_evaluator.py         # Evaluador de modelos
│   ├── metrics_visualizer.py      # Generador de visualizaciones
│   ├── pdf_generator.py           # Generador de reportes PDF
│   ├── model_utils.py             # Utilidades para modelos
│   ├── preprocessing.py           # Preprocesamiento de imágenes
│   ├── cache/                     # Cache de métricas y configuraciones
│   │   ├── model_metrics.json     # Métricas de evaluación
│   │   ├── model_comparisons.json # Comparaciones estadísticas
│   │   └── system_config.json     # Configuración del sistema
│   ├── models/                    # Modelos entrenados
│   │   ├── efficientnetb4.h5      # Modelo EfficientNetB4
│   │   ├── resnet152.h5           # Modelo ResNet152
│   │   └── cnn_personalizada.h5   # CNN personalizada
│   ├── plots/                     # Visualizaciones generadas
│   └── reports/                   # Reportes PDF generados
├── data/
│   ├── ISIC_dataset/             # Dataset de entrenamiento
│   └── ISIC_dataset_test/        # Dataset de test
├── requirements.txt               # Dependencias
├── Dockerfile                    # Configuración Docker
├── docker-compose.yml            # Orquestación Docker
└── README.md                     # Documentación
```

## Componentes Principales

### 1. Sistema Manager (system_manager.py)
Gestor central que coordina todos los componentes del sistema:

- **Inicialización automática** de todos los componentes
- **Gestión de configuraciones** persistentes
- **Manejo de errores** robusto
- **Cache inteligente** para métricas y resultados
- **Logging detallado** para debugging

### 2. Evaluador de Modelos (model_evaluator.py)
Componente especializado en la evaluación de modelos:

- **Evaluación automática** con dataset de test
- **Métricas completas** para análisis médico
- **Comparaciones estadísticas** entre modelos
- **Persistencia de resultados** en JSON
- **Detección automática** de modelos disponibles

### 3. Visualizador de Métricas (metrics_visualizer.py)
Generador de visualizaciones avanzadas:

- **Matrices de confusión** individuales y comparativas
- **Gráficos de barras** para comparar métricas
- **Curvas ROC** para análisis de rendimiento
- **Gráficos de radar** multidimensionales
- **Insights automáticos** basados en métricas

### 4. Generador de PDF (pdf_generator.py)
Creador de reportes profesionales:

- **Reportes completos** con análisis detallado
- **Inclusión de gráficos** y visualizaciones
- **Metodología científica** documentada
- **Conclusiones automáticas** basadas en datos
- **Formato profesional** para presentaciones

## Flujo de Trabajo

### 1. Inicialización del Sistema
```python
# El sistema se inicializa automáticamente
system = SkinCancerDiagnosisSystem()
system.initialize_components()
```

### 2. Evaluación de Modelos
```python
# Evaluación automática con dataset de test
metrics = system.evaluate_models(force_refresh=False)
```

### 3. Generación de Visualizaciones
```python
# Crear todas las visualizaciones
result = system.generate_visualizations(save_plots=True)
```

### 4. Predicción de Imágenes
```python
# Diagnóstico automático con mejor modelo
result = system.predict_single_image(image, model_name=None)
```

### 5. Generación de Reportes
```python
# Reporte completo en PDF
report = system.generate_complete_report(
    title="Reporte de Evaluación",
    include_images=True
)
```

## Métricas Evaluadas

### Métricas Principales
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Precisión para casos malignos
- **Recall (Sensitivity)**: Sensibilidad para detectar casos malignos
- **F1-Score**: Media armónica entre precision y recall
- **MCC**: Matthews Correlation Coefficient (métrica balanceada)
- **AUC-ROC**: Área bajo la curva ROC
- **Specificity**: Especificidad para casos benignos

### Análisis Estadístico
- **Pruebas McNemar**: Comparación entre pares de modelos
- **Intervalos de confianza**: Para métricas principales
- **Significancia estadística**: p < 0.05
- **Análisis de matriz de confusión**: Detalle de errores

## Interfaz de Usuario

### Páginas Principales

1. **🏠 Inicio**: Estado del sistema y información general
2. **🔍 Diagnóstico**: Análisis de imágenes individuales
3. **📊 Evaluación**: Evaluación completa de modelos
4. **📈 Visualizaciones**: Gráficos y análisis visual
5. **📄 Reportes**: Generación de documentos PDF

### Funcionalidades Interactivas

- **Selección automática** del mejor modelo
- **Visualizaciones dinámicas** generadas en tiempo real
- **Descarga de reportes** PDF completos
- **Cache inteligente** para rendimiento óptimo
- **Herramientas de sistema** para mantenimiento

## Configuración y Uso

### Requisitos del Sistema
- Python 3.8+
- TensorFlow 2.18+
- Streamlit 1.32+
- Dataset ISIC en `data/ISIC_dataset_test/`
- Modelos entrenados en `app/models/`

### Instalación
```bash
pip install -r requirements.txt
```

### Ejecución
```bash
cd app
streamlit run app.py
```

### Configuración Docker
```bash
docker-compose up -d
```

## Mejoras Implementadas en v2.0

### 🔧 Arquitectura
- **Sistema centralizado** con gestor principal
- **Eliminación de datos hardcodeados** en el código
- **Configuración dinámica** basada en archivos
- **Manejo de errores** robusto y logging

### 📊 Evaluación
- **Detección automática** de modelos disponibles
- **Métricas más completas** para análisis médico
- **Comparaciones estadísticas** rigurosas
- **Cache inteligente** para rendimiento

### 🎨 Interfaz
- **Diseño más limpio** y profesional
- **Navegación mejorada** entre secciones
- **Información dinámica** del sistema
- **Herramientas de diagnóstico** integradas

### 📈 Visualizaciones
- **Gráficos más informativos** y profesionales
- **Insights automáticos** basados en datos
- **Exportación mejorada** de visualizaciones
- **Tablas dinámicas** con métricas detalladas

### 📄 Reportes
- **Formato más profesional** y completo
- **Análisis automático** de resultados
- **Metodología documentada** científicamente
- **Conclusiones basadas en datos** reales

## Consideraciones Médicas

### ⚠️ Disclaimer Importante
Este sistema está diseñado exclusivamente para:
- **Fines educativos** y de investigación
- **Prototipado** de sistemas de IA médica
- **Análisis comparativo** de modelos
- **Demostración** de técnicas de ML

### 🚫 NO debe utilizarse para:
- **Diagnóstico médico real** en pacientes
- **Toma de decisiones clínicas** sin supervisión
- **Reemplazo** de profesionales médicos
- **Uso comercial** sin validación clínica

### 📋 Recomendaciones
- Siempre consultar con **profesionales médicos** cualificados
- Utilizar como **herramienta de apoyo** únicamente
- Validar resultados con **métodos clínicos** establecidos
- Mantener **supervisión médica** constante

## Limitaciones Conocidas

### 🔍 Técnicas
- **Dataset limitado** al conjunto ISIC disponible
- **Variabilidad** en condiciones de captura de imágenes
- **Posible sesgo** en selección de muestras
- **Necesidad de validación** en datasets externos

### 🏥 Clínicas
- **No validado clínicamente** en entornos reales
- **Falta de aprobación** regulatoria
- **Variabilidad** en diferentes poblaciones
- **Necesidad de estudios** prospectivos

## Trabajo Futuro

### 🔬 Investigación
- **Validación cruzada** con múltiples datasets
- **Análisis de interpretabilidad** de modelos
- **Optimización** de hiperparámetros
- **Técnicas de ensemble** para mejorar rendimiento

### 🏥 Aplicación
- **Validación clínica** en entornos reales
- **Integración** con sistemas hospitalarios
- **Interfaz** para profesionales médicos
- **Cumplimiento** de regulaciones médicas

### 🔧 Técnico
- **Optimización** de rendimiento
- **Escalabilidad** para grandes volúmenes
- **Integración** con APIs médicas
- **Monitoreo** en tiempo real

## Contacto y Soporte

Este sistema ha sido desarrollado como proyecto educativo para demostrar las capacidades de la inteligencia artificial en el análisis médico. Para cualquier consulta técnica o académica, por favor contactar con el equipo de desarrollo.

---

**Versión:** 2.0.0  
**Fecha:** Julio 2025  
**Licencia:** Educativa/Investigación  
**Estado:** Prototipo funcional
