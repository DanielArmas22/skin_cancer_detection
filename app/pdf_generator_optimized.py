# pdf_generator_optimized.py
"""
Generador de reportes PDF optimizado para el sistema de diagnóstico de cáncer de piel
"""

import os
import time
import base64
import streamlit as st
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
from visualization import (
    plot_confusion_matrix, 
    create_metrics_dashboard,
    create_advanced_metrics_dashboard,
    create_model_comparison_plots,
    create_mcc_comparison_chart,
    create_mcnemar_plot,
    save_plot_to_image
)


def clean_text_for_pdf(text):
    """
    Limpia el texto para que sea compatible con FPDF
    
    Args:
        text (str): Texto a limpiar
    
    Returns:
        str: Texto limpio compatible con FPDF
    """
    if text is None:
        return ""
    
    # Reemplazar caracteres problemáticos para FPDF
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N',
        'ü': 'u', 'Ü': 'U',
        '°': 'o',
        '≤': '<=',
        '≥': '>=',
        '"': '"', '"': '"',
        ''': "'", ''': "'",
        '–': '-',
        '—': '-'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Reemplazar emojis con símbolos compatibles
    emoji_replacements = {
        '⚠': '!', '✅': 'V', '❌': 'X', '🚨': '!', '💡': '*',
        '📊': '=', '🎯': '>', '📄': '[]', '🖨️': 'P', 'ℹ️': 'i',
        '📈': '^', '📋': '[]', '🔍': 'o', '🤖': 'R', '📸': 'I',
        '🔧': 'T', '🌐': 'W', '📞': 'T', '🎉': '*'
    }
    
    # Reemplazar emojis primero
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)
    
    # Reemplazar caracteres especiales
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', 'ç': 'c', 'Ç': 'C',
        '•': '-', '–': '-', '—': '-',
        '°': 'o', '²': '2', '³': '3',
        '€': 'EUR', '£': 'GBP', '$': 'USD',
        '©': '(c)', '®': '(R)', '™': '(TM)'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remover otros caracteres no ASCII
    text = ''.join(char for char in text if ord(char) < 128)
    return text


def generate_pdf_report(image, diagnosis, confidence_percent, raw_confidence, model_name, 
                       model_info, comparison_results=None, confidence_threshold=0.75, 
                       metrics_data=None, plots_data=None, translations=None):
    """
    Genera un reporte PDF completo y visualmente atractivo para el diagnóstico de cáncer de piel.
    
    El reporte está dividido en dos secciones principales: diagnóstico y análisis estadístico.
    Incluye visualizaciones adaptadas para formato PDF de todos los gráficos que se muestran
    en el sistema.
    
    Args:
        image: Imagen PIL original
        diagnosis (str): Diagnóstico (Benigno/Maligno)
        confidence_percent (float): Porcentaje de confianza
        raw_confidence (float): Valor raw de confianza
        model_name (str): Nombre del modelo usado
        model_info (dict): Información del modelo
        comparison_results (list): Resultados de comparación entre modelos
        confidence_threshold (float): Umbral de confianza
        metrics_data (dict): Datos de métricas del modelo
        plots_data (dict): Datos de gráficos generados
        translations (dict): Diccionario de traducciones
    """
    # Si no se proporciona un diccionario de traducciones, usamos textos en español por defecto
    t = translations or {}
    
    try:
        # Crear PDF con orientación horizontal para mejor layout
        pdf = FPDF(orientation='L', format='A4')
        pdf.add_page()
        
        # Configurar márgenes más pequeños para aprovechar mejor el espacio
        pdf.set_margins(15, 15, 15)
        
        # Configurar fuente y colores
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(44, 62, 80)  # Azul oscuro
        
        # Título principal con diseño mejorado - traducido
        title = t.get('app_title', "SISTEMA DE DIAGNOSTICO DE CANCER DE PIEL")
        pdf.cell(0, 15, txt=clean_text_for_pdf(title), ln=1, align='C')
        pdf.ln(3)
        
        # Subtítulo - traducido
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(52, 73, 94)  # Gris azulado
        subtitle = t.get('pdf_report_title', "Reporte Medico Inteligente") 
        pdf.cell(0, 10, txt=clean_text_for_pdf(subtitle), ln=1, align='C')
        pdf.ln(3)
        
        # Línea separadora visual
        pdf.set_draw_color(52, 152, 219)  # Azul
        pdf.set_line_width(0.8)
        pdf.line(15, pdf.get_y(), pdf.w-15, pdf.get_y())
        pdf.ln(8)
        
        # Información básica del reporte
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(44, 62, 80)
        
        # Fecha y hora
        current_time = time.strftime("%d/%m/%Y %H:%M:%S")
        date_time_label = t.get('report_date_time', "Fecha y hora del analisis")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{date_time_label}: {current_time}"), ln=1)
        
        model_used_label = t.get('model_used', "Modelo utilizado")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{model_used_label}: {model_name}"), ln=1)
        
        threshold_label = t.get('threshold_value', "Umbral de confianza")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{threshold_label}: {confidence_threshold*100:.1f}%"), ln=1)
        pdf.ln(5)
        
        # Guardar imagen original
        img_path = "temp_img_pdf.png"
        image.save(img_path, "PNG")
        
        # Añadir imagen original al PDF
        pdf.set_font("Arial", 'B', 12)
        img_title = t.get('analyzed_image', "IMAGEN ANALIZADA")
        pdf.cell(0, 10, txt=clean_text_for_pdf(img_title), ln=1, align='C')
        pdf.ln(3)
        
        # Calcular posición para centrar la imagen
        img_width, img_height = 80, 60  # Tamaño en el PDF
        img_x = (pdf.w - img_width) / 2
        
        # Añadir imagen
        pdf.image(img_path, x=img_x, y=pdf.get_y(), w=img_width, h=img_height)
        pdf.ln(img_height + 10)
        
        # --- PRIMERA SECCIÓN: DIAGNÓSTICO PRINCIPAL ---
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(236, 240, 241)  # Gris claro
        diagnosis_section_title = t.get('diagnosis_results', "RESULTADOS DEL DIAGNOSTICO")
        pdf.cell(0, 12, txt=clean_text_for_pdf(diagnosis_section_title.upper()), ln=1, align='C', fill=True)
        pdf.ln(5)
        
        # Crear tabla de resultados principal
        pdf.set_font("Arial", 'B', 12)
        
        # Configurar colores según el diagnóstico
        malignant_term = t.get('malignant', "Maligno")
        if diagnosis == malignant_term:
            pdf.set_fill_color(231, 76, 60)  # Rojo para maligno
            pdf.set_text_color(255, 255, 255)  # Texto blanco
        else:
            pdf.set_fill_color(46, 125, 50)  # Verde para benigno
            pdf.set_text_color(255, 255, 255)  # Texto blanco
        
        # Celda de diagnóstico principal
        diagnosis_label = t.get('prediction', "DIAGNOSTICO")
        pdf.cell(0, 15, txt=clean_text_for_pdf(f"{diagnosis_label.upper()}: {diagnosis.upper()}"), ln=1, align='C', fill=True)
        pdf.ln(3)
        
        # Restaurar colores
        pdf.set_text_color(44, 62, 80)
        pdf.set_font("Arial", size=10)
        
        # Tabla de métricas detalladas - traducido
        confidence_label = t.get('confidence', "Confianza del diagnostico")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{confidence_label}: {confidence_percent:.1f}%"), ln=1)
        
        raw_value_label = t.get('raw_confidence_value', "Valor raw del modelo")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{raw_value_label}: {raw_confidence:.4f}"), ln=1)
        
        model_used_label = t.get('model_used', "Modelo utilizado")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{model_used_label}: {model_name}"), ln=1)
        
        parameters_label = t.get('parameters', "Parametros del modelo")
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"{parameters_label}: {model_info['parameters']:,}"), ln=1)
        pdf.ln(5)
        
        # Interpretación del resultado - traducido
        pdf.set_font("Arial", 'B', 12)
        interpretation_title = t.get('results_interpretation', "INTERPRETACION CLINICA")
        pdf.cell(0, 10, txt=clean_text_for_pdf(interpretation_title.upper()), ln=1, align='C')
        pdf.set_font("Arial", size=10)
        pdf.ln(3)
        
        benign_term = t.get('benign', "Benigno")
        
        if confidence_percent < (confidence_threshold * 100):
            # Confianza baja
            interpretation = t.get('low_confidence_warning', "CONFIANZA BAJA: Se recomienda consultar especialista")
            # Limpiar markdown
            interpretation = interpretation.replace('⚠️', '').replace('**', '')
        else:
            if diagnosis == benign_term:
                # Resultado benigno
                interpretation = t.get('favorable_result', "RESULTADO FAVORABLE: Lesion probablemente benigna")
                # Limpiar markdown
                interpretation = interpretation.replace('✅', '').replace('**', '')
            else:
                # Resultado maligno
                interpretation = t.get('attention_required', "ATENCION REQUERIDA: Consulta urgente con dermatologo")
                # Limpiar markdown
                interpretation = interpretation.replace('🚨', '').replace('**', '')
        
        pdf.cell(0, 8, txt=clean_text_for_pdf(interpretation), ln=1)
        pdf.ln(5)
        
        # --- SEGUNDA SECCIÓN: ANÁLISIS ESTADÍSTICO ---
        # Nueva página para el análisis estadístico
        if metrics_data:
            pdf.add_page()
            
            # Título de la sección de análisis estadístico (traducido)
            pdf.set_font("Arial", 'B', 16)
            statistical_analysis_title = t.get('statistical_analysis_title', "ANALISIS ESTADISTICO AVANZADO")
            pdf.cell(0, 12, txt=clean_text_for_pdf(statistical_analysis_title.upper()), ln=1, align='C')
            pdf.ln(5)
            
            # Descripción del análisis (traducido)
            pdf.set_font("Arial", size=10)
            statistical_analysis_desc = t.get('statistical_analysis_description', 
                                             "Incluyendo Coeficiente de Matthews y métricas de rendimiento")
            pdf.cell(0, 8, txt=clean_text_for_pdf(statistical_analysis_desc), ln=1, align='C')
            pdf.ln(5)
            
            # Subtítulo para la matriz de confusión (traducido)
            pdf.set_font("Arial", 'B', 12)
            confusion_matrix_title = t.get('confusion_matrix_title', "MATRIZ DE CONFUSION")
            pdf.cell(0, 10, txt=clean_text_for_pdf(confusion_matrix_title.upper()), ln=1, align='C')
            pdf.ln(3)
            
            # Mostrar matriz de confusión en forma de tabla - traducido
            cm = metrics_data['confusion_matrix']
            pdf.set_font("Arial", 'B', 10)
            confusion_matrix_chart = t.get('confusion_matrix_chart', "Matriz de Confusion")
            pdf.cell(0, 8, txt=clean_text_for_pdf(f"{confusion_matrix_chart}:"), ln=1)
            pdf.set_font("Arial", size=8)
            pdf.set_fill_color(236, 240, 241)
            
            # Calcular posición para centrar la tabla
            table_width = 160  # Ancho total de la tabla
            table_x = (pdf.w - table_width) / 2
            
            # Mover a la posición de la tabla
            pdf.set_x(table_x)
            
            # Headers - traducido
            pdf.cell(40, 8, txt=clean_text_for_pdf(""), border=1, fill=True)
            prediction_label = t.get('prediction', "Prediccion")
            pdf.cell(40, 8, txt=clean_text_for_pdf(prediction_label), border=1, fill=True)
            benign_label = t.get('benign', "Benigno")
            pdf.cell(40, 8, txt=clean_text_for_pdf(benign_label), border=1, fill=True)
            malignant_label = t.get('malignant', "Maligno")
            pdf.cell(40, 8, txt=clean_text_for_pdf(malignant_label), border=1, fill=True)
            pdf.ln()
            
            # Fila 1
            pdf.set_x(table_x)
            real_value_label = t.get('real_value', "Valor Real")
            pdf.cell(40, 8, txt=clean_text_for_pdf(real_value_label), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf(benign_label), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[0][0])), border=1, align='C')
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[0][1])), border=1, align='C')
            pdf.ln()
            
            # Fila 2
            pdf.set_x(table_x)
            pdf.cell(40, 8, txt=clean_text_for_pdf(""), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf(malignant_label), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[1][0])), border=1, align='C')
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[1][1])), border=1, align='C')
            pdf.ln(10)
            
            # Métricas avanzadas (traducido)
            pdf.set_font("Arial", 'B', 12)
            advanced_metrics_title = t.get('advanced_metrics', "METRICAS DE RENDIMIENTO AVANZADAS")
            pdf.cell(0, 8, txt=clean_text_for_pdf(advanced_metrics_title), ln=1)
            pdf.set_font("Arial", size=10)
            
            # Generar gráficos de visualización
            if plots_data is None:
                plots_data = {}
            
            # Generar gráfico de matriz de confusión
            cm_fig = plot_confusion_matrix(metrics_data['confusion_matrix'], model_name)
            cm_img_path = "temp_cm_plot.png"
            if cm_fig:
                save_plot_to_image(cm_fig, cm_img_path)
                plt.close(cm_fig)  # Cerrar figura para liberar memoria
                
                # Añadir el gráfico de matriz de confusión al PDF
                pdf.ln(3)
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('confusion_matrix_visual', "Visualización de Matriz de Confusión")), ln=1, align='C')
                pdf.image(cm_img_path, x=70, w=150, h=100)
                pdf.ln(5)
            
            # Añadir nueva página para el dashboard de métricas
            pdf.add_page()
            
            # Generar dashboard de métricas
            metrics_fig = create_metrics_dashboard(metrics_data, model_name)
            metrics_img_path = "temp_metrics_plot.png"
            if metrics_fig:
                save_plot_to_image(metrics_fig, metrics_img_path)
                plt.close(metrics_fig)  # Cerrar figura para liberar memoria
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('metrics_dashboard_title', "DASHBOARD DE MÉTRICAS DE RENDIMIENTO")), ln=1, align='C')
                pdf.image(metrics_img_path, x=15, w=260, h=170)
                pdf.ln(5)
            
            # Añadir nueva página para el dashboard avanzado
            pdf.add_page()
            
            # Generar dashboard avanzado de métricas
            advanced_fig = create_advanced_metrics_dashboard(metrics_data, model_name)
            advanced_img_path = "temp_advanced_plot.png"
            if advanced_fig:
                save_plot_to_image(advanced_fig, advanced_img_path)
                plt.close(advanced_fig)  # Cerrar figura para liberar memoria
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('advanced_dashboard_title', "ANÁLISIS ESTADÍSTICO AVANZADO")), ln=1, align='C')
                pdf.image(advanced_img_path, x=15, w=260, h=170)
                pdf.ln(5)
            
            # Etiquetas traducidas para las métricas
            accuracy_label = t.get('accuracy', "Accuracy")
            sensitivity_label = t.get('sensitivity', "Sensibilidad")
            specificity_label = t.get('specificity', "Especificidad")
            precision_label = t.get('precision', "Precision")
            f1_score_label = t.get('f1_score', "F1-Score")
            mcc_label = t.get('mcc', "MCC")
            
            # Crear gráficos de visualización
            if plots_data is None:
                plots_data = {}
            
            # Generar gráfico de matriz de confusión
            cm_fig = plot_confusion_matrix(metrics_data['confusion_matrix'], model_name)
            cm_img_path = "temp_cm_plot.png"
            if cm_fig:
                save_plot_to_image(cm_fig, cm_img_path)
                plt.close(cm_fig)  # Cerrar figura para liberar memoria
                
            # Añadir el gráfico de matriz de confusión al PDF - optimizado para PDF
            pdf.ln(3)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('confusion_matrix_visual', "Visualización de Matriz de Confusión")), ln=1, align='C')
            pdf.image(cm_img_path, x=70, w=150, h=100)
            pdf.ln(5)
                
            # Generar dashboard de métricas - optimizado para PDF
            metrics_fig = create_metrics_dashboard(metrics_data, model_name)
            metrics_img_path = "temp_metrics_plot.png"
            if metrics_fig:
                save_plot_to_image(metrics_fig, metrics_img_path)
                plt.close(metrics_fig)  # Cerrar figura para liberar memoria
                
                # Añadir nueva página para el dashboard de métricas
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('metrics_dashboard_title', "DASHBOARD DE MÉTRICAS DE RENDIMIENTO")), ln=1, align='C')
                # Optimizar tamaño para mejor visualización en PDF
                pdf.image(metrics_img_path, x=15, w=260, h=170)
                pdf.ln(5)
                
                # Añadir explicación de las métricas mostradas en el dashboard
                pdf.set_font("Arial", size=9)
                dashboard_explanation = t.get('dashboard_explanation', 
                    "El dashboard de métricas muestra visualmente el rendimiento del modelo mediante diferentes visualizaciones, incluyendo accuracy, precision, sensitivity y specificity.")
                pdf.multi_cell(0, 8, txt=clean_text_for_pdf(dashboard_explanation), align='L')            # Generar dashboard avanzado de métricas - optimizado para PDF
            advanced_fig = create_advanced_metrics_dashboard(metrics_data, model_name)
            advanced_img_path = "temp_advanced_plot.png"
            if advanced_fig:
                save_plot_to_image(advanced_fig, advanced_img_path)
                plt.close(advanced_fig)  # Cerrar figura para liberar memoria
                
                # Añadir nueva página para el dashboard avanzado
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('advanced_dashboard_title', "ANÁLISIS ESTADÍSTICO AVANZADO")), ln=1, align='C')
                # Optimizar tamaño para mejor visualización en PDF
                pdf.image(advanced_img_path, x=15, w=260, h=170)
                pdf.ln(5)
                
                # Añadir explicación del dashboard avanzado
                pdf.set_font("Arial", size=9)
                advanced_explanation = t.get('advanced_explanation', 
                    "El dashboard avanzado muestra métricas estadísticas detalladas, incluyendo el Coeficiente de Matthews (MCC), que es especialmente útil para evaluar modelos con clases desbalanceadas.")
                pdf.multi_cell(0, 8, txt=clean_text_for_pdf(advanced_explanation), align='L')
                pdf.ln(2)
            
            # Mostrar métricas con sus etiquetas traducidas - con formato mejorado para PDF
            pdf.set_font("Arial", 'B', 10)
            metrics_title = t.get('metrics_summary', "RESUMEN DE MÉTRICAS PRINCIPALES")
            pdf.cell(0, 8, txt=clean_text_for_pdf(metrics_title), ln=1)
            pdf.set_font("Arial", size=10)
            
            # Crear una tabla para las métricas
            col_width = 90
            pdf.set_fill_color(240, 240, 240)
            
            # Headers
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(t.get('metric_name', "Métrica")), border=1, fill=True)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(t.get('metric_value', "Valor")), border=1, fill=True)
            pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('metric_percentage', "Porcentaje")), border=1, fill=True)
            pdf.ln()
            
            # Filas de métricas
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(accuracy_label), border=1)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f"{metrics_data['accuracy']:.3f}"), border=1)
            pdf.cell(0, 8, txt=clean_text_for_pdf(f"{metrics_data['accuracy']*100:.1f}%"), border=1)
            pdf.ln()
            
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(sensitivity_label), border=1)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f"{metrics_data['sensitivity']:.3f}"), border=1)
            pdf.cell(0, 8, txt=clean_text_for_pdf(f"{metrics_data['sensitivity']*100:.1f}%"), border=1)
            pdf.ln()
            
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(specificity_label), border=1)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f"{metrics_data['specificity']:.3f}"), border=1)
            pdf.cell(0, 8, txt=clean_text_for_pdf(f"{metrics_data['specificity']*100:.1f}%"), border=1)
            pdf.ln()
            
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(precision_label), border=1)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f"{metrics_data['precision']:.3f}"), border=1)
            pdf.cell(0, 8, txt=clean_text_for_pdf(f"{metrics_data['precision']*100:.1f}%"), border=1)
            pdf.ln()
            
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f1_score_label), border=1)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f"{metrics_data['f1_score']:.3f}"), border=1)
            pdf.cell(0, 8, txt=clean_text_for_pdf(f"{metrics_data['f1_score']*100:.1f}%"), border=1)
            pdf.ln()
            
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(mcc_label), border=1)
            pdf.cell(col_width, 8, txt=clean_text_for_pdf(f"{metrics_data['mcc']:.3f}"), border=1)
            pdf.cell(0, 8, txt=clean_text_for_pdf("N/A"), border=1)
            pdf.ln()
            
            # Interpretación de métricas (traducido)
            pdf.ln(5)
            metrics_interpretation = t.get('metrics_interpretation', "Interpretación:")
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, txt=clean_text_for_pdf(metrics_interpretation), ln=1)
            pdf.set_font("Arial", size=9)
            
            # Interpretaciones traducidas
            accuracy_explanation = t.get('accuracy_explanation', "de las predicciones son correctas")
            sensitivity_explanation = t.get('sensitivity_explanation', "de los casos malignos son detectados")
            specificity_explanation = t.get('specificity_explanation', "de los casos benignos son correctamente identificados")
            
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"• {accuracy_label}: {metrics_data['accuracy']*100:.1f}% {accuracy_explanation}"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"• {sensitivity_label}: {metrics_data['sensitivity']*100:.1f}% {sensitivity_explanation}"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"• {specificity_label}: {metrics_data['specificity']*100:.1f}% {specificity_explanation}"), ln=1)
            pdf.ln(5)
            
            # Añadir mapas de activación si están disponibles en plots_data
            if plots_data and 'activation_maps' in plots_data and plots_data['activation_maps']:
                # Nueva página para los mapas de activación
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                activation_title = t.get('activation_maps_title', "MAPAS DE ACTIVACIÓN (GRAD-CAM)")
                pdf.cell(0, 10, txt=clean_text_for_pdf(activation_title), ln=1, align='C')
                
                # Explicación de los mapas de activación
                pdf.set_font("Arial", size=9)
                activation_explanation = t.get('activation_maps_explanation', 
                    "Los mapas de activación muestran las regiones de la imagen en las que el modelo se enfoca para hacer su diagnóstico. Las zonas en rojo son las áreas más relevantes para la decisión del modelo.")
                pdf.multi_cell(0, 8, txt=clean_text_for_pdf(activation_explanation), align='L')
                pdf.ln(5)
                
                # Añadir los mapas de activación como imágenes
                activation_img_path = plots_data['activation_maps']
                if os.path.exists(activation_img_path):
                    # Optimizar tamaño para mejor visualización en PDF
                    pdf.image(activation_img_path, x=50, w=180, h=140)
                    pdf.ln(5)
        
        # Comparación de modelos (si está disponible) - En la sección de análisis estadístico
        if comparison_results:
            # Siempre añadir nueva página para la comparación de modelos
            pdf.add_page()
            
            # Título de la sección de análisis estadístico (traducido)
            pdf.set_font("Arial", 'B', 16)
            statistical_analysis_title = t.get('statistical_analysis_title', "ANALISIS ESTADISTICO AVANZADO")
            pdf.cell(0, 12, txt=clean_text_for_pdf(statistical_analysis_title.upper()), ln=1, align='C')
            pdf.ln(5)
            
            # Generar DataFrame para la comparación
            import pandas as pd
            df_comparison = pd.DataFrame(comparison_results)
            
            # Generar gráficos de comparación
            conf_fig, time_fig = create_model_comparison_plots(df_comparison)
            conf_img_path = "temp_comparison_conf.png"
            time_img_path = "temp_comparison_time.png"
            
            if conf_fig and time_fig:
                save_plot_to_image(conf_fig, conf_img_path)
                save_plot_to_image(time_fig, time_img_path)
                plt.close(conf_fig)
                plt.close(time_fig)
            else:
                # Siempre añadir nueva página para la comparación de modelos
                pdf.add_page()
            
            # Generar DataFrame para la comparación
            import pandas as pd
            df_comparison = pd.DataFrame(comparison_results)
            
            # Generar gráficos de comparación
            conf_fig, time_fig = create_model_comparison_plots(df_comparison)
            conf_img_path = "temp_comparison_conf.png"
            time_img_path = "temp_comparison_time.png"
            
            if conf_fig and time_fig:
                save_plot_to_image(conf_fig, conf_img_path)
                save_plot_to_image(time_fig, time_img_path)
                plt.close(conf_fig)
                plt.close(time_fig)
                
                # Subtítulo para la comparación de modelos - traducido
            pdf.set_font("Arial", 'B', 12)
            model_comparison_title = t.get('model_comparison', "COMPARACION DETALLADA DE MODELOS")
            pdf.cell(0, 10, txt=clean_text_for_pdf(model_comparison_title.upper()), ln=1, align='C')
            pdf.ln(3)
            
            # Descripción de la comparación (traducido)
            pdf.set_font("Arial", size=9)
            model_comparison_desc = t.get('model_comparison_desc', "Resultados del análisis de la misma imagen con diferentes modelos")
            pdf.cell(0, 8, txt=clean_text_for_pdf(model_comparison_desc), ln=1, align='C')
            pdf.ln(5)
            
            pdf.set_font("Arial", size=8)
            pdf.set_fill_color(236, 240, 241)
            
            # Headers de comparación (traducido)
            modelo_label = "Modelo"  # Esto no suele traducirse
            diagnostico_label = t.get('prediction', "Diagnostico")
            confianza_label = t.get('confidence', "Confianza")
            tiempo_label = "Tiempo (ms)"  # Esto no suele traducirse
            valor_raw_label = t.get('raw_value', "Valor Raw")
            
            pdf.cell(50, 8, txt=clean_text_for_pdf(modelo_label), border=1, fill=True)
            pdf.cell(30, 8, txt=clean_text_for_pdf(diagnostico_label), border=1, fill=True)
            pdf.cell(25, 8, txt=clean_text_for_pdf(confianza_label), border=1, fill=True)
            pdf.cell(25, 8, txt=clean_text_for_pdf(tiempo_label), border=1, fill=True)
            pdf.cell(0, 8, txt=clean_text_for_pdf(valor_raw_label), border=1, fill=True)
            pdf.ln()
            
            for result in comparison_results:
                pdf.cell(50, 8, txt=clean_text_for_pdf(str(result['Modelo'])[:20]), border=1)
                pdf.cell(30, 8, txt=clean_text_for_pdf(str(result['Diagnostico'])), border=1)
                pdf.cell(25, 8, txt=clean_text_for_pdf(f"{result['Confianza (%)']}%"), border=1)
                pdf.cell(25, 8, txt=clean_text_for_pdf(f"{result['Tiempo (ms)']}ms"), border=1)
                pdf.cell(0, 8, txt=clean_text_for_pdf(f"{result['Valor Raw']}"), border=1)
                pdf.ln()
            
            # Análisis de consistencia (traducido)
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 11)
            consistency_analysis_title = t.get('consistency_analysis', "ANALISIS DE CONSISTENCIA")
            pdf.cell(0, 8, txt=clean_text_for_pdf(consistency_analysis_title), ln=1)
            pdf.set_font("Arial", size=9)
            
            # Verificar si todos los diagnósticos coinciden
            diagnoses = [result['Diagnostico'] for result in comparison_results]
            unique_diagnoses = list(set(diagnoses))
            is_consistent = len(unique_diagnoses) == 1
            
            if is_consistent:
                perfect_consistency = t.get('perfect_consistency', "Consistencia perfecta: Todos los modelos coinciden en el diagnóstico:")
                # Limpiar markdown
                perfect_consistency = perfect_consistency.replace('✅', '').replace('**', '')
                pdf.cell(0, 8, txt=clean_text_for_pdf(f"{perfect_consistency} {unique_diagnoses[0]}"), ln=1)
            else:
                inconsistency_detected = t.get('inconsistency_detected', "Inconsistencia detectada: Los modelos no coinciden en el diagnóstico")
                # Limpiar markdown
                inconsistency_detected = inconsistency_detected.replace('⚠️', '').replace('**', '')
                pdf.cell(0, 8, txt=clean_text_for_pdf(inconsistency_detected), ln=1)
                
                diagnoses_obtained = t.get('diagnoses_obtained', "Diagnósticos obtenidos:")
                # Limpiar markdown
                diagnoses_obtained = diagnoses_obtained.replace('**', '')
                pdf.cell(0, 8, txt=clean_text_for_pdf(f"{diagnoses_obtained}"), ln=1)
                
                # Contar ocurrencias de cada diagnóstico
                diagnosis_counts = {}
                for d in diagnoses:
                    if d in diagnosis_counts:
                        diagnosis_counts[d] += 1
                    else:
                        diagnosis_counts[d] = 1
                
                for d, count in diagnosis_counts.items():
                    pdf.cell(0, 6, txt=clean_text_for_pdf(f"- {d}: {count}/{len(diagnoses)} modelos"), ln=1)
            
            # Añadir visualizaciones de comparación
            pdf.ln(5)
            
            # Gráfico de comparación de confianza - optimizado para PDF
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('confidence_comparison_chart', "COMPARACIÓN DE CONFIANZA POR MODELO")), ln=1, align='C')
            if os.path.exists(conf_img_path):
                # Optimizar tamaño para mejor visualización en PDF
                pdf.image(conf_img_path, x=30, w=220, h=120)
                
                # Añadir explicación del gráfico de confianza
                pdf.ln(3)
                pdf.set_font("Arial", size=9)
                confidence_explanation = t.get('confidence_chart_explanation', 
                    "Este gráfico compara los niveles de confianza de diferentes modelos. Las barras verdes indican diagnóstico benigno y las rojas indican maligno.")
                pdf.multi_cell(0, 8, txt=clean_text_for_pdf(confidence_explanation), align='C')
                pdf.ln(5)
            
            # Nueva página para el gráfico de tiempo - optimizado para PDF
            pdf.add_page()
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('inference_time_chart', "TIEMPO DE INFERENCIA POR MODELO")), ln=1, align='C')
            if os.path.exists(time_img_path):
                # Optimizar tamaño para mejor visualización en PDF
                pdf.image(time_img_path, x=30, w=220, h=120)
                
                # Añadir explicación del gráfico de tiempo
                pdf.ln(3)
                pdf.set_font("Arial", size=9)
                time_explanation = t.get('time_chart_explanation', 
                    "Este gráfico muestra el tiempo de inferencia en milisegundos para cada modelo. Un tiempo más bajo indica mayor eficiencia computacional.")
                pdf.multi_cell(0, 8, txt=clean_text_for_pdf(time_explanation), align='C')
            
            # Generar gráfico MCC si hay al menos 2 modelos con métricas
            if plots_data and 'mcc_comparison' in plots_data and len(plots_data['mcc_comparison']) >= 2:
                mcc_fig = create_mcc_comparison_chart(plots_data['mcc_comparison'])
                mcc_img_path = "temp_mcc_chart.png"
                
                if mcc_fig:
                    save_plot_to_image(mcc_fig, mcc_img_path)
                    plt.close(mcc_fig)
                    
                    # Añadir nueva página para el gráfico MCC
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('mcc_comparison_chart', "COMPARACIÓN DE COEFICIENTE DE MATTHEWS (MCC)")), ln=1, align='C')
                    pdf.image(mcc_img_path, x=30, w=220, h=150)
            
            # Generar gráfico McNemar si está disponible en los plots_data
            if plots_data and 'mcnemar_results' in plots_data and len(plots_data['mcnemar_results']) > 0:
                mcnemar_fig = create_mcnemar_plot(plots_data['mcnemar_results'])
                mcnemar_img_path = "temp_mcnemar_plot.png"
                
                if mcnemar_fig:
                    save_plot_to_image(mcnemar_fig, mcnemar_img_path)
                    plt.close(mcnemar_fig)
                    
                    # Añadir nueva página para el gráfico McNemar
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(0, 8, txt=clean_text_for_pdf(t.get('mcnemar_test_chart', "PRUEBA ESTADÍSTICA DE MCNEMAR")), ln=1, align='C')
                    pdf.image(mcnemar_img_path, x=30, w=220, h=150)
                    
                    # Añadir explicación de la prueba de McNemar
                    pdf.ln(5)
                    pdf.set_font("Arial", size=9)
                    mcnemar_explanation = t.get('mcnemar_explanation', 
                        "La prueba de McNemar evalúa la diferencia estadísticamente significativa entre modelos. Un p-valor < 0.05 indica que las diferencias entre modelos no son aleatorias.")
                    pdf.multi_cell(0, 8, txt=clean_text_for_pdf(mcnemar_explanation), align='L')
            
            pdf.ln(5)
        
        # Información técnica al final de la primera sección (diagnóstico) - traducida
        pdf.set_font("Arial", 'B', 12)
        technical_info_title = t.get('technical_info', "INFORMACION TECNICA")
        pdf.cell(0, 10, txt=clean_text_for_pdf(technical_info_title), ln=1, align='C')
        pdf.set_font("Arial", size=10)
        
        dataset_info = t.get('technical_dataset', "Dataset: ISIC 2019 (25,331 imagenes reales)")
        pdf.cell(0, 8, txt=clean_text_for_pdf(dataset_info), ln=1)
        
        type_info = t.get('technical_type', "Tipo: Clasificacion Binaria (Benigno/Maligno)")
        pdf.cell(0, 8, txt=clean_text_for_pdf(type_info), ln=1)
        
        accuracy_info = t.get('technical_accuracy', "Precision: ~69% (optimizado para cancer de piel)")
        pdf.cell(0, 8, txt=clean_text_for_pdf(accuracy_info), ln=1)
        
        input_info = t.get('technical_input', "Entrada: 300x300 pixeles")
        pdf.cell(0, 8, txt=clean_text_for_pdf(input_info), ln=1)
        
        architecture_info = t.get('technical_architecture', "Arquitectura: Transfer Learning con fine-tuning")
        pdf.cell(0, 8, txt=clean_text_for_pdf(architecture_info), ln=1)
        pdf.ln(5)
        
        # Advertencia médica con diseño destacado al final del documento - traducida
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(231, 76, 60)  # Rojo
        pdf.set_text_color(255, 255, 255)  # Blanco
        disclaimer_title = t.get('medical_disclaimer_title', "DESCARGO DE RESPONSABILIDAD MEDICA")
        pdf.cell(0, 10, txt=clean_text_for_pdf(disclaimer_title), ln=1, align='C', fill=True)
        pdf.ln(3)
        
        pdf.set_text_color(44, 62, 80)  # Volver a color normal
        pdf.set_font("Arial", size=10)
        
        disclaimer_1 = t.get('medical_disclaimer_1', "Este sistema es para fines educativos y de investigacion.")
        pdf.cell(0, 8, txt=clean_text_for_pdf(disclaimer_1), ln=1)
        
        disclaimer_2 = t.get('medical_disclaimer_2', "Los resultados NO constituyen diagnostico medico.")
        pdf.cell(0, 8, txt=clean_text_for_pdf(disclaimer_2), ln=1)
        
        disclaimer_3 = t.get('medical_disclaimer_3', "SIEMPRE consulte con un dermatologo para diagnostico profesional.")
        pdf.cell(0, 8, txt=clean_text_for_pdf(disclaimer_3), ln=1)
        pdf.ln(5)
        
        # Guardar PDF
        pdf_filename = f"diagnostico_cancer_piel_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_filename)
        
        # Proporcionar descarga
        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
        
        b64 = base64.b64encode(pdf_bytes).decode()
        download_text = t.get('download_pdf', 'Descargar Reporte PDF')
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_filename}">📄 {download_text}</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Limpiar archivos temporales
        temp_files = [
            img_path, 
            pdf_filename,
            "temp_cm_plot.png",
            "temp_metrics_plot.png", 
            "temp_advanced_plot.png",
            "temp_comparison_conf.png",
            "temp_comparison_time.png",
            "temp_mcc_chart.png",
            "temp_mcnemar_plot.png"
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        st.success("✅ " + t.get('pdf_success', 'Reporte PDF generado exitosamente'))
        
    except Exception as e:
        st.error(f"❌ Error al generar el reporte PDF: {str(e)}")
        # Limpiar archivos temporales en caso de error
        temp_files = [
            "temp_img_pdf.png",
            "temp_cm_plot.png",
            "temp_metrics_plot.png", 
            "temp_advanced_plot.png",
            "temp_comparison_conf.png",
            "temp_comparison_time.png",
            "temp_mcc_chart.png",
            "temp_mcnemar_plot.png"
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
