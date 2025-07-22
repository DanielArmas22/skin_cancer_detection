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


def clean_text_for_pdf(text):
    """
    Limpia el texto para que sea compatible con FPDF
    
    Args:
        text (str): Texto a limpiar
    
    Returns:
        str: Texto limpio compatible con FPDF
    """
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
    Genera un reporte PDF completo y visualmente atractivo para el diagnóstico de cáncer de piel
    
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
        
        # Título principal con diseño mejorado
        pdf.cell(0, 15, txt=clean_text_for_pdf("SISTEMA DE DIAGNOSTICO DE CANCER DE PIEL"), ln=1, align='C')
        pdf.ln(3)
        
        # Subtítulo
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(52, 73, 94)  # Gris azulado
        pdf.cell(0, 10, txt=clean_text_for_pdf("Reporte Medico Inteligente"), ln=1, align='C')
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
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Fecha y hora del analisis: {current_time}"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Modelo utilizado: {model_name}"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Umbral de confianza: {confidence_threshold*100:.1f}%"), ln=1)
        pdf.ln(5)
        
        # Guardar imagen original
        img_path = "temp_img_pdf.png"
        image.save(img_path, "PNG")
        
        # Añadir imagen original al PDF
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=clean_text_for_pdf("IMAGEN ANALIZADA"), ln=1, align='C')
        pdf.ln(3)
        
        # Calcular posición para centrar la imagen
        img_width, img_height = 80, 60  # Tamaño en el PDF
        img_x = (pdf.w - img_width) / 2
        
        # Añadir imagen
        pdf.image(img_path, x=img_x, y=pdf.get_y(), w=img_width, h=img_height)
        pdf.ln(img_height + 10)
        
        # Sección de resultados del diagnóstico
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(236, 240, 241)  # Gris claro
        pdf.cell(0, 12, txt=clean_text_for_pdf("RESULTADOS DEL DIAGNOSTICO"), ln=1, align='C', fill=True)
        pdf.ln(5)
        
        # Crear tabla de resultados principal
        pdf.set_font("Arial", 'B', 12)
        
        # Configurar colores según el diagnóstico
        if diagnosis == "Maligno":
            pdf.set_fill_color(231, 76, 60)  # Rojo para maligno
            pdf.set_text_color(255, 255, 255)  # Texto blanco
        else:
            pdf.set_fill_color(46, 125, 50)  # Verde para benigno
            pdf.set_text_color(255, 255, 255)  # Texto blanco
        
        # Celda de diagnóstico principal
        pdf.cell(0, 15, txt=clean_text_for_pdf(f"DIAGNOSTICO: {diagnosis.upper()}"), ln=1, align='C', fill=True)
        pdf.ln(3)
        
        # Restaurar colores
        pdf.set_text_color(44, 62, 80)
        pdf.set_font("Arial", size=10)
        
        # Tabla de métricas detalladas
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Confianza del diagnostico: {confidence_percent:.1f}%"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Valor raw del modelo: {raw_confidence:.4f}"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Modelo utilizado: {model_name}"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf(f"Parametros del modelo: {model_info['parameters']:,}"), ln=1)
        pdf.ln(5)
        
        # Interpretación del resultado
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=clean_text_for_pdf("INTERPRETACION CLINICA"), ln=1, align='C')
        pdf.set_font("Arial", size=10)
        pdf.ln(3)
        
        if confidence_percent < (confidence_threshold * 100):
            interpretation = "CONFIANZA BAJA: Se recomienda consultar especialista"
        else:
            if diagnosis == "Benigno":
                interpretation = "RESULTADO FAVORABLE: Lesion probablemente benigna"
            else:
                interpretation = "ATENCION REQUERIDA: Consulta urgente con dermatologo"
        
        pdf.cell(0, 8, txt=clean_text_for_pdf(interpretation), ln=1)
        pdf.ln(5)
        
        # Sección de métricas del modelo (si están disponibles)
        if metrics_data:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, txt=clean_text_for_pdf("METRICAS DEL MODELO"), ln=1, align='C')
            pdf.ln(3)
            
            # Mostrar matriz de confusión en forma de tabla
            cm = metrics_data['confusion_matrix']
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, txt=clean_text_for_pdf("MATRIZ DE CONFUSION:"), ln=1)
            pdf.set_font("Arial", size=8)
            pdf.set_fill_color(236, 240, 241)
            
            # Calcular posición para centrar la tabla
            table_width = 160  # Ancho total de la tabla
            table_x = (pdf.w - table_width) / 2
            
            # Mover a la posición de la tabla
            pdf.set_x(table_x)
            
            # Headers
            pdf.cell(40, 8, txt=clean_text_for_pdf(""), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf("Prediccion"), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf("Benigno"), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf("Maligno"), border=1, fill=True)
            pdf.ln()
            
            # Fila 1
            pdf.set_x(table_x)
            pdf.cell(40, 8, txt=clean_text_for_pdf("Valor Real"), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf("Benigno"), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[0][0])), border=1, align='C')
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[0][1])), border=1, align='C')
            pdf.ln()
            
            # Fila 2
            pdf.set_x(table_x)
            pdf.cell(40, 8, txt=clean_text_for_pdf(""), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf("Maligno"), border=1, fill=True)
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[1][0])), border=1, align='C')
            pdf.cell(40, 8, txt=clean_text_for_pdf(str(cm[1][1])), border=1, align='C')
            pdf.ln(10)
            
            # Métricas avanzadas
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, txt=clean_text_for_pdf("METRICAS DE RENDIMIENTO:"), ln=1)
            pdf.set_font("Arial", size=9)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"Accuracy: {metrics_data['accuracy']:.3f} ({metrics_data['accuracy']*100:.1f}%)"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"Sensitivity: {metrics_data['sensitivity']:.3f} ({metrics_data['sensitivity']*100:.1f}%)"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"Specificity: {metrics_data['specificity']:.3f} ({metrics_data['specificity']*100:.1f}%)"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"Precision: {metrics_data['precision']:.3f} ({metrics_data['precision']*100:.1f}%)"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"F1-Score: {metrics_data['f1_score']:.3f} ({metrics_data['f1_score']*100:.1f}%)"), ln=1)
            pdf.cell(0, 6, txt=clean_text_for_pdf(f"MCC: {metrics_data['mcc']:.3f}"), ln=1)
            pdf.ln(5)
        
        # Comparación de modelos (si está disponible) - Nueva página
        if comparison_results:
            # Nueva página para comparación
            pdf.add_page()
            
            # Título de la nueva página
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 12, txt=clean_text_for_pdf("COMPARACION DETALLADA DE MODELOS"), ln=1, align='C')
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, txt=clean_text_for_pdf("COMPARACION DE MODELOS"), ln=1, align='C')
            pdf.ln(3)
            
            pdf.set_font("Arial", size=8)
            pdf.set_fill_color(236, 240, 241)
            
            # Headers de comparación
            pdf.cell(50, 8, txt=clean_text_for_pdf("Modelo"), border=1, fill=True)
            pdf.cell(30, 8, txt=clean_text_for_pdf("Diagnostico"), border=1, fill=True)
            pdf.cell(25, 8, txt=clean_text_for_pdf("Confianza"), border=1, fill=True)
            pdf.cell(25, 8, txt=clean_text_for_pdf("Tiempo"), border=1, fill=True)
            pdf.cell(0, 8, txt=clean_text_for_pdf("Valor Raw"), border=1, fill=True)
            pdf.ln()
            
            for result in comparison_results:
                pdf.cell(50, 8, txt=clean_text_for_pdf(str(result['Modelo'])[:20]), border=1)
                pdf.cell(30, 8, txt=clean_text_for_pdf(str(result['Diagnostico'])), border=1)
                pdf.cell(25, 8, txt=clean_text_for_pdf(f"{result['Confianza (%)']}%"), border=1)
                pdf.cell(25, 8, txt=clean_text_for_pdf(f"{result['Tiempo (ms)']}ms"), border=1)
                pdf.cell(0, 8, txt=clean_text_for_pdf(f"{result['Valor Raw']}"), border=1)
                pdf.ln()
            
            pdf.ln(5)
        
        # Información técnica
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=clean_text_for_pdf("INFORMACION TECNICA"), ln=1, align='C')
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Dataset: ISIC 2019 (25,331 imagenes reales)"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Tipo: Clasificacion Binaria (Benigno/Maligno)"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Precision: ~69% (optimizado para cancer de piel)"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Entrada: 300x300 pixeles"), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Arquitectura: Transfer Learning con fine-tuning"), ln=1)
        pdf.ln(5)
        
        # Advertencia médica con diseño destacado
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(231, 76, 60)  # Rojo
        pdf.set_text_color(255, 255, 255)  # Blanco
        pdf.cell(0, 10, txt=clean_text_for_pdf("DESCARGO DE RESPONSABILIDAD MEDICA"), ln=1, align='C', fill=True)
        pdf.ln(3)
        
        pdf.set_text_color(44, 62, 80)  # Volver a color normal
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Este sistema es para fines educativos y de investigacion."), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf("Los resultados NO constituyen diagnostico medico."), ln=1)
        pdf.cell(0, 8, txt=clean_text_for_pdf("SIEMPRE consulte con un dermatologo para diagnostico profesional."), ln=1)
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
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        
        st.success("✅ " + t.get('pdf_success', 'Reporte PDF generado exitosamente'))
        
    except Exception as e:
        st.error(f"❌ Error al generar el reporte PDF: {str(e)}")
        # Limpiar archivos temporales en caso de error
        for temp_file in ["temp_img_pdf.png"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
