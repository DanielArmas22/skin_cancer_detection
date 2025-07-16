from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

class PDFReportGenerator:
    """
    Generador de reportes PDF para evaluación de modelos de cáncer de piel
    """
    
    def __init__(self, output_dir="reports"):
        """
        Inicializa el generador de reportes
        
        Args:
            output_dir: Directorio para guardar reportes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar PDF
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)
        
    def add_title_page(self, title="Reporte de Evaluación de Modelos"):
        """
        Añade página de título
        
        Args:
            title: Título del reporte
        """
        self.pdf.set_font("Arial", style="B", size=20)
        self.pdf.cell(0, 20, title, ln=True, align="C")
        
        self.pdf.ln(10)
        
        # Información del reporte
        self.pdf.set_font("Arial", size=14)
        self.pdf.cell(0, 10, "Sistema de Diagnóstico de Cáncer de Piel", ln=True, align="C")
        
        self.pdf.ln(10)
        
        self.pdf.set_font("Arial", size=12)
        self.pdf.cell(0, 10, f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True, align="C")
        
        self.pdf.ln(20)
        
        # Disclaimer
        self.pdf.set_font("Arial", style="B", size=14)
        self.pdf.cell(0, 10, "⚠️ DISCLAIMER MÉDICO", ln=True, align="C")
        
        self.pdf.ln(5)
        
        disclaimer_text = """
        Este sistema es solo para fines educativos y de investigación.
        NO debe utilizarse para diagnóstico médico real.
        Siempre consulte a un profesional médico calificado.
        """
        
        self.pdf.set_font("Arial", size=10)
        lines = disclaimer_text.strip().split('\\n')
        for line in lines:
            self.pdf.cell(0, 6, line.strip(), ln=True, align="C")
    
    def add_section_title(self, title):
        """
        Añade título de sección
        
        Args:
            title: Título de la sección
        """
        self.pdf.ln(10)
        self.pdf.set_font("Arial", style="B", size=14)
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.ln(5)
    
    def add_text(self, text, font_size=12):
        """
        Añade texto normal
        
        Args:
            text: Texto a añadir
            font_size: Tamaño de fuente
        """
        self.pdf.set_font("Arial", size=font_size)
        
        # Dividir texto en líneas que quepan en la página
        lines = text.split('\\n')
        for line in lines:
            # Manejar líneas largas
            while len(line) > 80:
                break_point = 80
                while break_point > 0 and line[break_point] != ' ':
                    break_point -= 1
                if break_point == 0:
                    break_point = 80
                
                self.pdf.cell(0, 6, line[:break_point], ln=True)
                line = line[break_point:].lstrip()
            
            if line:
                self.pdf.cell(0, 6, line, ln=True)
        
        self.pdf.ln(5)
    
    def add_metrics_table(self, metrics_data):
        """
        Añade tabla de métricas
        
        Args:
            metrics_data: Datos de métricas de los modelos
        """
        self.add_section_title("📊 Resumen de Métricas")
        
        # Encabezados
        self.pdf.set_font("Arial", style="B", size=10)
        
        # Calcular anchos de columna
        col_widths = [40, 20, 20, 20, 20, 20, 20]
        headers = ["Modelo", "Accuracy", "Precision", "Recall", "F1-Score", "MCC", "AUC-ROC"]
        
        # Añadir encabezados
        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 8, header, border=1, align="C")
        self.pdf.ln()
        
        # Añadir datos
        self.pdf.set_font("Arial", size=9)
        
        # Ordenar por MCC
        sorted_models = sorted(metrics_data.items(), key=lambda x: x[1]['mcc'], reverse=True)
        
        for model_name, data in sorted_models:
            values = [
                model_name[:15],  # Truncar nombre si es muy largo
                f"{data['accuracy']:.3f}",
                f"{data['precision']:.3f}",
                f"{data['recall']:.3f}",
                f"{data['f1_score']:.3f}",
                f"{data['mcc']:.3f}",
                f"{data['auc_roc']:.3f}"
            ]
            
            for i, value in enumerate(values):
                self.pdf.cell(col_widths[i], 8, str(value), border=1, align="C")
            self.pdf.ln()
        
        self.pdf.ln(10)
    
    def add_best_model_analysis(self, metrics_data):
        """
        Añade análisis del mejor modelo
        
        Args:
            metrics_data: Datos de métricas
        """
        # Encontrar mejor modelo
        best_model = max(metrics_data.items(), key=lambda x: x[1]['mcc'])
        model_name, model_data = best_model
        
        self.add_section_title(f"🏆 Análisis del Mejor Modelo: {model_name}")
        
        analysis_text = f"""
        Modelo con mejor rendimiento basado en Matthews Correlation Coefficient (MCC).
        
        Métricas principales:
        • Accuracy: {model_data['accuracy']:.3f} ({model_data['accuracy']*100:.1f}%)
        • Precision: {model_data['precision']:.3f}
        • Recall (Sensibilidad): {model_data['recall']:.3f}
        • F1-Score: {model_data['f1_score']:.3f}
        • MCC: {model_data['mcc']:.3f}
        • AUC-ROC: {model_data['auc_roc']:.3f}
        • Especificidad: {model_data['specificity']:.3f}
        
        Interpretación:
        • Muestras evaluadas: {model_data['test_samples']:,}
        • Casos benignos: {model_data['class_distribution'][0]:,}
        • Casos malignos: {model_data['class_distribution'][1]:,}
        """
        
        self.add_text(analysis_text)
        
        # Análisis de matriz de confusión
        cm = model_data['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        confusion_text = f"""
        Matriz de Confusión:
        • Verdaderos Negativos (TN): {tn} - Casos benignos correctamente identificados
        • Falsos Positivos (FP): {fp} - Casos benignos incorrectamente clasificados como malignos
        • Falsos Negativos (FN): {fn} - Casos malignos incorrectamente clasificados como benignos
        • Verdaderos Positivos (TP): {tp} - Casos malignos correctamente identificados
        
        Implicaciones clínicas:
        • Sensibilidad alta: Buena detección de casos malignos
        • Especificidad alta: Pocas falsas alarmas
        • MCC alto: Rendimiento balanceado en ambas clases
        """
        
        self.add_text(confusion_text)
    
    def add_comparison_analysis(self, comparison_data):
        """
        Añade análisis de comparaciones estadísticas
        
        Args:
            comparison_data: Datos de comparaciones entre modelos
        """
        if not comparison_data:
            return
        
        self.add_section_title("📊 Comparación Estadística entre Modelos")
        
        self.add_text("Análisis utilizando la prueba de McNemar para comparar el rendimiento entre pares de modelos:")
        
        significant_comparisons = []
        non_significant_comparisons = []
        
        for comparison_key, data in comparison_data.items():
            if data and data.get('significant', False):
                significant_comparisons.append((comparison_key, data))
            elif data:
                non_significant_comparisons.append((comparison_key, data))
        
        # Comparaciones significativas
        if significant_comparisons:
            self.add_text("\\n🔍 Diferencias Estadísticamente Significativas (p < 0.05):")
            
            for comparison_key, data in significant_comparisons:
                better_model = data['model1'] if data['model1_better'] else data['model2']
                p_value = data['p_value']
                
                comparison_text = f"• {comparison_key}: {better_model} es significativamente mejor (p = {p_value:.4f})"
                self.add_text(comparison_text)
        
        # Comparaciones no significativas
        if non_significant_comparisons:
            self.add_text("\\n⚖️ Sin Diferencias Estadísticamente Significativas:")
            
            for comparison_key, data in non_significant_comparisons:
                p_value = data['p_value']
                comparison_text = f"• {comparison_key}: No hay diferencia significativa (p = {p_value:.4f})"
                self.add_text(comparison_text)
    
    def add_methodology(self):
        """
        Añade sección de metodología
        """
        self.add_section_title("🔬 Metodología de Evaluación")
        
        methodology_text = """
        Dataset:
        • Fuente: ISIC (International Skin Imaging Collaboration)
        • Tipo: Imágenes reales de lesiones cutáneas
        • Clases: Benigno (0) y Maligno (1)
        • División: 80% entrenamiento, 20% evaluación
        
        Modelos Evaluados:
        • EfficientNetB4: Red neuronal convolucional pre-entrenada
        • ResNet152: Arquitectura residual profunda
        • CNN Personalizada: Arquitectura diseñada específicamente
        
        Métricas de Evaluación:
        • Accuracy: Proporción de predicciones correctas
        • Precision: Proporción de verdaderos positivos entre predicciones positivas
        • Recall (Sensibilidad): Proporción de verdaderos positivos detectados
        • F1-Score: Media armónica entre precisión y recall
        • MCC: Matthews Correlation Coefficient (métrica balanceada)
        • AUC-ROC: Área bajo la curva ROC
        • Especificidad: Proporción de verdaderos negativos detectados
        
        Análisis Estadístico:
        • Prueba de McNemar: Para comparar rendimiento entre modelos
        • Nivel de significancia: α = 0.05
        • Corrección de continuidad aplicada
        """
        
        self.add_text(methodology_text)
    
    def add_conclusions(self, metrics_data):
        """
        Añade conclusiones y recomendaciones
        
        Args:
            metrics_data: Datos de métricas
        """
        self.add_section_title("💡 Conclusiones y Recomendaciones")
        
        # Encontrar mejor modelo
        best_model = max(metrics_data.items(), key=lambda x: x[1]['mcc'])
        model_name, model_data = best_model
        
        # Análisis general
        avg_accuracy = np.mean([data['accuracy'] for data in metrics_data.values()])
        avg_mcc = np.mean([data['mcc'] for data in metrics_data.values()])
        
        conclusions_text = f"""
        Resultados Principales:
        • Mejor modelo: {model_name} con MCC = {model_data['mcc']:.3f}
        • Accuracy promedio: {avg_accuracy:.3f}
        • MCC promedio: {avg_mcc:.3f}
        
        Recomendaciones:
        1. Para uso clínico, considerar el modelo con mayor especificidad para minimizar falsas alarmas
        2. Para screening, priorizar modelos con alta sensibilidad para detectar más casos malignos
        3. El MCC es la métrica más apropiada para datasets desbalanceados como este
        4. Se recomienda validación adicional con datasets externos
        
        Limitaciones:
        • Evaluación limitada a dataset ISIC
        • Posible sesgo en la selección de imágenes
        • Necesidad de validación clínica adicional
        • Variabilidad en condiciones de captura de imágenes
        
        Trabajo Futuro:
        • Validación cruzada con múltiples datasets
        • Análisis de interpretabilidad de los modelos
        • Evaluación en condiciones clínicas reales
        • Optimización de hiperparámetros específicos
        """
        
        self.add_text(conclusions_text)
    
    def add_image_if_exists(self, image_path, width=180):
        """
        Añade imagen si existe
        
        Args:
            image_path: Ruta a la imagen
            width: Ancho de la imagen en el PDF
        """
        if Path(image_path).exists():
            try:
                self.pdf.image(str(image_path), x=10, w=width)
                self.pdf.ln(width * 0.6)  # Espaciado basado en el ancho
            except Exception as e:
                print(f"❌ Error añadiendo imagen {image_path}: {e}")
    
    def generate_complete_report(self, metrics_data, comparison_data=None, 
                               title="Reporte Completo de Evaluación", 
                               include_images=True, output_filename="reporte_completo.pdf"):
        """
        Genera el reporte completo
        
        Args:
            metrics_data: Datos de métricas de los modelos
            comparison_data: Datos de comparaciones estadísticas
            title: Título del reporte
            include_images: Si incluir imágenes
            output_filename: Nombre del archivo de salida
            
        Returns:
            str: Ruta del archivo generado
        """
        print(f"📄 Generando reporte: {title}")
        
        # Página de título
        self.add_title_page(title)
        
        # Nueva página para contenido
        self.pdf.add_page()
        
        # Resumen ejecutivo
        self.add_section_title("📋 Resumen Ejecutivo")
        best_model = max(metrics_data.items(), key=lambda x: x[1]['mcc'])
        
        executive_summary = f"""
        Se evaluaron {len(metrics_data)} modelos de aprendizaje automático para la clasificación de lesiones cutáneas 
        utilizando el dataset ISIC. El mejor rendimiento fue obtenido por {best_model[0]} con un MCC de {best_model[1]['mcc']:.3f}.
        
        Total de muestras evaluadas: {list(metrics_data.values())[0]['test_samples']:,}
        Distribución: {list(metrics_data.values())[0]['class_distribution'][0]:,} benignos, {list(metrics_data.values())[0]['class_distribution'][1]:,} malignos
        """
        
        self.add_text(executive_summary)
        
        # Tabla de métricas
        self.add_metrics_table(metrics_data)
        
        # Análisis del mejor modelo
        self.add_best_model_analysis(metrics_data)
        
        # Nueva página para comparaciones
        self.pdf.add_page()
        
        # Comparaciones estadísticas
        if comparison_data:
            self.add_comparison_analysis(comparison_data)
        
        # Metodología
        self.add_methodology()
        
        # Nueva página para conclusiones
        self.pdf.add_page()
        
        # Conclusiones
        self.add_conclusions(metrics_data)
        
        # Añadir imágenes si están disponibles
        if include_images:
            self.pdf.add_page()
            self.add_section_title("📊 Visualizaciones")
            
            plots_dir = Path("plots")
            if plots_dir.exists():
                # Matriz de confusión comparativa
                img_path = plots_dir / "confusion_matrices_comparison.png"
                if img_path.exists():
                    self.add_text("Matrices de Confusión:")
                    self.add_image_if_exists(img_path)
                
                # Comparación de métricas
                img_path = plots_dir / "metrics_comparison.png"
                if img_path.exists():
                    self.add_text("Comparación de Métricas:")
                    self.add_image_if_exists(img_path)
                
                # Curvas ROC
                img_path = plots_dir / "roc_curves.png"
                if img_path.exists():
                    self.add_text("Curvas ROC:")
                    self.add_image_if_exists(img_path)
        
        # Guardar PDF
        output_path = self.output_dir / output_filename
        self.pdf.output(str(output_path))
        
        print(f"✅ Reporte generado: {output_path}")
        
        return str(output_path)
