# visualization.py
"""
Módulo de visualización para el sistema de diagnóstico de cáncer de piel
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from config import get_plot_config

# Obtener configuración de gráficos
PLOT_CONFIG = get_plot_config()


def plot_confusion_matrix(cm, model_name):
    """
    Genera una visualización atractiva de la matriz de confusión
    
    Args:
        cm (array): Matriz de confusión
        model_name (str): Nombre del modelo
    
    Returns:
        matplotlib.figure.Figure: Figura de la matriz de confusión
    """
    try:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_small'])
        
        # Crear matriz de confusión con seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benigno', 'Maligno'],
                   yticklabels=['Benigno', 'Maligno'],
                   ax=ax)
        
        ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Valor Real', fontsize=12)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de matriz de confusión: {str(e)}")
        return None


def create_metrics_dashboard(metrics_data, model_name):
    """
    Crea un dashboard visual con las métricas de rendimiento
    
    Args:
        metrics_data (dict): Diccionario con métricas del modelo
        model_name (str): Nombre del modelo
    
    Returns:
        matplotlib.figure.Figure: Figura del dashboard
    """
    try:
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=PLOT_CONFIG['figsize_default'])
        fig.suptitle(f'Dashboard de Métricas - {model_name}', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Accuracy como gráfico de pastel
        ax1.pie([metrics_data['accuracy'], 1-metrics_data['accuracy']], 
               labels=['Accuracy', 'Error'], 
               colors=[PLOT_CONFIG['colors']['excellent'], '#FF6B6B'],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Accuracy del Modelo', fontweight='bold')
        
        # Gráfico 2: Precision, Recall, F1-Score
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [metrics_data['precision'], metrics_data['sensitivity'], metrics_data['f1_score']]
        colors = [PLOT_CONFIG['colors']['good'], '#2196F3', '#FF9800']
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_title('Métricas de Clasificación', fontweight='bold')
        ax2.set_ylabel('Valor')
        
        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 3: Sensitivity vs Specificity
        categories = ['Sensitivity\n(Recall)', 'Specificity']
        sens_spec_values = [metrics_data['sensitivity'], metrics_data['specificity']]
        
        bars3 = ax3.bar(categories, sens_spec_values, 
                       color=[PLOT_CONFIG['colors']['malignant'], PLOT_CONFIG['colors']['benign']], 
                       alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_title('Sensitivity vs Specificity', fontweight='bold')
        ax3.set_ylabel('Valor')
        
        for bar, value in zip(bars3, sens_spec_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 4: Resumen de métricas en texto
        ax4.axis('off')
        
        # Extraer valores de la matriz de confusión
        # La matriz sigue el formato [[TN, FP], [FN, TP]]
        cm = metrics_data['confusion_matrix']
        tp = cm[1][1]
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        
        metrics_text = f"""
        == RESUMEN DE METRICAS ==
        
        Accuracy: {metrics_data['accuracy']:.3f} ({metrics_data['accuracy']*100:.1f}%)
        Precision: {metrics_data['precision']:.3f} ({metrics_data['precision']*100:.1f}%)
        Sensitivity: {metrics_data['sensitivity']:.3f} ({metrics_data['sensitivity']*100:.1f}%)
        Specificity: {metrics_data['specificity']:.3f} ({metrics_data['specificity']*100:.1f}%)
        F1-Score: {metrics_data['f1_score']:.3f} ({metrics_data['f1_score']*100:.1f}%)
        MCC: {metrics_data['mcc']:.3f}
        
        >> ELEMENTOS MATRIZ:
        TP: {tp} | TN: {tn}
        FP: {fp} | FN: {fn}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error al crear dashboard de métricas: {str(e)}")
        return None


def create_advanced_metrics_dashboard(metrics_data, model_name):
    """
    Crea un dashboard avanzado con todas las métricas incluyendo MCC
    
    Args:
        metrics_data (dict): Diccionario con métricas del modelo
        model_name (str): Nombre del modelo
    
    Returns:
        matplotlib.figure.Figure: Figura del dashboard avanzado
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=PLOT_CONFIG['figsize_large'])
        fig.suptitle(f'Análisis Estadístico Avanzado - {model_name}', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Métricas principales
        metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'MCC']
        metrics_values = [
            metrics_data['accuracy'],
            metrics_data['sensitivity'],
            metrics_data['specificity'],
            metrics_data['precision'],
            metrics_data['f1_score'],
            metrics_data['mcc']
        ]
        colors = ['#2E8B57', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
        
        bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_title('Métricas de Rendimiento', fontweight='bold')
        ax1.set_ylabel('Valor')
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión con valores
        cm = metrics_data['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benigno', 'Maligno'],
                   yticklabels=['Benigno', 'Maligno'],
                   ax=ax2)
        ax2.set_title('Matriz de Confusión', fontweight='bold')
        ax2.set_xlabel('Predicción')
        ax2.set_ylabel('Valor Real')
        
        # Gráfico 3: Interpretación de MCC
        ax3.axis('off')
        mcc_interpretation = f"""
        == COEFICIENTE DE MATTHEWS (MCC) ==
        
        Valor: {metrics_data['mcc']:.3f}
        
        -> INTERPRETACION:
        * MCC = 1.0: Predicción perfecta
        * MCC = 0.0: Predicción aleatoria
        * MCC = -1.0: Predicción inversa perfecta
        
        >> CLASIFICACION:
        * Excelente: MCC > 0.7
        * Bueno: 0.3 < MCC ≤ 0.7
        * Regular: 0.1 < MCC ≤ 0.3
        * Pobre: MCC ≤ 0.1
        
        -> VENTAJAS:
        * Balanceado para clases desequilibradas
        * Considera todos los elementos de la matriz
        * Más robusto que accuracy para datasets desbalanceados
        """
        ax3.text(0.1, 0.9, mcc_interpretation, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Gráfico 4: Resumen estadístico
        ax4.axis('off')
        cm = metrics_data['confusion_matrix']
        tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
        
        summary_text = f"""
        == RESUMEN ESTADISTICO COMPLETO ==
        
        ** METRICAS PRINCIPALES:
        -> Accuracy: {metrics_data['accuracy']:.3f} ({metrics_data['accuracy']*100:.1f}%)
        -> Sensibilidad: {metrics_data['sensitivity']:.3f} ({metrics_data['sensitivity']*100:.1f}%)
        -> Specificidad: {metrics_data['specificity']:.3f} ({metrics_data['specificity']*100:.1f}%)
        -> Precision: {metrics_data['precision']:.3f} ({metrics_data['precision']*100:.1f}%)
        -> F1-Score: {metrics_data['f1_score']:.3f} ({metrics_data['f1_score']*100:.1f}%)
        -> MCC: {metrics_data['mcc']:.3f}
        
        ** ELEMENTOS DE LA MATRIZ:
        -> TP: {tp} (Verdaderos Positivos)
        -> TN: {tn} (Verdaderos Negativos)
        -> FP: {fp} (Falsos Positivos)
        -> FN: {fn} (Falsos Negativos)
        
        >> EVALUACION MEDICA:
        * Sensibilidad alta: Detección temprana
        * Especificidad alta: Menos falsas alarmas
        * MCC alto: Balance general excelente
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error al crear dashboard avanzado: {str(e)}")
        return None


def create_model_comparison_plots(df_comparison):
    """
    Crea gráficos de comparación entre modelos
    
    Args:
        df_comparison (pd.DataFrame): DataFrame con resultados de comparación
    
    Returns:
        tuple: (figura_confianza, figura_tiempo)
    """
    try:
        # Gráfico de comparación de confianza
        fig1, ax1 = plt.subplots(figsize=PLOT_CONFIG['figsize_small'])
        bars = ax1.bar(df_comparison['Modelo'], df_comparison['Confianza (%)'])
        
        # Colorear barras según diagnóstico
        colors = [PLOT_CONFIG['colors']['benign'] if d == 'Benigno' else PLOT_CONFIG['colors']['malignant'] 
                 for d in df_comparison['Diagnostico']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        ax1.set_ylabel('Confianza (%)')
        ax1.set_title('Comparación de Confianza por Modelo')
        ax1.set_ylim(0, 100)
        
        # Añadir valores en las barras
        for i, v in enumerate(df_comparison['Confianza (%)']):
            ax1.text(i, v + 1, f'{v}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Gráfico de tiempo de inferencia
        fig2, ax2 = plt.subplots(figsize=PLOT_CONFIG['figsize_small'])
        bars2 = ax2.bar(df_comparison['Modelo'], df_comparison['Tiempo (ms)'])
        ax2.set_ylabel('Tiempo de Inferencia (ms)')
        ax2.set_title('Velocidad de Inferencia por Modelo')
        
        # Añadir valores en las barras
        for i, v in enumerate(df_comparison['Tiempo (ms)']):
            ax2.text(i, v + 0.5, f'{v}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig1, fig2
    except Exception as e:
        st.error(f"Error al crear gráficos de comparación: {str(e)}")
        return None, None


def create_mcc_comparison_chart(mcc_data):
    """
    Crea gráfico comparativo de MCC
    
    Args:
        mcc_data (dict): Datos de MCC para comparación
    
    Returns:
        matplotlib.figure.Figure: Figura del gráfico MCC
    """
    try:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_default'])
        
        # Datos para el gráfico
        model_names = list(mcc_data.keys())
        mcc_values = [data['MCC'] for data in mcc_data.values()]
        colors = [data['Color'] for data in mcc_data.values()]
        
        # Crear barras
        bars = ax.bar(model_names, mcc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Configurar el gráfico
        ax.set_ylabel('Coeficiente de Matthews (MCC)', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Coeficientes de Matthews por Modelo', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, max(mcc_values) * 1.2)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, mcc_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Añadir líneas de referencia para interpretación
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Excelente (>0.7)')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Bueno (>0.3)')
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Regular (>0.1)')
        
        # Configurar leyenda
        ax.legend(loc='upper right', fontsize=10)
        
        # Rotar etiquetas del eje x
        plt.xticks(rotation=45, ha='right')
        
        # Añadir grid para mejor lectura
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Mejorar el layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error al crear gráfico de MCC: {str(e)}")
        return None


def create_mcnemar_plot(mcnemar_results):
    """
    Crea gráfico de resultados de prueba de McNemar
    
    Args:
        mcnemar_results (list): Lista de resultados de McNemar
    
    Returns:
        matplotlib.figure.Figure: Figura del gráfico de McNemar
    """
    try:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_small'])
        
        comparisons_names = [result['Comparación'] for result in mcnemar_results]
        p_values = [result['P-valor'] for result in mcnemar_results]
        
        # Colorear barras según significancia
        colors = []
        for i, (comparison, p) in enumerate(zip(comparisons_names, p_values)):
            if 'EfficientNetB4' in comparison and p < 0.05:
                colors.append(PLOT_CONFIG['colors']['excellent'])  # Verde para EfficientNetB4 superior
            elif p < 0.05:
                colors.append('#FFC107')  # Amarillo para otras diferencias significativas
            else:
                colors.append('#6C757D')  # Gris para no significativas
        
        bars = ax.bar(comparisons_names, p_values, color=colors, alpha=0.7)
        
        # Línea de referencia para p = 0.05
        ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.8, label='α = 0.05')
        
        # Configurar el gráfico
        ax.set_ylabel('P-valor', fontsize=12, fontweight='bold')
        ax.set_title('Prueba de McNemar - Superioridad de EfficientNetB4', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(p_values) * 1.2)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, p_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Configurar leyenda y layout
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error al crear gráfico de McNemar: {str(e)}")
        return None


def save_plot_to_image(fig, filename):
    """
    Guarda un gráfico matplotlib como imagen
    
    Args:
        fig (matplotlib.figure.Figure): Figura a guardar
        filename (str): Nombre del archivo
    
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario
    """
    try:
        fig.savefig(
            filename, 
            dpi=PLOT_CONFIG['dpi'],
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1,
            format='png',
            transparent=False
        )
        return True
    except Exception as e:
        st.error(f"Error al guardar gráfico: {str(e)}")
        return False
