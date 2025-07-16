import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class MetricsVisualizer:
    """
    Visualizador de métricas para modelos de cáncer de piel
    """
    
    def __init__(self, metrics_data, plots_dir="plots"):
        """
        Inicializa el visualizador
        
        Args:
            metrics_data: Diccionario con métricas de los modelos
            plots_dir: Directorio para guardar gráficos
        """
        self.metrics_data = metrics_data
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_confusion_matrix(self, model_name, save=True):
        """
        Grafica la matriz de confusión para un modelo
        
        Args:
            model_name: Nombre del modelo
            save: Si guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure
        """
        if model_name not in self.metrics_data:
            print(f"❌ Modelo {model_name} no encontrado")
            return None
        
        cm = np.array(self.metrics_data[model_name]['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benigno', 'Maligno'],
                   yticklabels=['Benigno', 'Maligno'],
                   ax=ax)
        
        ax.set_title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Valor Real', fontsize=12)
        
        # Añadir métricas
        accuracy = self.metrics_data[model_name]['accuracy']
        precision = self.metrics_data[model_name]['precision']
        recall = self.metrics_data[model_name]['recall']
        f1 = self.metrics_data[model_name]['f1_score']
        
        metrics_text = f'Accuracy: {accuracy:.3f}\\nPrecision: {precision:.3f}\\nRecall: {recall:.3f}\\nF1-Score: {f1:.3f}'
        ax.text(2.5, 0.5, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            filename = self.plots_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Matriz de confusión guardada: {filename}")
        
        return fig
    
    def plot_confusion_matrices_comparison(self, save=True):
        """
        Grafica todas las matrices de confusión en una sola figura
        
        Args:
            save: Si guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure
        """
        n_models = len(self.metrics_data)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, data) in enumerate(self.metrics_data.items()):
            row = idx // cols
            col = idx % cols
            
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            cm = np.array(data['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Benigno', 'Maligno'],
                       yticklabels=['Benigno', 'Maligno'],
                       ax=ax)
            
            ax.set_title(f'{model_name}\\nMCC: {data["mcc"]:.3f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Valor Real')
        
        # Ocultar axes vacíos
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            filename = self.plots_dir / 'confusion_matrices_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Comparación de matrices guardada: {filename}")
        
        return fig
    
    def plot_metrics_comparison(self, save=True):
        """
        Grafica comparación de métricas entre modelos
        
        Args:
            save: Si guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure
        """
        # Preparar datos
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc_roc']
        model_names = list(self.metrics_data.keys())
        
        data = []
        for model_name in model_names:
            for metric_name in metrics_names:
                data.append({
                    'Model': model_name,
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': self.metrics_data[model_name][metric_name]
                })
        
        df = pd.DataFrame(data)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Gráfico de barras agrupadas
        sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
        
        ax.set_title('Comparación de Métricas entre Modelos', fontsize=16, fontweight='bold')
        ax.set_xlabel('Métricas', fontsize=12)
        ax.set_ylabel('Valor', fontsize=12)
        ax.legend(title='Modelos', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Añadir líneas de referencia
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Referencia 0.5')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Referencia 0.7')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = self.plots_dir / 'metrics_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Comparación de métricas guardada: {filename}")
        
        return fig
    
    def plot_roc_curves(self, save=True):
        """
        Grafica las curvas ROC para todos los modelos
        
        Args:
            save: Si guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.metrics_data)))
        
        for (model_name, data), color in zip(self.metrics_data.items(), colors):
            try:
                # Calcular curva ROC
                true_labels = np.array(data['true_labels'])
                pred_probs = np.array(data['prediction_probabilities'])
                
                fpr, tpr, _ = roc_curve(true_labels, pred_probs)
                auc_score = auc(fpr, tpr)
                
                # Graficar curva
                ax.plot(fpr, tpr, color=color, linewidth=2,
                       label=f'{model_name} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                print(f"❌ Error graficando ROC para {model_name}: {e}")
                continue
        
        # Línea diagonal (clasificador aleatorio)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador Aleatorio')
        
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
        ax.set_title('Curvas ROC - Comparación de Modelos', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.plots_dir / 'roc_curves.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Curvas ROC guardadas: {filename}")
        
        return fig
    
    def plot_mcc_comparison(self, save=True):
        """
        Grafica comparación específica de MCC
        
        Args:
            save: Si guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure
        """
        # Preparar datos
        model_names = list(self.metrics_data.keys())
        mcc_values = [self.metrics_data[name]['mcc'] for name in model_names]
        
        # Ordenar por MCC
        sorted_data = sorted(zip(model_names, mcc_values), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_mcc = zip(*sorted_data)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Colores basados en el valor MCC
        colors = ['green' if mcc > 0.5 else 'orange' if mcc > 0.3 else 'red' for mcc in sorted_mcc]
        
        bars = ax.bar(sorted_names, sorted_mcc, color=colors, alpha=0.7, edgecolor='black')
        
        # Añadir valores en las barras
        for bar, mcc in zip(bars, sorted_mcc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mcc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Comparación de Matthews Correlation Coefficient (MCC)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('MCC', fontsize=12)
        
        # Líneas de referencia
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='MCC = 0.3 (Bueno)')
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='MCC = 0.5 (Excelente)')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = self.plots_dir / 'mcc_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Comparación MCC guardada: {filename}")
        
        return fig
    
    def plot_performance_radar(self, save=True):
        """
        Grafica un gráfico de radar con el rendimiento de los modelos
        
        Args:
            save: Si guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure
        """
        # Métricas para el radar
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUC-ROC']
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc_roc']
        
        # Número de métricas
        N = len(metrics_names)
        
        # Ángulos para cada métrica
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Cerrar el círculo
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.metrics_data)))
        
        for (model_name, data), color in zip(self.metrics_data.items(), colors):
            # Obtener valores de métricas
            values = [data[key] for key in metrics_keys]
            values += values[:1]  # Cerrar el círculo
            
            # Graficar
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Configurar ejes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        ax.set_title('Rendimiento de Modelos - Gráfico de Radar', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        plt.tight_layout()
        
        if save:
            filename = self.plots_dir / 'performance_radar.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico de radar guardado: {filename}")
        
        return fig
    
    def save_all_plots(self):
        """
        Guarda todos los gráficos disponibles
        
        Returns:
            list: Lista de archivos guardados
        """
        print("💾 Guardando todas las visualizaciones...")
        
        saved_files = []
        
        try:
            # Matrices de confusión individuales
            for model_name in self.metrics_data.keys():
                fig = self.plot_confusion_matrix(model_name, save=True)
                if fig:
                    plt.close(fig)
                    saved_files.append(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
            
            # Comparación de matrices
            fig = self.plot_confusion_matrices_comparison(save=True)
            if fig:
                plt.close(fig)
                saved_files.append('confusion_matrices_comparison.png')
            
            # Comparación de métricas
            fig = self.plot_metrics_comparison(save=True)
            if fig:
                plt.close(fig)
                saved_files.append('metrics_comparison.png')
            
            # Curvas ROC
            fig = self.plot_roc_curves(save=True)
            if fig:
                plt.close(fig)
                saved_files.append('roc_curves.png')
            
            # Comparación MCC
            fig = self.plot_mcc_comparison(save=True)
            if fig:
                plt.close(fig)
                saved_files.append('mcc_comparison.png')
            
            # Gráfico de radar
            fig = self.plot_performance_radar(save=True)
            if fig:
                plt.close(fig)
                saved_files.append('performance_radar.png')
            
            print(f"✅ {len(saved_files)} visualizaciones guardadas en {self.plots_dir}")
            
        except Exception as e:
            print(f"❌ Error guardando visualizaciones: {e}")
        
        return saved_files
    
    def create_metrics_summary_table(self):
        """
        Crea una tabla resumen con todas las métricas
        
        Returns:
            pandas.DataFrame
        """
        data = []
        
        for model_name, metrics in self.metrics_data.items():
            data.append({
                'Modelo': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'MCC': f"{metrics['mcc']:.3f}",
                'AUC-ROC': f"{metrics['auc_roc']:.3f}",
                'Sensitivity': f"{metrics['sensitivity']:.3f}",
                'Specificity': f"{metrics['specificity']:.3f}",
                'Muestras Test': f"{metrics['test_samples']:,}"
            })
        
        df = pd.DataFrame(data)
        
        # Ordenar por MCC
        df['MCC_numeric'] = df['MCC'].astype(float)
        df = df.sort_values('MCC_numeric', ascending=False)
        df = df.drop('MCC_numeric', axis=1)
        
        return df
    
    def generate_insights(self):
        """
        Genera insights automáticos basados en las métricas
        
        Returns:
            list: Lista de insights
        """
        insights = []
        
        # Encontrar el mejor modelo
        best_model = max(self.metrics_data.items(), key=lambda x: x[1]['mcc'])
        insights.append(f"🏆 El mejor modelo es **{best_model[0]}** con MCC = {best_model[1]['mcc']:.3f}")
        
        # Análisis de accuracy
        accuracies = [data['accuracy'] for data in self.metrics_data.values()]
        avg_accuracy = np.mean(accuracies)
        insights.append(f"📊 Accuracy promedio: {avg_accuracy:.3f}")
        
        # Análisis de precision vs recall
        for model_name, data in self.metrics_data.items():
            precision = data['precision']
            recall = data['recall']
            
            if precision > recall + 0.1:
                insights.append(f"⚖️ {model_name}: Alta precisión ({precision:.3f}) pero baja sensibilidad ({recall:.3f}) - Pocos falsos positivos")
            elif recall > precision + 0.1:
                insights.append(f"⚖️ {model_name}: Alta sensibilidad ({recall:.3f}) pero baja precisión ({precision:.3f}) - Detecta más casos malignos")
        
        # Análisis de MCC
        mcc_values = [data['mcc'] for data in self.metrics_data.values()]
        excellent_models = [name for name, data in self.metrics_data.items() if data['mcc'] > 0.5]
        
        if excellent_models:
            insights.append(f"🌟 Modelos con rendimiento excelente (MCC > 0.5): {', '.join(excellent_models)}")
        
        # Análisis de especificidad
        high_specificity = [name for name, data in self.metrics_data.items() if data['specificity'] > 0.9]
        if high_specificity:
            insights.append(f"🎯 Modelos con alta especificidad (> 0.9): {', '.join(high_specificity)}")
        
        return insights
