"""
Sistema de Gestión Principal para Diagnóstico de Cáncer de Piel

Este módulo centraliza la gestión de modelos, evaluación y análisis del sistema.
Elimina datos hardcodeados y proporciona una interfaz robusta.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# Importar módulos del sistema
from model_utils import load_models, predict_image, get_model_info
from model_evaluator import ModelEvaluator
from metrics_visualizer import MetricsVisualizer
from pdf_generator import PDFReportGenerator
from preprocessing import preprocess_image

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinCancerDiagnosisSystem:
    """
    Sistema principal para diagnóstico de cáncer de piel
    
    Características:
    - Gestión centralizada de modelos
    - Evaluación automática con dataset de test
    - Visualizaciones dinámicas
    - Generación de reportes PDF
    - Cache persistente de métricas
    - Sin datos hardcodeados
    """
    
    def __init__(self, 
                 test_data_path: str = "../data/ISIC_dataset_test",
                 models_dir: str = "models",
                 cache_dir: str = "cache",
                 plots_dir: str = "plots",
                 reports_dir: str = "reports"):
        """
        Inicializa el sistema
        
        Args:
            test_data_path: Ruta al dataset de test
            models_dir: Directorio de modelos
            cache_dir: Directorio de cache
            plots_dir: Directorio de gráficos
            reports_dir: Directorio de reportes
        """
        self.test_data_path = Path(test_data_path)
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.plots_dir = Path(plots_dir)
        self.reports_dir = Path(reports_dir)
        
        # Crear directorios si no existen
        for directory in [self.cache_dir, self.plots_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.models = None
        self.evaluator = None
        self.visualizer = None
        self.pdf_generator = None
        self.metrics_cache = {}
        
        # Archivos de cache
        self.metrics_file = self.cache_dir / "model_metrics.json"
        self.comparison_file = self.cache_dir / "model_comparisons.json"
        self.system_config_file = self.cache_dir / "system_config.json"
        
        # Configuración del sistema
        self.config = self._load_system_config()
        
        logger.info("Sistema de diagnóstico de cáncer de piel inicializado")
    
    def _load_system_config(self) -> Dict:
        """Carga la configuración del sistema"""
        default_config = {
            'model_threshold': 0.5,
            'evaluation_batch_size': 32,
            'image_target_size': [300, 300],
            'supported_formats': ['jpg', 'jpeg', 'png'],
            'cache_enabled': True,
            'auto_refresh_hours': 24,
            'last_evaluation': None,
            'system_version': '2.0.0'
        }
        
        if self.system_config_file.exists():
            try:
                with open(self.system_config_file, 'r') as f:
                    config = json.load(f)
                    # Actualizar con valores por defecto si faltan
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.warning(f"Error cargando configuración: {e}")
        
        return default_config
    
    def _save_system_config(self):
        """Guarda la configuración del sistema"""
        try:
            with open(self.system_config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
    
    def initialize_components(self) -> bool:
        """
        Inicializa todos los componentes del sistema
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Cargar modelos
            logger.info("Cargando modelos...")
            self.models = load_models()
            
            if not self.models:
                logger.error("No se pudieron cargar los modelos")
                return False
            
            logger.info(f"Modelos cargados: {list(self.models.keys())}")
            
            # Inicializar evaluador
            if self.test_data_path.exists():
                logger.info("Inicializando evaluador...")
                self.evaluator = ModelEvaluator(
                    test_data_path=str(self.test_data_path),
                    cache_dir=str(self.cache_dir)
                )
            else:
                logger.warning(f"Dataset de test no encontrado: {self.test_data_path}")
            
            # Cargar métricas existentes
            self._load_cached_metrics()
            
            # Inicializar visualizador si hay métricas
            if self.metrics_cache:
                self.visualizer = MetricsVisualizer(
                    self.metrics_cache,
                    plots_dir=str(self.plots_dir)
                )
            
            # Inicializar generador de PDF
            self.pdf_generator = PDFReportGenerator(
                output_dir=str(self.reports_dir)
            )
            
            logger.info("Componentes inicializados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
            return False
    
    def _load_cached_metrics(self):
        """Carga métricas desde cache"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_cache = json.load(f)
                logger.info(f"Métricas cargadas para {len(self.metrics_cache)} modelos")
            except Exception as e:
                logger.error(f"Error cargando métricas: {e}")
                self.metrics_cache = {}
    
    def evaluate_models(self, force_refresh: bool = False) -> Dict:
        """
        Evalúa todos los modelos disponibles
        
        Args:
            force_refresh: Forzar reevaluación
            
        Returns:
            Dict: Métricas de evaluación
        """
        if not self.evaluator:
            logger.error("Evaluador no inicializado")
            return {}
        
        try:
            logger.info("Iniciando evaluación de modelos...")
            
            # Verificar si necesita actualización
            needs_update = force_refresh
            
            if not needs_update and self.config.get('last_evaluation'):
                last_eval = datetime.fromisoformat(self.config['last_evaluation'])
                hours_since_eval = (datetime.now() - last_eval).total_seconds() / 3600
                needs_update = hours_since_eval > self.config.get('auto_refresh_hours', 24)
            
            if not needs_update and self.metrics_cache:
                logger.info("Usando métricas en cache")
                return self.metrics_cache
            
            # Realizar evaluación
            metrics = self.evaluator.evaluate_all_models(force_refresh=force_refresh)
            
            if metrics:
                self.metrics_cache = metrics
                self.config['last_evaluation'] = datetime.now().isoformat()
                self._save_system_config()
                
                # Actualizar visualizador
                self.visualizer = MetricsVisualizer(
                    self.metrics_cache,
                    plots_dir=str(self.plots_dir)
                )
                
                logger.info(f"Evaluación completada para {len(metrics)} modelos")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            return {}
    
    def predict_single_image(self, image, model_name: str = None) -> Dict:
        """
        Realiza predicción en una imagen
        
        Args:
            image: Imagen a predecir
            model_name: Nombre del modelo (usa el mejor si no se especifica)
            
        Returns:
            Dict: Resultado de la predicción
        """
        if not self.models:
            return {'error': 'No hay modelos disponibles'}
        
        try:
            # Seleccionar modelo
            if model_name and model_name in self.models:
                model = self.models[model_name]
            else:
                # Usar el mejor modelo basado en MCC
                model_name = self.get_best_model_name()
                if not model_name:
                    model_name = list(self.models.keys())[0]
                model = self.models[model_name]
            
            # Preprocesar imagen
            processed_image = preprocess_image(
                image, 
                target_size=tuple(self.config['image_target_size'])
            )
            
            # Realizar predicción
            diagnosis, confidence_percent, raw_confidence = predict_image(
                model, 
                processed_image,
                threshold=self.config['model_threshold']
            )
            
            # Obtener métricas del modelo si están disponibles
            model_metrics = self.metrics_cache.get(model_name.lower().replace(' ', '_'), {})
            
            result = {
                'diagnosis': diagnosis,
                'confidence_percent': confidence_percent,
                'raw_confidence': raw_confidence,
                'model_used': model_name,
                'model_metrics': {
                    'accuracy': model_metrics.get('accuracy', 'N/A'),
                    'precision': model_metrics.get('precision', 'N/A'),
                    'recall': model_metrics.get('recall', 'N/A'),
                    'mcc': model_metrics.get('mcc', 'N/A')
                },
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return {'error': str(e)}
    
    def get_best_model_name(self) -> Optional[str]:
        """
        Obtiene el nombre del mejor modelo basado en MCC
        
        Returns:
            str: Nombre del mejor modelo o None
        """
        if not self.metrics_cache:
            return None
        
        best_model = max(
            self.metrics_cache.items(), 
            key=lambda x: x[1].get('mcc', 0)
        )
        
        return best_model[0]
    
    def get_models_info(self) -> Dict:
        """
        Obtiene información de todos los modelos
        
        Returns:
            Dict: Información de los modelos
        """
        if not self.models:
            return {}
        
        info = {}
        for name, model in self.models.items():
            model_info = get_model_info(model)
            
            # Añadir métricas si están disponibles
            metrics_key = name.lower().replace(' ', '_')
            if metrics_key in self.metrics_cache:
                model_info['metrics'] = self.metrics_cache[metrics_key]
            
            info[name] = model_info
        
        return info
    
    def get_system_status(self) -> Dict:
        """
        Obtiene el estado del sistema
        
        Returns:
            Dict: Estado del sistema
        """
        status = {
            'models_loaded': len(self.models) if self.models else 0,
            'models_available': list(self.models.keys()) if self.models else [],
            'metrics_cached': len(self.metrics_cache),
            'evaluator_ready': self.evaluator is not None,
            'visualizer_ready': self.visualizer is not None,
            'pdf_generator_ready': self.pdf_generator is not None,
            'test_dataset_available': self.test_data_path.exists(),
            'last_evaluation': self.config.get('last_evaluation'),
            'system_version': self.config.get('system_version', '2.0.0'),
            'cache_enabled': self.config.get('cache_enabled', True)
        }
        
        if self.metrics_cache:
            best_model = self.get_best_model_name()
            if best_model:
                status['best_model'] = {
                    'name': best_model,
                    'mcc': self.metrics_cache[best_model].get('mcc', 0)
                }
        
        return status
    
    def generate_visualizations(self, save_plots: bool = True) -> Dict:
        """
        Genera todas las visualizaciones
        
        Args:
            save_plots: Si guardar los gráficos
            
        Returns:
            Dict: Información sobre las visualizaciones generadas
        """
        if not self.visualizer:
            return {'error': 'Visualizador no disponible'}
        
        try:
            if save_plots:
                saved_files = self.visualizer.save_all_plots()
                return {
                    'success': True,
                    'files_saved': saved_files,
                    'plots_directory': str(self.plots_dir)
                }
            else:
                return {'success': True, 'plots_generated': True}
                
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {e}")
            return {'error': str(e)}
    
    def generate_complete_report(self, 
                               title: str = "Reporte Completo de Evaluación",
                               include_images: bool = True,
                               filename: str = None) -> Dict:
        """
        Genera un reporte completo en PDF
        
        Args:
            title: Título del reporte
            include_images: Incluir imágenes
            filename: Nombre del archivo
            
        Returns:
            Dict: Información sobre el reporte generado
        """
        if not self.pdf_generator:
            return {'error': 'Generador de PDF no disponible'}
        
        if not self.metrics_cache:
            return {'error': 'No hay métricas disponibles'}
        
        try:
            # Generar visualizaciones si se incluyen imágenes
            if include_images:
                self.generate_visualizations(save_plots=True)
            
            # Cargar comparaciones
            comparison_data = None
            if self.comparison_file.exists():
                with open(self.comparison_file, 'r') as f:
                    comparison_data = json.load(f)
            
            # Generar nombre de archivo si no se proporciona
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reporte_evaluacion_{timestamp}.pdf"
            
            # Generar reporte
            output_path = self.pdf_generator.generate_complete_report(
                metrics_data=self.metrics_cache,
                comparison_data=comparison_data,
                title=title,
                include_images=include_images,
                output_filename=filename
            )
            
            return {
                'success': True,
                'output_path': output_path,
                'filename': filename,
                'report_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            return {'error': str(e)}
    
    def get_evaluation_summary(self) -> Dict:
        """
        Obtiene un resumen de la evaluación
        
        Returns:
            Dict: Resumen de la evaluación
        """
        if not self.metrics_cache:
            return {'error': 'No hay métricas disponibles'}
        
        try:
            # Estadísticas generales
            accuracies = [m['accuracy'] for m in self.metrics_cache.values()]
            mccs = [m['mcc'] for m in self.metrics_cache.values()]
            
            # Mejor modelo
            best_model_name = self.get_best_model_name()
            best_model_metrics = self.metrics_cache.get(best_model_name, {})
            
            # Información del dataset
            first_model = list(self.metrics_cache.values())[0]
            
            summary = {
                'total_models': len(self.metrics_cache),
                'test_samples': first_model.get('test_samples', 0),
                'class_distribution': first_model.get('class_distribution', [0, 0]),
                'average_accuracy': np.mean(accuracies),
                'average_mcc': np.mean(mccs),
                'best_model': {
                    'name': best_model_name,
                    'accuracy': best_model_metrics.get('accuracy', 0),
                    'mcc': best_model_metrics.get('mcc', 0),
                    'precision': best_model_metrics.get('precision', 0),
                    'recall': best_model_metrics.get('recall', 0)
                },
                'model_rankings': sorted(
                    [(name, metrics['mcc']) for name, metrics in self.metrics_cache.items()],
                    key=lambda x: x[1],
                    reverse=True
                ),
                'last_evaluation': self.config.get('last_evaluation')
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return {'error': str(e)}
    
    def cleanup_cache(self):
        """Limpia el cache del sistema"""
        try:
            cache_files = [
                self.metrics_file,
                self.comparison_file,
                self.system_config_file
            ]
            
            for file in cache_files:
                if file.exists():
                    file.unlink()
            
            self.metrics_cache = {}
            self.config = self._load_system_config()
            
            logger.info("Cache limpiado exitosamente")
            
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
