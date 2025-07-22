import subprocess
import threading
import os
import sys
import time
from pathlib import Path
import streamlit as st
import json

def train_hybrid_models_async(progress_callback=None, on_complete=None):
    """
    Ejecuta el entrenamiento de los modelos híbridos de manera asíncrona
    
    Args:
        progress_callback (callable): Función para actualizar el progreso
        on_complete (callable): Función a ejecutar cuando finalice el entrenamiento
    """
    def run_training():
        # Ruta al script de entrenamiento
        script_path = Path('entrenamiento-modelos-hibridos.py').absolute()
        
        if not script_path.exists():
            if progress_callback:
                progress_callback("Error: No se encuentra el script de entrenamiento", 100, error=True)
            return False
            
        # Crear directorio para logs de entrenamiento
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / 'training_progress.json'
        
        # Inicializar archivo de progreso
        with open(log_path, 'w') as f:
            json.dump({
                'status': 'starting',
                'progress': 0,
                'message': 'Iniciando entrenamiento...',
                'current_epoch': 0,
                'total_epochs': 0,
                'current_model': '',
                'error': None
            }, f)
            
        try:
            # Ejecutar el script con Python, redirigiendo la salida a un archivo
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Procesar la salida en tiempo real
            last_progress = 0
            total_models = 2  # Esperamos entrenar 2 modelos híbridos
            models_completed = 0
            
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    # Analizar la línea para extraer información de progreso
                    line = output_line.strip()
                    
                    # Actualizar el archivo de progreso según la información
                    progress_info = {
                        'status': 'running',
                        'message': line,
                        'progress': last_progress
                    }
                    
                    # Detectar inicio de entrenamiento de un modelo
                    if 'Creando modelo híbrido' in line:
                        current_model = line.split(':')[-1].strip()
                        progress_info['current_model'] = current_model
                        last_progress = (models_completed / total_models) * 100
                        progress_info['progress'] = last_progress
                    
                    # Detectar información de epoch
                    if 'Epoch' in line and '/' in line:
                        try:
                            # Intentar extraer información de epoch (e.g., "Epoch 2/25")
                            epoch_info = line.split()[0]
                            current_epoch, total_epochs = map(int, epoch_info.split('/'))
                            progress_info['current_epoch'] = current_epoch
                            progress_info['total_epochs'] = total_epochs
                            
                            # Calcular progreso combinado
                            model_progress = current_epoch / total_epochs
                            last_progress = ((models_completed + model_progress) / total_models) * 100
                            progress_info['progress'] = last_progress
                        except:
                            pass
                    
                    # Detectar finalización de un modelo
                    if 'Resultados finales de' in line:
                        models_completed += 1
                        last_progress = (models_completed / total_models) * 100
                        progress_info['progress'] = last_progress
                    
                    # Detectar finalización del entrenamiento
                    if 'Entrenamiento de modelos híbridos completado' in line:
                        progress_info['status'] = 'completed'
                        progress_info['progress'] = 100
                        last_progress = 100
                    
                    # Guardar la información de progreso
                    with open(log_path, 'w') as f:
                        json.dump(progress_info, f)
                    
                    # Llamar al callback si existe
                    if progress_callback:
                        progress_callback(line, last_progress)
            
            # Procesar errores si existen
            stderr = process.stderr.read()
            if stderr:
                error_message = f"Error en el entrenamiento: {stderr}"
                with open(log_path, 'w') as f:
                    json.dump({
                        'status': 'error',
                        'progress': last_progress,
                        'message': error_message,
                        'error': stderr
                    }, f)
                
                if progress_callback:
                    progress_callback(error_message, last_progress, error=True)
            
            # Comprobar el código de salida
            if process.returncode != 0:
                error_message = f"El proceso de entrenamiento falló con código {process.returncode}"
                with open(log_path, 'w') as f:
                    json.dump({
                        'status': 'error',
                        'progress': last_progress,
                        'message': error_message,
                        'error': f"Return code: {process.returncode}"
                    }, f)
                
                if progress_callback:
                    progress_callback(error_message, last_progress, error=True)
                return False
            
            # Completado con éxito
            if progress_callback:
                progress_callback("Entrenamiento completado con éxito", 100)
            
            # Llamar al callback de finalización si existe
            if on_complete:
                on_complete(True)
            
            return True
            
        except Exception as e:
            error_message = f"Error durante el entrenamiento: {str(e)}"
            with open(log_path, 'w') as f:
                json.dump({
                    'status': 'error',
                    'progress': 0,
                    'message': error_message,
                    'error': str(e)
                }, f)
            
            if progress_callback:
                progress_callback(error_message, 0, error=True)
            
            # Llamar al callback de finalización si existe
            if on_complete:
                on_complete(False)
                
            return False
    
    # Iniciar el entrenamiento en un hilo separado
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    return thread

def get_training_progress():
    """
    Lee el archivo de progreso de entrenamiento
    
    Returns:
        dict: Información del progreso
    """
    log_path = Path('logs/training_progress.json')
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except:
            return {
                'status': 'unknown',
                'progress': 0,
                'message': 'No se pudo leer el archivo de progreso',
                'error': 'Error al leer el archivo'
            }
    else:
        return None

def check_hybrid_models_exist():
    """
    Verifica si los modelos híbridos ya existen
    
    Returns:
        tuple: (bool, list) - Si existen modelos y la lista de modelos encontrados
    """
    models_dir = Path("app/models")
    hybrid_models = []
    
    # Nombres de los modelos híbridos
    expected_models = [
        "efficientnet_resnet_hybrid.keras",
        "efficientnet_cnn_hybrid.keras"
    ]
    
    if models_dir.exists():
        for model_name in expected_models:
            model_path = models_dir / model_name
            if model_path.exists():
                hybrid_models.append(model_path.stem)
    
    return len(hybrid_models) > 0, hybrid_models
