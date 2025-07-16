#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento del sistema
"""

import sys
import os
from pathlib import Path

print("=== Prueba del Sistema de Diagnóstico de Cáncer de Piel ===\n")

# Verificar importaciones básicas
print("1. Verificando importaciones básicas...")
try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except Exception as e:
    print("❌ NumPy:", e)

try:
    import pandas as pd
    print("✅ Pandas:", pd.__version__)
except Exception as e:
    print("❌ Pandas:", e)

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib: disponible")
except Exception as e:
    print("❌ Matplotlib:", e)

try:
    import PIL
    print("✅ PIL: disponible")
except Exception as e:
    print("❌ PIL:", e)

# Verificar TensorFlow
print("\n2. Verificando TensorFlow...")
try:
    import tensorflow as tf
    print("✅ TensorFlow:", tf.__version__)
except Exception as e:
    print("❌ TensorFlow:", e)
    print("   Instalando TensorFlow...")
    os.system("pip install tensorflow==2.18.0")

# Verificar módulos del sistema
print("\n3. Verificando módulos del sistema...")

# Verificar que los archivos existen
required_files = [
    "model_utils.py",
    "model_evaluator.py", 
    "metrics_visualizer.py",
    "pdf_generator.py",
    "preprocessing.py"
]

for file in required_files:
    if Path(file).exists():
        print(f"✅ {file}: existe")
    else:
        print(f"❌ {file}: no encontrado")

# Probar importaciones de módulos
print("\n4. Probando importaciones de módulos...")
try:
    from model_utils import load_models
    print("✅ model_utils.load_models")
except Exception as e:
    print("❌ model_utils.load_models:", e)

try:
    from model_evaluator import ModelEvaluator
    print("✅ model_evaluator.ModelEvaluator")
except Exception as e:
    print("❌ model_evaluator.ModelEvaluator:", e)

try:
    from metrics_visualizer import MetricsVisualizer
    print("✅ metrics_visualizer.MetricsVisualizer")
except Exception as e:
    print("❌ metrics_visualizer.MetricsVisualizer:", e)

try:
    from pdf_generator import PDFReportGenerator
    print("✅ pdf_generator.PDFReportGenerator")
except Exception as e:
    print("❌ pdf_generator.PDFReportGenerator:", e)

try:
    from preprocessing import preprocess_image
    print("✅ preprocessing.preprocess_image")
except Exception as e:
    print("❌ preprocessing.preprocess_image:", e)

# Probar system_manager
print("\n5. Probando system_manager...")
try:
    from system_manager import SkinCancerDiagnosisSystem
    print("✅ system_manager.SkinCancerDiagnosisSystem")
    
    # Probar inicialización
    system = SkinCancerDiagnosisSystem()
    print("✅ Sistema creado")
    
    # Probar configuración
    config = system.config
    print("✅ Configuración cargada")
    
    # Probar inicialización de componentes
    if system.initialize_components():
        print("✅ Componentes inicializados")
        
        # Probar estado del sistema
        status = system.get_system_status()
        print("✅ Estado del sistema obtenido")
        
        print("\n=== Estado del Sistema ===")
        print(f"Modelos cargados: {status.get('models_loaded', 0)}")
        print(f"Métricas en cache: {status.get('metrics_cached', 0)}")
        print(f"Dataset disponible: {status.get('test_dataset_available', False)}")
        
    else:
        print("❌ Error inicializando componentes")
        
except Exception as e:
    print("❌ system_manager:", e)
    import traceback
    traceback.print_exc()

# Verificar directorios
print("\n6. Verificando estructura de directorios...")
directories = [
    "models",
    "cache", 
    "plots",
    "reports",
    "../data/ISIC_dataset_test"
]

for dir_name in directories:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"✅ {dir_name}: existe")
        if dir_name == "models":
            model_files = list(dir_path.glob("*.h5"))
            print(f"   Modelos encontrados: {len(model_files)}")
            for model_file in model_files:
                print(f"     - {model_file.name}")
    else:
        print(f"❌ {dir_name}: no encontrado")

print("\n=== Prueba Completada ===")
