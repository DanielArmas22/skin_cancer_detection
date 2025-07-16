#!/usr/bin/env python3
"""
Script de inicialización para el Sistema de Diagnóstico de Cáncer de Piel v2.0
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Función principal para inicializar el sistema"""
    
    print("🩺 Sistema de Diagnóstico de Cáncer de Piel v2.0")
    print("=" * 50)
    
    # Verificar que estamos en el directorio correcto
    app_dir = Path("app")
    if not app_dir.exists():
        print("❌ Error: No se encontró el directorio 'app'")
        print("📍 Ejecute este script desde el directorio raíz del proyecto")
        return False
    
    # Verificar archivos principales
    required_files = [
        "app/app.py",
        "app/system_manager.py",
        "app/model_evaluator.py",
        "app/metrics_visualizer.py",
        "app/pdf_generator.py",
        "app/model_utils.py",
        "app/preprocessing.py"
    ]
    
    print("\n🔍 Verificando archivos del sistema...")
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - No encontrado")
            return False
    
    # Verificar modelos
    models_dir = Path("app/models")
    if models_dir.exists():
        models = list(models_dir.glob("*.h5"))
        print(f"\n🤖 Modelos encontrados: {len(models)}")
        for model in models:
            print(f"   - {model.name}")
    else:
        print("❌ No se encontró el directorio de modelos")
        return False
    
    # Verificar dataset
    dataset_dir = Path("data/ISIC_dataset_test")
    if dataset_dir.exists():
        print("✅ Dataset de test encontrado")
    else:
        print("❌ Dataset de test no encontrado")
        return False
    
    # Cambiar al directorio de la aplicación
    os.chdir("app")
    
    # Ejecutar pruebas del sistema
    print("\n🧪 Ejecutando pruebas del sistema...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Pruebas del sistema completadas exitosamente")
        else:
            print("❌ Error en las pruebas del sistema")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error ejecutando pruebas: {e}")
        return False
    
    # Inicializar aplicación Streamlit
    print("\n🚀 Iniciando aplicación Streamlit...")
    print("   URL: http://localhost:8502")
    print("   Para detener: Ctrl+C")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Sistema detenido por el usuario")
        return True
    except Exception as e:
        print(f"❌ Error iniciando Streamlit: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
