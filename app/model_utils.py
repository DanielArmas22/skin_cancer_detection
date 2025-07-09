import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

def load_models():
    """Carga los modelos entrenados para cáncer de piel"""
    models = {}
    
    # Cargar modelos entrenados desde app/models
    models_dir = Path("app/models")
    if models_dir.exists():
        trained_models = list(models_dir.glob("*.h5"))
        if trained_models:
            print("📁 Cargando modelos entrenados para cáncer de piel...")
            for model_path in trained_models:
                try:
                    model_name = model_path.stem.replace('_', ' ').title()
                    model = load_model(str(model_path))
                    models[model_name] = model
                    print(f"✅ Cargado: {model_name} ({model.count_params():,} parámetros)")
                except Exception as e:
                    print(f"❌ Error cargando {model_path}: {e}")
        else:
            print("❌ No se encontraron modelos entrenados en app/models/")
            print("📝 Asegúrate de que los archivos .h5 estén en la carpeta app/models/")
    else:
        print("❌ No se encontró la carpeta app/models/")
        print("📝 Asegúrate de que los modelos entrenados estén en la ubicación correcta")
    
    return models

def predict_image(model, image):
    """Realiza una predicción con el modelo entrenado"""
    # Normalizar la imagen (0-1)
    image = image / 255.0
    
    # Realizar predicción
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
    confidence = float(prediction[0][0])
    
    # Interpretar resultado
    if confidence > 0.5:
        diagnosis = "Maligno"
        confidence_percent = confidence * 100
    else:
        diagnosis = "Benigno"
        confidence_percent = (1 - confidence) * 100
    
    return diagnosis, confidence_percent, confidence

def get_model_info(model):
    """Obtiene información del modelo"""
    info = {
        'name': model.name,
        'parameters': model.count_params(),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    return info
