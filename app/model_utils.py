import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import cv2

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

def predict_image(model, image, debug=False, threshold=0.5):
    """Realiza una predicción con el modelo entrenado"""
    if debug:
        print(f"🔍 Debug - Imagen entrada: shape={image.shape}, rango=[{image.min():.3f}, {image.max():.3f}]")
    
    # Verificar si la imagen ya está normalizada (rango 0-1)
    if image.max() > 1.0:
        # Si no está normalizada, normalizar
        if debug:
            print(f"⚠️  Normalizando imagen (rango > 1.0)")
        image = image / 255.0
    elif debug:
        print(f"✅ Imagen ya normalizada")
    
    # Verificar que la imagen tenga la forma correcta para el modelo
    expected_shape = model.input_shape[1:]  # Excluir batch dimension
    if debug:
        print(f"📐 Forma esperada: {expected_shape}, forma actual: {image.shape}")
    
    if image.shape != expected_shape:
        # Redimensionar si es necesario
        if len(expected_shape) == 3:
            if debug:
                print(f"🔄 Redimensionando imagen de {image.shape} a {expected_shape}")
            image = cv2.resize(image, (expected_shape[1], expected_shape[0]))
    
    # Realizar predicción
    if debug:
        print(f"🤖 Realizando predicción...")
    
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
    confidence = float(prediction[0][0])
    
    if debug:
        print(f"📊 Predicción raw: {prediction}, confianza: {confidence:.6f}")
    
    # Interpretar resultado con umbral personalizado
    if confidence > threshold:
        diagnosis = "Maligno"
        confidence_percent = confidence * 100
    else:
        diagnosis = "Benigno"
        confidence_percent = (1 - confidence) * 100
    
    if debug:
        print(f"🏥 Diagnóstico: {diagnosis}, Confianza: {confidence_percent:.1f}% (umbral: {threshold})")
    
    return diagnosis, confidence_percent, confidence

def predict_image_with_debug(model, image):
    """Versión de debug de la función de predicción"""
    return predict_image(model, image, debug=True)

def predict_image_with_custom_threshold(model, image, threshold=0.5):
    """Versión con umbral personalizado"""
    return predict_image(model, image, debug=False, threshold=threshold)

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
