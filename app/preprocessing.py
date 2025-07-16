import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(300, 300)):
    """
    Preprocesa una imagen para la predicción
    
    Args:
        image: PIL Image o numpy array
        target_size: Tamaño objetivo (height, width)
    
    Returns:
        numpy array normalizado
    """
    # Convertir PIL Image a numpy array si es necesario
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convertir a RGB si es necesario
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar
    image = cv2.resize(image, target_size)
    
    # Normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def prepare_batch_images(images, target_size=(300, 300)):
    """
    Preprocesa un lote de imágenes
    
    Args:
        images: Lista de imágenes PIL o numpy arrays
        target_size: Tamaño objetivo
    
    Returns:
        numpy array con el lote procesado
    """
    processed_images = []
    
    for image in images:
        processed_image = preprocess_image(image, target_size)
        processed_images.append(processed_image)
    
    return np.array(processed_images)
