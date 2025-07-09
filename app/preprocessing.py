import cv2
import numpy as np

def preprocess_image(image, target_size=(300, 300)):
    """Preprocesa la imagen para el modelo"""
    # Convertir a RGB si es en escala de grises
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Redimensionar manteniendo relación de aspecto
    h, w = image.shape[:2]
    if h != target_size[0] or w != target_size[1]:
        interpolation = cv2.INTER_AREA if h > target_size[0] or w > target_size[1] else cv2.INTER_CUBIC
        image = cv2.resize(image, target_size, interpolation=interpolation)
    
    # Mejorar contraste (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalización
    image = image.astype(np.float32) / 255.0
    
    return image
