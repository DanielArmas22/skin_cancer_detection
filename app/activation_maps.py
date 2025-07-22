# activation_maps.py
"""
Módulo para generación de mapas de activación (Grad-CAM) para visualización
de regiones de interés en imágenes de cáncer de piel
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
import streamlit as st

def generate_gradcam(model, image, alpha=0.4):
    """
    Genera un mapa de calor de Grad-CAM para visualizar áreas importantes
    en la toma de decisiones del modelo.
    
    Args:
        model: Modelo de TensorFlow/Keras
        image: Imagen preprocesada (tensor)
        alpha: Factor de mezcla para superposición (0-1)
        
    Returns:
        Imagen con superposición del mapa de calor o None si hay error
    """
    try:
        # Asegurarse de que la imagen tenga la forma correcta (batch_size, height, width, channels)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Obtener la última capa convolucional
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            st.warning("No se pudo encontrar una capa convolucional adecuada para el mapa de activación")
            return None

        # Crear modelo que mapea la entrada a la salida de la última capa convolucional
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        # Calcular gradientes
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image)
            loss = predictions[:, tf.argmax(predictions[0])]

        # Calcular gradientes
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Crear mapa de calor
        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Redimensionar el mapa de calor al tamaño de la imagen original
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Obtener la imagen original y asegurarse de que esté en el formato correcto
        original_image = image[0]
        original_image = np.uint8(original_image * 255)  # Convertir a rango 0-255
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # Asegurarse de que ambas imágenes tengan el mismo tipo de datos
        heatmap = np.float32(heatmap)
        original_image = np.float32(original_image)

        # Superposición del mapa de calor sobre la imagen original
        superimposed_img = cv2.addWeighted(
            original_image, 1 - alpha, 
            heatmap, alpha, 
            0
        )

        # Convertir de vuelta a RGB para mostrar en Streamlit
        return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        st.error(f"Error al generar el mapa de activación Grad-CAM: {e}")
        return None
