"""
Este archivo contiene funciones de utilidad para manejar errores comunes relacionados con traducciones.
"""

def get_error_message(key, translations_dict, default_message="Error desconocido"):
    """
    Obtiene un mensaje de error traducido.
    
    Args:
        key: Clave del mensaje en el diccionario de traducciones
        translations_dict: Diccionario de traducciones
        default_message: Mensaje por defecto si no se encuentra la clave
        
    Returns:
        str: Mensaje de error traducido
    """
    if key in translations_dict:
        return translations_dict[key]
    return default_message
