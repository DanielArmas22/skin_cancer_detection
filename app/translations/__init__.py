"""
Este archivo proporciona funciones para manejar traducciones en la aplicación de detección de cáncer de piel.
"""

def get_available_languages():
    """
    Devuelve un diccionario con los idiomas disponibles y sus códigos.
    """
    return {
        'Español': 'es',
        'English': 'en',
        'Français': 'fr',
        'Deutsch': 'de',
    }

def load_translations(lang_code):
    """
    Carga las traducciones para el idioma especificado.
    
    Args:
        lang_code (str): Código del idioma (es, en, fr, de)
        
    Returns:
        dict: Diccionario con las traducciones
    """
    try:
        if lang_code == 'es':
            from translations.es import translations
        elif lang_code == 'en':
            from translations.en import translations
        elif lang_code == 'fr':
            from translations.fr import translations
        elif lang_code == 'de':
            from translations.de import translations
        else:
            # Usar español como idioma por defecto
            from translations.es import translations
            
        return translations
    except Exception as e:
        # En caso de error, devolver un diccionario vacío
        print(f"Error cargando traducciones: {str(e)}")
        return {}
