# Sistema de Diagnóstico de Cáncer de Piel

![Banner](https://via.placeholder.com/800x200/0062cc/ffffff?text=Sistema+de+Diagn%C3%B3stico+de+C%C3%A1ncer+de+Piel)

## 📋 Descripción

Este proyecto implementa una aplicación web basada en modelos de Inteligencia Artificial para el diagnóstico asistido de cáncer de piel. El sistema analiza imágenes dermatoscópicas y clasifica lesiones como benignas o malignas utilizando modelos de aprendizaje profundo.

### 🌟 Características principales

- **Análisis de imágenes** con múltiples modelos de IA (CNN personalizada, ResNet152, EfficientNetB4)
- **Sistema multilenguaje** (Español, Inglés, Francés y Alemán)
- **Generación de informes PDF** con resultados del diagnóstico
- **Visualización avanzada** con mapas de calor de activación
- **Métricas detalladas** de rendimiento de los modelos
- **Interfaz de usuario intuitiva** desarrollada con Streamlit
- **Configuración flexible** de parámetros de diagnóstico

## 🔧 Requisitos previos

### Modelos y datos

> ⚠️ **IMPORTANTE:** Los archivos de modelos y datos no están incluidos en este repositorio debido a su tamaño.

#### Estructura requerida:

```
app/models/
  ├── cnn_personalizada.h5
  ├── efficientnetb4.h5
  └── resnet152.h5

data/
  └── ISIC_dataset/
      ├── benign/
      │   └── [imágenes benignas]
      └── malignant/
          └── [imágenes malignas]
```

### Software requerido

- Python 3.11 o superior
- Docker y Docker Compose (opcional, para despliegue contenerizado)

## 🚀 Instalación

### Opción 1: Instalación directa

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/DanielArmas22/skin_cancer_detection.git
   cd skin_cancer_detection
   ```

2. Crear y activar un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   # En Windows
   venv\Scripts\activate
   # En Linux/Mac
   source venv/bin/activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Descargar y ubicar los modelos entrenados en la carpeta `app/models/`:

   - `cnn_personalizada.h5`
   - `efficientnetb4.h5`
   - `resnet152.h5`

5. (Opcional) Descargar y ubicar los datos en la carpeta `data/`.

### Opción 2: Uso con Docker

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/DanielArmas22/skin_cancer_detection.git
   cd skin_cancer_detection
   ```

2. Descargar y ubicar los modelos entrenados en la carpeta `app/models/`.

3. Construir y ejecutar el contenedor:
   ```bash
   docker-compose up --build
   ```

## 🖥️ Uso

### Ejecución local

1. Ejecutar la aplicación:

   ```bash
   streamlit run app/app.py
   ```

2. Acceder a la aplicación en el navegador: `http://localhost:8501`

### Ejecución con Docker

1. Una vez iniciado el contenedor con `docker-compose up`, acceder a la aplicación en el navegador: `http://localhost:8501`

### Funcionalidades principales

1. **Selección de idioma**: Cambiar el idioma de la interfaz en el menú lateral.
2. **Subida de imágenes**: Cargar una imagen dermatoscópica para análisis.
3. **Selección de modelo**: Elegir entre diferentes modelos de IA para el diagnóstico.
4. **Configuración de umbrales**: Ajustar umbrales de confianza y decisión.
5. **Generación de informes**: Descargar informes PDF con los resultados.

## 📁 Estructura del proyecto

```
skin_cancer_detection/
├── app/
│   ├── models/              # Modelos pre-entrenados (no incluidos en GitHub)
│   ├── translations/        # Archivos de traducción
│   ├── app.py               # Aplicación principal Streamlit
│   ├── model_utils.py       # Utilidades para gestión de modelos
│   └── preprocessing.py     # Funciones de preprocesamiento de imágenes
├── data/                    # Conjunto de datos (no incluido en GitHub)
├── docker-compose.yml       # Configuración de Docker Compose
├── Dockerfile               # Definición de la imagen Docker
└── requirements.txt         # Dependencias del proyecto
```

## 🔄 Obtención de modelos y datos

### Modelos pre-entrenados

Los modelos pre-entrenados se pueden obtener de:

- [Enlace para solicitar los modelos]

Una vez obtenidos, deben ser colocados en la carpeta `app/models/` con los nombres correspondientes:

- `cnn_personalizada.h5`
- `efficientnetb4.h5`
- `resnet152.h5`

### Conjunto de datos

El proyecto utiliza imágenes dermatoscópicas del conjunto de datos ISIC (International Skin Imaging Collaboration). Puede descargar un subconjunto de estas imágenes desde:

- [ISIC Archive](https://challenge.isic-archive.com/data/#2019)

Para utilizar el conjunto de datos con el proyecto, organice las imágenes en la siguiente estructura:

```
data/
└── ISIC_dataset/
    ├── benign/
    │   └── [imágenes benignas]
    └── malignant/
        └── [imágenes malignas]
```

## 📋 Limitaciones y consideraciones

- Esta aplicación está diseñada únicamente como una herramienta de **asistencia al diagnóstico** y no reemplaza la evaluación profesional médica.
- El rendimiento depende de la calidad de las imágenes proporcionadas.
- Se recomienda el uso de imágenes dermatoscópicas de alta calidad para mejores resultados.

## ⚙️ Personalización

### Añadir nuevos modelos

1. Entrene su modelo y guárdelo en formato `.h5`
2. Coloque el archivo en la carpeta `app/models/`
3. El sistema detectará automáticamente el nuevo modelo

### Agregar nuevos idiomas

1. Cree un archivo nuevo en la carpeta `app/translations/` siguiendo el patrón de los archivos existentes
2. Importe y registre el nuevo idioma en `app/translations/__init__.py`

## 🔒 Requisitos del sistema

- **CPU**: Procesador de 4 núcleos o superior recomendado
- **RAM**: 8GB mínimo, 16GB recomendado
- **GPU**: Opcional, pero recomendada para mejor rendimiento
- **Almacenamiento**: Al menos 2GB de espacio libre

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haga un fork del repositorio
2. Cree una rama para su funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Realice sus cambios
4. Envíe un pull request

## 📜 Licencia

Este proyecto está licenciado bajo [MIT License](LICENSE).

## 📞 Contacto

Para consultas o soporte, contactar a:

- [Tu información de contacto]

---

Desarrollado como parte del proyecto de Ingeniería de Software - Universidad [Nombre de tu Universidad]
