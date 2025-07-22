import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.applications import EfficientNetB4, ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import cv2
import random
import json
import datetime
import zipfile
import shutil

print("🔧 TensorFlow version:", tf.__version__)
print("🖥️ GPU disponible:", tf.config.list_physical_devices('GPU'))

# Configuraciones
SEED = 42
IMG_SIZE = 300
BATCH_SIZE = 32
BASE_DIR = Path('data')
MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')
PLOTS_DIR = Path('plots')

# Crear directorios necesarios
for directory in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# Semilla para reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Funciones de preprocesamiento con CLAHE obligatorio
def apply_clahe(image):
    """Aplica CLAHE a una imagen RGB"""
    # Convertir a LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Separar canales
    l, a, b = cv2.split(lab)
    # Aplicar CLAHE al canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    # Combinar canales
    lab = cv2.merge((l, a, b))
    # Convertir de vuelta a RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced

def preprocess_image_for_segmentation(image):
    """Preprocesa imagen para segmentación"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Ecualización con CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return clahe_img

def segment_roi(image):
    """Segmenta la región de interés en la imagen de lesión de piel"""
    # Preprocesar para segmentación
    gray = preprocess_image_for_segmentation(image)
    
    # Umbralización con Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # Si no hay contornos, devolver la imagen original
    
    # Obtener el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Crear máscara
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Aplicar máscara a la imagen original
    result = image.copy()
    result[mask == 0] = 0
    
    # Recortar la región de interés
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Añadir margen
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    roi = result[y:y+h, x:x+w]
    
    # Si el ROI es muy pequeño, usar la imagen original
    if roi.size < image.size * 0.1:
        return image
    
    return roi

class CustomDataGenerator(tf.keras.utils.Sequence):
    """Generador personalizado con preprocesamiento avanzado"""
    def __init__(self, directory, batch_size=32, target_size=(300, 300), 
                 shuffle=True, augment=False, class_mode='binary'):
        self.datagen = ImageDataGenerator(
            rescale=1./255
        )
        self.generator = self.datagen.flow_from_directory(
            directory, 
            target_size=target_size, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            class_mode=class_mode
        )
        self.batch_size = batch_size
        self.augment = augment
        self.n = len(self.generator)
        self.classes = self.generator.classes
        self.class_indices = self.generator.class_indices
        self.samples = self.generator.samples
        
    def __len__(self):
        return self.n
        
    def __getitem__(self, idx):
        x_batch, y_batch = self.generator.__getitem__(idx)
        
        # Aplicar preprocesamiento avanzado a cada imagen
        processed_batch = np.zeros_like(x_batch)
        for i, img in enumerate(x_batch):
            # Convertir a formato OpenCV (0-255, uint8)
            img_cv = (img * 255).astype(np.uint8)
            
            # Aplicar CLAHE
            img_cv = apply_clahe(img_cv)
            
            # Segmentación ROI (opcional)
            if random.random() < 0.7:  # Aplicar a 70% de las muestras
                img_cv = segment_roi(img_cv)
            
            # Augmentaciones adicionales durante entrenamiento
            if self.augment:
                # Rotación aleatoria
                if random.random() < 0.5:
                    angle = random.uniform(-15, 15)
                    h, w = img_cv.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    img_cv = cv2.warpAffine(img_cv, M, (w, h))
                
                # Flip horizontal
                if random.random() < 0.5:
                    img_cv = cv2.flip(img_cv, 1)
                
                # Ajustes de brillo/contraste
                if random.random() < 0.5:
                    alpha = random.uniform(0.8, 1.2)  # Contraste
                    beta = random.uniform(-10, 10)    # Brillo
                    img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)
            
            # Normalizar a [0-1]
            img_cv = img_cv.astype(np.float32) / 255.0
            processed_batch[i] = img_cv
            
        return processed_batch, y_batch

# Funciones para crear los modelos híbridos
def create_hybrid_efficientnet_resnet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Crea un modelo híbrido que combina EfficientNetB4 y ResNet152
    """
    # EfficientNetB4 como base principal
    efficientnet_base = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # ResNet152 como extractor de características complementario
    resnet_base = ResNet152(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar capas iniciales de ResNet (más pesado)
    for layer in resnet_base.layers[:100]:
        layer.trainable = False
    
    # Entrada compartida
    input_tensor = Input(shape=input_shape)
    
    # Procesar la entrada con ambos modelos
    efficientnet_features = efficientnet_base(input_tensor)
    resnet_features = resnet_base(input_tensor)
    
    # Pooling global para cada rama
    efficientnet_pooled = GlobalAveragePooling2D()(efficientnet_features)
    resnet_pooled = GlobalAveragePooling2D()(resnet_features)
    
    # Concatenar características
    merged_features = concatenate([efficientnet_pooled, resnet_pooled])
    
    # Capas densas para clasificación
    x = Dense(1024, activation='relu')(merged_features)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Capa de salida para clasificación binaria
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Crear y compilar modelo
    model = Model(inputs=input_tensor, outputs=outputs, name="EfficientNet_ResNet_Hybrid")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )
    
    return model

def create_hybrid_efficientnet_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Crea un modelo híbrido que combina EfficientNetB4 con una CNN personalizada
    """
    # EfficientNetB4 como base
    efficientnet_base = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Entrada compartida
    input_tensor = Input(shape=input_shape)
    
    # Rama 1: EfficientNetB4
    efficientnet_features = efficientnet_base(input_tensor)
    efficientnet_pooled = GlobalAveragePooling2D()(efficientnet_features)
    
    # Rama 2: CNN Personalizada con atención
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Mecanismo de atención
    attention = layers.Conv2D(128, (1, 1), activation='relu')(x)
    attention = layers.GlobalAveragePooling2D()(attention)
    attention = layers.Reshape((1, 1, 128))(attention)
    attention = layers.Conv2D(256, (1, 1), activation='sigmoid')(attention)
    
    # Aplicar atención
    x = layers.Multiply()([x, attention])
    cnn_pooled = GlobalAveragePooling2D()(x)
    
    # Concatenar características de ambas ramas
    merged_features = concatenate([efficientnet_pooled, cnn_pooled])
    
    # Capas densas para clasificación
    x = Dense(1024, activation='relu')(merged_features)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Capa de salida para clasificación binaria
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Crear y compilar modelo
    model = Model(inputs=input_tensor, outputs=outputs, name="EfficientNet_CNN_Hybrid")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )
    
    return model

def train_and_evaluate_model(model, train_gen, val_gen, class_weights, model_name, epochs=20):
    """
    Entrena y evalúa un modelo
    """
    print(f"\n{'='*60}")
    print(f"🚀 Entrenando modelo híbrido: {model_name}")
    print(f"{'='*60}")
    
    # Callbacks
    checkpoint_path = f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}.h5"
    log_path = f"{REPORTS_DIR}/{model_name.lower().replace(' ', '_')}_log.csv"
    
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(log_path)
    ]
    
    # Entrenamiento
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Cargar mejor modelo guardado
    model = load_model(checkpoint_path)
    
    # Evaluación
    val_results = model.evaluate(val_gen, verbose=1)
    metrics_dict = {name: float(value) for name, value in zip(model.metrics_names, val_results)}
    
    print(f"\n✅ Resultados finales de {model_name}:")
    for metric_name, metric_value in metrics_dict.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    # Generar gráficos de entrenamiento
    plot_training_history(history, model_name)
    
    # Generar matriz de confusión y métricas
    generate_evaluation_metrics(model, val_gen, model_name)
    
    # Guardar resultados en JSON
    save_model_results(model_name, metrics_dict, history.history)
    
    # Comprimir modelo para facilitar su distribución
    zip_file = f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}.zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(checkpoint_path, os.path.basename(checkpoint_path))
    
    print(f"📦 Modelo comprimido guardado en: {zip_file}")
    
    return history, metrics_dict

def plot_training_history(history, model_name):
    """
    Genera gráficas del historial de entrenamiento
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name}: Pérdida durante entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfica de accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name}: Precisión durante entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{model_name.lower().replace(' ', '_')}_training.png")
    plt.close()
    
    # Gráfica de métricas adicionales
    plt.figure(figsize=(12, 5))
    
    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['Precision'], label='Train Precision')
    plt.plot(history.history['val_Precision'], label='Validation Precision')
    plt.title(f'{model_name}: Precisión durante entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfica de recall
    plt.subplot(1, 2, 2)
    plt.plot(history.history['Recall'], label='Train Recall')
    plt.plot(history.history['val_Recall'], label='Validation Recall')
    plt.title(f'{model_name}: Recall durante entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{model_name.lower().replace(' ', '_')}_metrics.png")
    plt.close()

def generate_evaluation_metrics(model, val_gen, model_name):
    """
    Genera matriz de confusión y métricas adicionales
    """
    # Predecir en el conjunto de validación
    val_gen.reset()
    y_true = val_gen.classes
    
    # Obtener probabilidades predichas
    y_prob = model.predict(val_gen, steps=np.ceil(val_gen.samples/val_gen.batch_size))
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Benigno', 'Maligno'],
               yticklabels=['Benigno', 'Maligno'])
    plt.title(f'{model_name}: Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{model_name.lower().replace(' ', '_')}_confusion.png")
    plt.close()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'{model_name}: Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{PLOTS_DIR}/{model_name.lower().replace(' ', '_')}_roc.png")
    plt.close()
    
    # Informe de clasificación
    report = classification_report(y_true, y_pred, target_names=['Benigno', 'Maligno'], output_dict=True)
    
    # Guardar resultados en JSON
    with open(f"{REPORTS_DIR}/{model_name.lower().replace(' ', '_')}_classification_report.json", 'w') as f:
        json.dump(report, f, indent=4)

def save_model_results(model_name, metrics, history):
    """
    Guarda los resultados del modelo en JSON
    """
    results = {
        "model_name": model_name,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_metrics": metrics,
        "training_history": {
            k: [float(val) for val in v] for k, v in history.items()
        }
    }
    
    with open(f"{REPORTS_DIR}/{model_name.lower().replace(' ', '_')}_results.json", 'w') as f:
        json.dump(results, f, indent=4)

def main():
    """
    Función principal para entrenar modelos híbridos
    """
    print("🌟 Iniciando entrenamiento de modelos híbridos para detección de cáncer de piel")
    
    # Preparar datos
    print("\n📊 Preparando datos...")
    
    train_dir = Path('data/ISIC_dataset')
    test_dir = Path('data/ISIC_dataset_test')
    
    if not train_dir.exists() or not test_dir.exists():
        print("❌ Error: Directorios de datos no encontrados. Asegúrate de haber ejecutado la organización del dataset.")
        return
    
    # Crear generadores personalizados
    print("🔍 Creando generadores de datos con preprocesamiento avanzado...")
    
    train_gen = CustomDataGenerator(
        str(train_dir),
        batch_size=BATCH_SIZE,
        target_size=(IMG_SIZE, IMG_SIZE),
        augment=True
    )
    
    val_gen = CustomDataGenerator(
        str(test_dir),
        batch_size=BATCH_SIZE,
        target_size=(IMG_SIZE, IMG_SIZE),
        augment=False,
        shuffle=False
    )
    
    print(f"✅ Datos preparados:")
    print(f"   Train: {train_gen.samples} muestras")
    print(f"   Validation: {val_gen.samples} muestras")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Calcular pesos de clases
    labels = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights_array))
    print(f"📊 Pesos de clases: {class_weights}")
    
    # Crear y entrenar modelo híbrido 1: EfficientNetB4 + ResNet152
    print("\n🚀 Creando modelo híbrido 1: EfficientNetB4 + ResNet152")
    model1 = create_hybrid_efficientnet_resnet()
    model1.summary()
    history1, metrics1 = train_and_evaluate_model(
        model1, 
        train_gen, 
        val_gen, 
        class_weights, 
        "EfficientNet_ResNet_Hybrid", 
        epochs=25
    )
    
    # Crear y entrenar modelo híbrido 2: EfficientNetB4 + CNN Personalizada
    print("\n🚀 Creando modelo híbrido 2: EfficientNetB4 + CNN Personalizada con Atención")
    model2 = create_hybrid_efficientnet_cnn()
    model2.summary()
    history2, metrics2 = train_and_evaluate_model(
        model2, 
        train_gen, 
        val_gen, 
        class_weights, 
        "EfficientNet_CNN_Hybrid", 
        epochs=25
    )
    
    # Comparar modelos
    models_comparison = {
        "EfficientNet_ResNet_Hybrid": metrics1,
        "EfficientNet_CNN_Hybrid": metrics2
    }
    
    with open(f"{REPORTS_DIR}/modelos_hibridos_comparacion.json", 'w') as f:
        json.dump(models_comparison, f, indent=4)
    
    print("\n✅ Entrenamiento de modelos híbridos completado.")
    print(f"📊 Reportes guardados en: {REPORTS_DIR}")
    print(f"📈 Gráficos guardados en: {PLOTS_DIR}")
    print(f"💾 Modelos guardados en: {MODELS_DIR}")

if __name__ == "__main__":
    main()
