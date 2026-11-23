import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. MONTAGE DU GOOGLE DRIVE ET COPIE DU ZIP ---
from google.colab import drive
drive.mount('/content/drive')

# Chemin vers ton dossier IPSA sur ton Google Drive (modifie si le sous-dossier exact diffère)
CHEMIN_ZIP_DRIVE = "/content/drive/MyDrive/IPSA/Vignettes.zip"
DOSSIER_CIBLE = "/content/dataset/Vignettes"

print("📂 Copie et extraction de Vignettes.zip depuis Google Drive...")
os.makedirs("/content/dataset", exist_ok=True)

if os.path.exists(CHEMIN_ZIP_DRIVE):
    shutil.copy(CHEMIN_ZIP_DRIVE, "/content/dataset/Vignettes.zip")
    !unzip -q /content/dataset/Vignettes.zip -d /content/dataset/
    print("✅ Extraction terminée avec succès !")
else:
    print(f"❌ Erreur : Le fichier est introuvable ici : {CHEMIN_ZIP_DRIVE}")
    print("Vérifie l'emplacement exact de ton fichier dans ton dossier IPSA sur Google Drive.")

# --- 2. CONFIGURATION DU DATASET ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_MAX = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER_CIBLE,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER_CIBLE,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n✅ {num_classes} classes chargées : {class_names}\n")

y_train = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
dict_class_weights = dict(enumerate(class_weights))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 3. DATA AUGMENTATION & MODÈLE ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(factor=0.2),
    layers.RandomContrast(factor=0.2),
])

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = True

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint("/content/drive/MyDrive/IPSA/modele_avions_propre.keras", monitor='val_accuracy', save_best_only=True)
]

# --- 4. ENTRAÎNEMENT UNIQUE ---
print("🚀 Lancement de l'apprentissage continu...")
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_MAX,
    class_weight=dict_class_weights, callbacks=callbacks
)

# --- 5. TRACÉ DE LA COURBE PROPRE ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Entraînement', color='#2ca02c', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, val_acc, label='Validation', color='#ff7f0e', linewidth=2, marker='o', markersize=4)
plt.title('Évolution de la Précision (Accuracy)')
plt.xlabel('Époques (Epochs)')
plt.ylabel('Précision')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Entraînement', color='#1f77b4', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, val_loss, label='Validation', color='#d62728', linewidth=2, marker='o', markersize=4)
plt.title('Évolution de l\'Erreur (Loss)')
plt.xlabel('Époques (Epochs)')
plt.ylabel('Erreur (Loss)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()