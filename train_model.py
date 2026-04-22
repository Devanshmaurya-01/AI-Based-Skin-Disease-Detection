import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # 🔥 NEW
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   # 🔥 NEW
from sklearn.utils import class_weight
import numpy as np
import json
import os
import matplotlib.pyplot as plt   #  NEW

train_dir = "dataset/train"

#  DATA AUGMENTATION (IMPROVED)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # 🔥 better than rescale
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

#  LOAD DATA
train = datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

#  CHECK CLASSES
print("Detected Classes:", train.class_indices)

num_classes = len(train.class_indices)

#  SAVE CLASS MAPPING
os.makedirs("model", exist_ok=True)
with open("model/classes.json", "w") as f:
    json.dump(train.class_indices, f)

#  CLASS WEIGHTS
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train.classes),
    y=train.classes
)
class_weights = dict(enumerate(weights))

#  BASE MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False)

#  PARTIAL TRAINING
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

#  CUSTOM HEAD
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)   #  NEW
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

#  COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#  CALLBACKS (PRO LEVEL )
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("model/best_model.h5", save_best_only=True)
]

#  TRAIN
history = model.fit(
    train,
    validation_data=val,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)

#  SAVE FINAL MODEL
model.save("model/skin_model.h5")

print(" TRAINING COMPLETE - PRO MODEL READY")

#  SAVE TRAINING GRAPH
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig("model/accuracy.png")