import os
import mlflow
import numpy as np
import warnings
import mlflow.keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
warnings.filterwarnings('ignore')

def load_data(train_dir, test_dir, img_height=150, img_width=150, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, test_generator

def visualize_samples(generator, num_samples=5):
    class_labels = list(generator.class_indices.keys())
    num_samples_per_row = 3
    num_rows = (num_samples + num_samples_per_row - 1) // num_samples_per_row
    fig, axes = plt.subplots(num_rows, num_samples_per_row, figsize=(10, 3*num_rows))
    for i in range(num_rows):
        for j in range(num_samples_per_row):
            if (i * num_samples_per_row + j) < num_samples:
                batch = next(generator)
                axes[i, j].imshow(batch[0][j])
                axes[i, j].set_title(class_labels[int(batch[1][j])])
                axes[i, j].axis('off')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_aspect('auto')
                axes[i, j].margins(0.05)
    plt.tight_layout()
    plt.show()

def build_model(input_shape=(150, 150, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

def train_model(model, train_generator, test_generator, epochs=10):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": train_generator.batch_size,
            "num_train_samples": train_generator.samples,
            "num_test_samples": test_generator.samples
        })

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=test_generator.samples // test_generator.batch_size
        )

        # Log metrics
        for key, value in history.history.items():
            mlflow.log_metric(key, value[-1])  # Log only the last value of each metric

        # Save model
        model.save("model.h5")
        mlflow.log_artifact("model.h5")

        return history

def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print('\nTest accuracy:', test_acc)
    mlflow.log_metric("test_accuracy", test_acc)


