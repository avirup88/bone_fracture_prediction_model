import os
import logging
import warnings

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import traceback as tb
from datetime import datetime as dt
from joblib import dump, load
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is configured successfully.")
    except RuntimeError as e:
        print(f"Error in GPU configuration: {e}")

# Enable mixed precision for GPUs
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print(f"Mixed precision policy: {policy}")

################################################################################
# Rest of the code (unchanged except for optimizer updates to support mixed precision)
################################################################################
# Function to remove corrupted images
def fn_remove_corrupted_images_tf(dataset_dir):
    num_deleted = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img_raw = tf.io.read_file(file_path)
                img = tf.image.decode_image(img_raw)
            except tf.errors.InvalidArgumentError:
                os.remove(file_path)
                num_deleted += 1
    print(f"Total corrupted images removed (TensorFlow check): {num_deleted}")

def fn_data_augmentation():
    return [
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        tf.keras.layers.RandomBrightness(factor=(-0.1, 0.1)),
        tf.keras.layers.RandomContrast(factor=0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.5, width_factor=0.5)
    ]

if __name__ == '__main__':
    try:
        start_time = dt.now()
        print(f"Script Start Time: {start_time}")

        dataset_path = './datasets'
        model_path = './model_files'

        train_dir = os.path.join(dataset_path, 'train')
        validation_dir = os.path.join(dataset_path, 'val')
        test_dir = os.path.join(dataset_path, 'test')

        fn_remove_corrupted_images_tf(train_dir)
        fn_remove_corrupted_images_tf(validation_dir)
        fn_remove_corrupted_images_tf(test_dir)

        batch_size = 32
        img_size = (224, 224)
        img_shape = img_size + (3,)

        train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir, shuffle=True, batch_size=batch_size, image_size=img_size
        )
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            validation_dir, shuffle=True, batch_size=batch_size, image_size=img_size
        )
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dir, shuffle=True, batch_size=batch_size, image_size=img_size
        )

        class_names = train_dataset.class_names
        print("Class_Names :", class_names)

        data_augmentation = tf.keras.Sequential(fn_data_augmentation())

        autotune = tf.data.AUTOTUNE
        train_dataset = train_dataset.prefetch(buffer_size=autotune)
        validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
        test_dataset = test_dataset.prefetch(buffer_size=autotune)

        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = True
        for layer in base_model.layers[:50]:
            layer.trainable = False

        inputs = tf.keras.Input(shape=img_shape)
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
        model = tf.keras.Model(inputs, outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=lr_schedule))

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        fine_tune_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        fine_tune_model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(model_path, 'bone_fracture_model.keras'),
            monitor='val_accuracy', save_best_only=True, mode='max'
        )

        model_logs = model.fit(
            train_dataset,
            epochs=100,
            validation_data=validation_dataset,
            callbacks=[fine_tune_early_stopping, fine_tune_model_checkpoint],
            verbose=1
        )

        with open(os.path.join(model_path, 'bone_fracture_model_logs.json'), 'w') as file:
            json.dump(model_logs.history, file)

        test_loss, test_accuracy = model.evaluate(test_dataset)
        print("Model Testing Completed successfully.")

        end_time = dt.now()
        duration = end_time - start_time
        print(f"Execution Time: {duration.total_seconds() / 60:.2f} minutes")

    except Exception as e:
        error = f"Script Failure: {e}\nTraceback: {tb.format_exc()}"
        print(error)
