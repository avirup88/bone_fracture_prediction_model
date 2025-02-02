import os

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import traceback as tb
from datetime import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import json

################################################################################
# Thread Configuration
################################################################################

tf.config.threading.set_intra_op_parallelism_threads(0) #Auto-Configure
tf.config.threading.set_inter_op_parallelism_threads(0) #Auto-Configure


################################################################################
# GPU or CPU Detection
################################################################################
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU available. Running on CPU.")
except Exception as e:
    print(f"Error configuring GPU: {e}. Defaulting to CPU.")

################################################################################
# Function to remove corrupted images
################################################################################
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

################################################################################
# Function for data augmentation
################################################################################
def fn_data_augmentation():
    """
    Returns a list of data augmentation layers to apply to the dataset.
    """
    return [
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomFlip('vertical'),
        tf.keras.layers.RandomRotation(0.10),  # Rotate by up to ±10%
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        tf.keras.layers.RandomBrightness(factor=(-0.1, 0.1)),  # Adjust brightness by ±10%
        tf.keras.layers.RandomContrast(factor=0.2),  # Adjust contrast by ±20%
        tf.keras.layers.RandomTranslation(height_factor=0.5, width_factor=0.5)
    ]

################################################################################
# Main Function
################################################################################
if __name__ == '__main__':
    try:
        start_time = dt.now()
        print(f"Script Start Time: {start_time}")

        # Define paths
        dataset_path = './datasets'
        model_path = './model_files'

        train_dir = os.path.join(dataset_path, 'train')
        validation_dir = os.path.join(dataset_path, 'val')
        test_dir = os.path.join(dataset_path, 'test')

        # Remove corrupted images
        fn_remove_corrupted_images_tf(train_dir)
        fn_remove_corrupted_images_tf(validation_dir)
        fn_remove_corrupted_images_tf(test_dir)

        # Define dataset properties
        batch_size = 32
        img_size = (224, 224)
        img_shape = img_size + (3,)

        # Load datasets
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

        # Data augmentation
        data_augmentation = tf.keras.Sequential(fn_data_augmentation())

        # Optimize dataset performance
        autotune = tf.data.AUTOTUNE
        train_dataset = train_dataset.prefetch(buffer_size=autotune)
        validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
        test_dataset = test_dataset.prefetch(buffer_size=autotune)

        ################################################################################
        # Model Definition
        ################################################################################
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet'
        )

        # Freeze some layers in the base model
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # Define model architecture
        inputs = tf.keras.Input(shape=img_shape)
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)

        # Compile the model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(model_path, 'bone_fracture_model.keras'),
            monitor='val_accuracy', save_best_only=True, mode='max'
        )

        ################################################################################
        # Model Training
        ################################################################################
        model_logs = model.fit(
            train_dataset,
            epochs=100,
            validation_data=validation_dataset,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Save training history
        with open(os.path.join(model_path, 'bone_fracture_model_logs.json'), 'w') as file:
            json.dump(model_logs.history, file)

        ################################################################################
        # Model Evaluation
        ################################################################################

        # Generate classification report
        ground_truth = np.concatenate([labels.numpy() for _, labels in test_dataset])
        predictions = np.concatenate([
            tf.where(model(X).numpy().flatten() < 0.5, 0, 1).numpy()
            for X, _ in test_dataset
        ])
        print("Classification Report:")
        print(classification_report(ground_truth, predictions, target_names=class_names))

        print("Model Testing Completed successfully.")

        # Script execution time
        end_time = dt.now()
        print(f"Script Completed: {end_time}")
        duration = end_time - start_time
        print(f"Execution Time: {duration.total_seconds() / 60:.2f} minutes")

    except Exception as e:
        error = f"Script Failure: {e}\nTraceback: {tb.format_exc()}"
        print(error)
