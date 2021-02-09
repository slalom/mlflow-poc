import math
import numpy as np
import os
import click
#note: must use nightly build - pip install tf-nightly
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

import mlflow

@click.command(
    help="Trains an Keras model on flower_photos dataset."
    "The input is expected as a directory tree with pictures for each category in a"
    " folder named by the category."
    "The model and its metrics are logged with mlflow."
)
@click.option("--epochs", type=click.INT, default=1, help="Maximum number of epochs to evaluate.")
@click.option(
    "--batch-size", type=click.INT, default=16, help="Batch size passed to the learning algo."
)
@click.option("--image-width", type=click.INT, default=224, help="Input image width in pixels.")
@click.option("--image-height", type=click.INT, default=224, help="Input image height in pixels.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.option("--training-data", type=click.STRING, default="./flower_photos")
@click.option("--test-ratio", type=click.FLOAT, default=0.2)
def run(training_data, test_ratio, epochs, batch_size, image_width, image_height, seed):
    mlflow.set_tracking_uri("http://localhost:5000")
    print("tf.__version__ " + tf.__version__)
    print("Training model with the following parameters:")
    for param, value in locals().items():
        print("  ", param, "=", value)  

    data_dir = pathlib.Path(training_data)
    image_count = len(list(data_dir.glob('./flower_photos/*.jpg')))
    print("image count: " + str(image_count))

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=test_ratio,
        subset="training",
        seed=seed,
        image_size=(image_height, image_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=test_ratio,
        subset="validation",
        seed=seed,
        image_size=(image_height, image_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)
    #configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = _create_model(classes=len(class_names))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary()
    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        MLflowLogger(
            model=model,
            x_train=train_ds,
            x_valid=val_ds,
            artifact_path="model"
        )]
    )

class MLflowLogger(Callback):
    """
    Keras callback for logging metrics and final model with MLflow.

    Metrics are logged after every epoch. The logger keeps track of the best model based on the
    validation metric. At the end of the training, the best model is logged with MLflow.
    """
    def __init__(self, model, x_train, x_valid, **kwargs):
        self._model = model
        self._best_val_loss = math.inf
        self._train = (x_train)
        self._valid = (x_valid)
        self._pyfunc_params = kwargs
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. Update the best model if the model improved on the validation
        data.
        """
        print("on_epoch_end")
        if not logs:
            print("no logs for epoch")
            return
        for name, value in logs.items():
            if name.startswith("val_"):
                name = "valid_" + name[4:]
            else:
                name = "train_" + name
            mlflow.log_metric(name, value)
        val_loss = logs["val_loss"]
        if val_loss < self._best_val_loss:
            # Save the "best" weights
            self._best_val_loss = val_loss
            self._best_weights = [x.copy() for x in self._model.get_weights()]

    def on_train_end(self, *args, **kwargs):
        """
        Log the best model with MLflow and evaluate it on the train and validation data so that the
        metrics stored with MLflow reflect the logged model.
        """
        self._model.set_weights(self._best_weights)
        x = self._train
        train_res = self._model.evaluate(x=x)
        for name, value in zip(self._model.metrics_names, train_res):
            mlflow.log_metric("train_{}".format(name), value)
        x = self._valid
        valid_res = self._model.evaluate(x=x)
        for name, value in zip(self._model.metrics_names, valid_res):
            mlflow.log_metric("valid_{}".format(name), value)
        mlflow.keras.log_model(self._model,**self._pyfunc_params)
        run = mlflow.active_run()
        model_uri = "runs:/{}/model".format(run.info.run_id)
        print("model_uri: {}".format(model_uri))
        mv = mlflow.register_model(model_uri, "KerasFlowerClassifierModel")
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))

def _create_model(classes):
    print("create model")
    num_classes = classes

    #data augmentation
    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                    input_shape=(180, 
                                                                180,
                                                                3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
    )

    model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='softmax'),
    layers.Dense(num_classes)
    ])
    return model

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", str(epochs))
        mlflow.log_param("batch_size", str(batch_size))
        mlflow.log_param("validation_ratio", str(test_ratio))

if __name__ == "__main__":
    run()
