"""
Example of scoring images with MLflow model deployed to a REST API endpoint.

The MLflow model to be scored is expected to be an instance of KerasImageClassifierPyfunc
(e.g. produced by running this project) and deployed with MLflow prior to invoking this script.
"""
import os
import base64
from urllib.parse import urlparse
import requests
import json

import click
import pandas as pd

import mlflow

import PIL

import tensorflow as tf

import numpy as np

from tensorflow import keras

from mlflow.utils import cli_args


def score_model(path, host, port):
    """
    Score images on the local path with MLflow model deployed at given uri and port.

    :param path: Path to a single image file or a directory of images.
    :param host: host the model is deployed at
    :param port: Port the model is deployed at.
    :return: Server response.
    """
    filename = os.path.basename(path)
    flower_path = tf.keras.utils.get_file(filename, origin=path)

    img = keras.preprocessing.image.load_img(
        flower_path, target_size=(180, 180)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    model = mlflow.keras.load_model("models:/KerasFlowerClassifierModel/Production")
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return predictions



@click.command(help="Score images.")
@click.option("--port", type=click.INT, default=80, help="Port at which the model is deployed.")
@cli_args.HOST
@click.argument("data-path")
def run(data_path, host, port):
    """
    Score images with MLflow deployed deployed at given uri and port and print out the response
    to standard out.
    """
    print(score_model(data_path, host, port))


if __name__ == "__main__":
    run()
