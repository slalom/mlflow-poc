## How To Train and Deploy Image Classifier with MLflow and Keras

In this example we demonstrate how to train and deploy image classification models with MLflow.
We train a deep learning model to classify flower species from photos using a `dataset <http://download.tensorflow.org/example_images/flower_photos.tgz>`_ available from `tensorflow.org <http://www.tensorflow.org>`_. Note that although we use Keras to train the model in this case,
a similar approach can be applied to other deep learning frameworks such as `PyTorch`.

In order to include custom image pre-processing logic with the model, we define the model as a
custom python function model wrapping around the underlying Keras model. The wrapper provides
necessary preprocessing to convert input data into multidimensional arrays expected by the
Keras model.

The example contains the following files:

-   MLproject
    Contains definition of this project. Contains only one entry point to train the model.

-   conda.yaml
    Defines project dependencies. NOTE: You might want to change tensorflow package to tensorflow-gpu
    if you have gpu(s) available.

-   train.py
    Main entry point of the projects. Handles command line arguments and possibly downloads the
    dataset. creates, trains and registers the model.

-   score_images.py
    Score an image using a model

Running this Example
^^^^^^^^^^^^^^^^^^^^
Install mlflow

    Pip install mlflow

requires DB as datastore for Model Registry

    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

set
export MLFLOW_TRACKING_URI=http://localhost:5000

To train the model, start the mlfow server and run the example as a standard MLflow project:

the MLflow ui will be at localhost:5000
.. code-block:: bash

    mlflow run ./ --no-conda

This will download the training dataset from "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" if it does not exist, train a classifier using Keras and
log results with Mlflow.

1. Apply the model to new data using the provided score_images.py script:

Note, the model name/path is hardcoded in this example, so keep the name the same, and you must promote the model to Production in the MLFlow ui

http://localhost:5000/#/models

.. code-block:: bash

      # score the model
      python3 ./score_images.py file:///Users/clayton.gibb/path/to/you/image/flower_images_inference/110472418_87b6a3aa98_m.jpg

      #output should be something like:
      This image most likely belongs to dandelion with a 96.68 percent confidence.

here's an issue that came up:

OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

http://127.0.0.1:5000/api/2.0/preview/mlflow/registered-models/get-latest-versions?name=KerasFlowerClassifierModel&stages=Production
