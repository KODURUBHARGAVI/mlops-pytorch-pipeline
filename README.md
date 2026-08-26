# mlops-pytorch-pipeline

This project trains an image classifier using PyTorch and serves it through a
web API. An image can be sent to the API, and it returns the predicted class.

The project runs on a normal laptop CPU. A GPU is not required.

The dataset is Fashion-MNIST. It contains 70,000 grey images of clothing
items, sized 28 by 28, divided into 10 classes such as T-shirt, Trouser,
Sneaker and Bag.

---

## Requirements

Python 3.11 is used here. Versions 3.10 and 3.12 also work.

To check the installed version:

    python --version

---

## Step 1: Create a virtual environment

A virtual environment keeps the packages for this project separate from the
rest of the system.

On Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\activate

On Mac or Linux:

    python -m venv .venv
    source .venv/bin/activate

Once it is active, the prompt shows (.venv) at the beginning.

---

## Step 2: Install the packages

    pip install -r requirements/dev.txt

The first install takes a few minutes because PyTorch is a large download.

---

## Step 3: Run the tests

The tests confirm that the model, the settings file and the API all behave as
expected.

    pytest tests -v

All tests should pass in under a minute.

---

## Step 4: Short training run

This run uses 2 epochs and 15 percent of the data. It is meant as a quick
check that the training loop works, and it finishes in under a minute.

On Windows PowerShell:

    $env:EPOCHS="2"
    $env:SUBSET_FRACTION="0.15"
    python src\train.py

On Mac or Linux:

    EPOCHS=2 SUBSET_FRACTION=0.15 python src/train.py

On the first run, the dataset is downloaded into a folder named data. This is
about 30 MB. Later runs reuse the same copy.

---

## Step 5: Full training run

On Windows, the two settings from Step 4 stay active for the rest of the
terminal session. They must be cleared first, otherwise the short run is
repeated.

    Remove-Item Env:EPOCHS, Env:SUBSET_FRACTION
    $env:TORCH_NUM_THREADS="8"
    python src\train.py

On Mac or Linux, open a new terminal or run:

    TORCH_NUM_THREADS=8 python src/train.py

The full run takes about 10 to 15 minutes on 8 CPU cores and reaches around
91 percent accuracy.

Each epoch prints one line of output:

    {"event": "epoch_completed", "epoch": 6, "train_loss": 0.2351,
     "train_accuracy": 0.9162, "val_loss": 0.2295, "val_accuracy": 0.9155,
     "duration_seconds": 68.4}

The values mean the following:

- epoch is the number of the training round
- train_accuracy is the score on the images used for training
- val_accuracy is the score on images the model has not seen before, so this
  is the more useful number
- val_loss decreasing means the model is still improving

The model is saved to checkpoints/classifier_v1.pt, but only when val_loss
improves. If there is no improvement for 3 epochs in a row, training stops
early and an early_stopping line is printed.

---

## Step 6: Start the API

This terminal must stay open while the API is running.

    $env:MODEL_PATH=".\checkpoints\classifier_v1.pt"
    python src\serve.py

The API starts on port 8080. This is the port the assignment asks the
container to use in Part C, so the same port is used here to keep everything
consistent.

If another program on the machine is already using 8080, such as Airflow or
Jenkins, the replies will come from that program instead of this API. In that
case pick a different port and pass the same one to the test script:

    $env:PORT="8000"
    python src\serve.py

---

## Step 7: Test the API

Open a second terminal, activate the virtual environment again, then run:

    python scripts\make_test_image.py --from-dataset
    python scripts\smoke_test.py

The first command saves a real image from the dataset as test_image.png and
prints its actual class. The second command calls all three endpoints and
prints the responses. The predicted class should match the actual class.

The address http://localhost:8080/docs also opens a page in the browser where
an image can be uploaded and the result viewed directly.

If the API was started on a different port, pass it to the test script:

    python scripts\smoke_test.py --base-url http://localhost:8000

---

## API endpoints

- /health reports whether the model is loaded. It returns 200 when the model
  is ready and 503 when it is not. Kubernetes uses this endpoint in Part D to
  check whether the application is healthy.
- /metadata reports which model is loaded, which dataset it was trained on and
  the accuracy it reached.
- /predict accepts an uploaded image and returns the predicted class along
  with a score for all 10 classes.

---

## Project files

    src/model.py      the two model designs: a small CNN and ResNet-18
    src/dataset.py    loads the Fashion-MNIST images and prepares them
    src/train.py      the training loop
    src/serve.py      the web API

    configs/training_config.yaml   the settings

    scripts/make_test_image.py    creates a test image
    scripts/smoke_test.py         calls the API and prints the responses

    tests/test_model.py   tests for the model and the settings
    tests/test_serve.py   tests for the API

---

## Changing the settings

All settings are in configs/training_config.yaml, including epochs,
batch_size and learning_rate.

For a temporary change, an environment variable can be used instead of editing
the file:

    EPOCHS              number of training rounds
    SUBSET_FRACTION     portion of the data to use, where 1.0 means all of it
    DATA_DIR            location of the dataset
    CHECKPOINT_DIR      location where the trained model is saved

The paths in the settings file are relative, such as ./data. Docker uses /app
as its working directory, so ./data becomes /app/data inside a container. The
same settings file therefore works both on a laptop and in Kubernetes.

---