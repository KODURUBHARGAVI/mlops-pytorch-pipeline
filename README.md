# mlops-pytorch-pipeline

This project trains an image classifier using PyTorch and serves it through a
web API. An image can be sent to the API, and it returns the predicted class.

The project runs on a normal laptop CPU. A GPU is not required.

The dataset is Fashion-MNIST. It contains 70,000 grey images of clothing
items, sized 28 by 28, divided into 10 classes such as T-shirt, Trouser,
Sneaker and Bag.

The work is done in parts, and each part adds to the same project. The
sections below follow that order.

---

# Part A: Repository setup

This part creates the folder structure, the ignore rules and the CI workflow.

## The folders

    .github/workflows/ci.yml       runs the linter, the tests and the builds
    configs/training_config.yaml   all the settings
    src/                           model.py, dataset.py, train.py, serve.py
    docker/                        Dockerfile.train and Dockerfile.serve
    k8s/                           the Kubernetes files
    requirements/                  train.txt, serve.txt and dev.txt
    scripts/                       helper scripts
    tests/                         the tests

## The branches

Work is done on feature branches, which are merged into develop through pull
requests, and develop is merged into main at the end.

    git branch -a
    git log --oneline --graph --all

## The ignore rules

The dataset and the trained model are large and are not committed. After a
training run the data and checkpoints folders will hold more than 100 MB, and
git should still report nothing to commit:

    git status

---

# Part B: Model, training and the API

This part adds the model, the training script and the web API.

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

## Step 2: Install the packages

    pip install -r requirements/dev.txt

The first install takes a few minutes because PyTorch is a large download.

## Step 3: Run the tests

    pytest tests -v
    ruff check src tests scripts

The tests check the model output shapes, saving and loading a model file, the
handling of uploaded images, the settings file and the API. They all pass in
under a minute.

## Step 4: Short training run

This run uses 2 epochs and 15 percent of the data, and finishes in under a
minute. It is a quick check that the training loop works.

On Windows PowerShell:

    $env:EPOCHS="2"
    $env:SUBSET_FRACTION="0.15"
    python src\train.py

On Mac or Linux:

    EPOCHS=2 SUBSET_FRACTION=0.15 python src/train.py

On the first run the dataset is downloaded into a folder named data. This is
about 30 MB, and later runs reuse the same copy.

## Step 5: Full training run

On Windows the settings from Step 4 stay active for the rest of the terminal
session, so they must be cleared first.

    Remove-Item Env:EPOCHS, Env:SUBSET_FRACTION
    $env:TORCH_NUM_THREADS="8"
    python src\train.py

On Mac or Linux, open a new terminal or run:

    TORCH_NUM_THREADS=8 python src/train.py

The full run takes about 10 to 15 minutes on 8 CPU cores and reaches around
91 percent accuracy.

Each epoch prints one line:

    {"event": "epoch_completed", "epoch": 6, "train_loss": 0.2351,
     "train_accuracy": 0.9162, "val_loss": 0.2295, "val_accuracy": 0.9155,
     "duration_seconds": 68.4}

The values mean the following:

- epoch is the number of the training round
- train_accuracy is the score on the images used for training
- val_accuracy is the score on images the model has not seen, so this is the
  more useful number
- val_loss decreasing means the model is still improving

The model is saved to checkpoints/classifier_v1.pt, but only when val_loss
improves. If there is no improvement for 3 epochs in a row, training stops
early and an early_stopping line is printed.

## Step 6: Start the API

This terminal must stay open while the API is running.

    $env:MODEL_PATH=".\checkpoints\classifier_v1.pt"
    python src\serve.py

The API starts on port 8080. If another program is already using that port,
choose a different one with $env:PORT and pass the same one to the test script
in the next step.

## Step 7: Test the API

Open a second terminal, activate the virtual environment again, then run:

    python scripts\make_test_image.py --from-dataset
    python scripts\smoke_test.py

The first command saves a real image from the dataset as test_image.png and
prints its actual class. The second command calls all three endpoints and
prints the responses. The predicted class should match the actual class.

The address http://localhost:8080/docs also opens a page in the browser where
an image can be uploaded and the result viewed directly.

## The endpoints

- /health reports whether the model is loaded. It returns 200 when the model
  is ready and 503 when it is not.
- /metadata reports which model is loaded, which dataset it was trained on and
  the accuracy it reached.
- /predict accepts an uploaded image and returns the predicted class along
  with a score for all 10 classes.

---

# Part C: Docker images

This part packages the training script and the API into two images.

Docker Desktop must be running before any of these commands.

## Step 8: Build and run the training image

    docker build -f docker/Dockerfile.train -t mlops-train:v1 .

The first build takes several minutes, because PyTorch is downloaded inside
the image.

    docker run --rm -e EPOCHS=2 -e SUBSET_FRACTION=0.15 -v ${PWD}/data:/app/data -v ${PWD}/checkpoints:/app/checkpoints mlops-train:v1

The two mounted folders keep the images and the trained model on the laptop
instead of inside the container, so nothing is lost when the container stops,
and the dataset is not downloaded again.

Leave out the two environment variables for a full run.

## Step 9: Build and run the serving image

    docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

    docker run --rm -p 8080:8080 -v ${PWD}/checkpoints:/app/checkpoints mlops-serve:v1

Then test it from a second terminal, the same way as in Step 7:

    python scripts\smoke_test.py

On Mac or Linux, replace ${PWD} with $(pwd).

## Step 10: Check the images

The container reports itself as healthy after about 20 seconds:

    docker ps

The STATUS column shows healthy once Docker has called /health inside the
container and received a 200. This comes from the HEALTHCHECK line in
Dockerfile.serve.

The sizes can be listed with:

    docker image ls --filter "reference=mlops-*"

Both images are around 1.5 GB on disk and about 315 MB compressed. PyTorch is
most of that, and both images need it, so their sizes are close.

The packages installed in each image can be compared:

    docker run --rm --entrypoint pip mlops-train:v1 list
    docker run --rm mlops-serve:v1 pip list

The training image needs --entrypoint pip because its entrypoint is set to run
the training script.

The serving image does not contain the training script at all:

    docker run --rm mlops-serve:v1 ls src/

This prints dataset.py, model.py and serve.py only, because Dockerfile.serve
copies those three files by name instead of the whole folder.

## How the images are built

Both use more than one stage. The first stage installs the packages into a
virtual environment, and only that environment is copied into the final image,
so the compiler and the download cache are left behind.

The training image copies the src and configs folders, creates the folders for
the data and the saved model, and runs as a user other than root. CONFIG_PATH
points at the settings file, so a different file can be mounted over it
without rebuilding the image.

The serving image installs from requirements/serve.txt, which has no packages
that are only used for training. It runs as a user other than root, opens port
8080, and has a HEALTHCHECK that calls /health using Python rather than curl,
so no extra package has to be installed.

All the versions in requirements/train.txt and requirements/serve.txt are
pinned, so the same image is produced each time it is built.

---

# Settings

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
