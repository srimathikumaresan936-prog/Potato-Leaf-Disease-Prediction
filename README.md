# Potato Leaf Disease Prediction

A beginner-friendly computer vision project that classifies potato leaf images into three categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Classes

- **Early Blight**
- **Late Blight**
- **Healthy**

The project includes a training script and a Gradio web interface for uploading a leaf image and viewing predicted class probabilities.

## How it works

1. Prepare a dataset with one folder for each class.
2. Randomly sample a balanced number of images per class.
3. Split images into training (70%), validation (15%), and test (15%) sets.
4. Train a CNN on 224 × 224 RGB images for the selected number of epochs.
5. Evaluate the model and save it as `models/potato_leaf_model.keras`.
6. Run the Gradio app and upload a leaf image.

## Project structure

```text
Potato-Leaf-Disease-Prediction/
├── app.py
├── train.py
├── notebooks/
│   └── potato_plant_disease_prediction.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── .gitkeep
└── screenshots/
    └── .gitkeep
```

The original Colab notebook is included under `notebooks/` for reference. The dataset and trained model are not included in this repository. The model file can be large, and the dataset may have separate usage terms.

## Setup

Use Python 3.10 or 3.11 in a virtual environment. TensorFlow compatibility depends on your operating system and Python version.

### 1. Install dependencies

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install packages:

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download the Potato PlantVillage dataset used in the original notebook from Kaggle:

`aarishasifkhan/plantvillage-potato-disease-dataset`

Follow Kaggle's current terms and setup instructions. Extract the dataset so that the folder passed to `--dataset-dir` directly contains these three class folders:

```text
Potato___Early_blight/
Potato___Late_blight/
Potato___healthy/
```

The script needs at least 152 images in each class by default. If your extracted folder has an extra parent directory, pass the path that contains the three class folders.

### 3. Train the model

From the project root, run:

```bash
python train.py --dataset-dir "PATH_TO_FOLDER_CONTAINING_CLASS_FOLDERS"
```

For example, replace the placeholder with your actual local dataset path. The script creates balanced train/validation/test folders under `data/split/`, evaluates the model, prints a classification report and confusion matrix, and saves the trained model to `models/potato_leaf_model.keras`.

You can change the number of epochs:

```bash
python train.py --dataset-dir "PATH_TO_DATASET" --epochs 10
```

### 4. Launch the app

After training completes:

```bash
python app.py
```

Open the local Gradio URL shown in the terminal and upload a potato leaf image.

## Model details

- Framework: TensorFlow / Keras
- Model: Sequential CNN
- Input image size: 224 × 224 × 3
- Batch size: 32
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Output: 3-class softmax probabilities
- Data augmentation: random horizontal flip, rotation, and zoom
- Default epochs: 10

The CNN includes convolution and max-pooling blocks, followed by a dense layer and a three-unit softmax output layer.

## Results

The original notebook was set up to evaluate test accuracy, test loss, a confusion matrix, and a classification report. Run `train.py` to generate results for your local training run. Accuracy can vary based on the sampled images, dataset version, software environment, and training run.

## Limitations

- This is an educational prototype, not a professional plant-health diagnosis.
- It predicts only the three classes listed above.
- Performance on real-world photos may differ from performance on the dataset.
- A prediction confidence score is not a guarantee that the prediction is correct.

## Dataset

Dataset: [Potato PlantVillage Disease Dataset on Kaggle](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset)

Please follow the dataset provider's licensing and attribution requirements.

## Author

**Srimathi K**
