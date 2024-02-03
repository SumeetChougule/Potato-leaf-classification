Agri Disease Classification
==============================

 End-to-end deep learning project for potato disease classification.


# Potato Leaf Classification

This project is about classifying healthy and unhealthy potato leaves.

## Usage

To run the project locally, you will need to set up a Python environment with the required dependencies. You can use `conda` to create an environment using the provided `environment.yml` file.

### Create Conda Environment

```bash
conda env create -f environment.yml
```

### Activate Conda Environment
```bash
conda activate agri_classification
```

### Run FastAPI Application

Make sure your conda environment is activated, then run the FastAPI application:

```bash
uvicorn main:app --reload
```

### Jupyter Notebook

If you want to use Jupyter notebooks, you can launch the notebook server:

```bash
jupyter notebook
```
### Model

The model used for classification is stored in the file saved_models/1.h5.

