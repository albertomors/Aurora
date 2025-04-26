# Aurora: 3D Object Recognition and Classification

Aurora is a deep learning project focused on the recognition and classification of 3D objects. This repository contains the implementation details, dataset augmentation scripts, and relevant resources to facilitate understanding and replication of the work.

[Read the paper](Aurora_paper.pdf)

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Dataset Augmentation](#dataset-augmentation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Overview

The Aurora project aims to enhance the accuracy of 3D object recognition by employing advanced deep learning techniques. The approach includes dataset augmentation and the conversion of data formats to optimize the training process.

## Repository Structure

The repository is organized as follows:

```
Aurora/
├── Aurora_architecture.ipynb      # Jupyter Notebook detailing the model architecture
├── Aurora_paper.pdf               # Research paper describing the project
├── dataset_augmentation_offline.py# Script for offline dataset augmentation
├── npy_to_hdf5.py                 # Script to convert .npy files to .hdf5 format
├── LICENSE                        # License information
└── README.md                      # Project documentation
```

## Installation and Setup

To set up the project environment, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/1richi1/Aurora.git
   cd Aurora
   ```

2. **Create a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv aurora_env
   source aurora_env/bin/activate   # On Windows: aurora_env\Scripts\activate
   ```

3. **Install Dependencies:**

   Ensure you have `pip` installed. Then, install the required packages following first cell of Aurora_architecture.ipynb

## Dataset Augmentation

The project includes a script for offline dataset augmentation:

- **`dataset_augmentation_offline.py`**: This script performs augmentation on the existing dataset to increase its size and variability, which can help improve the model's performance.


## Usage

1. **Model Architecture:**

   The `Aurora_architecture.ipynb` notebook provides a detailed walkthrough of the model architecture. You can open this notebook using Jupyter Notebook or Jupyter Lab to explore and run the code cells interactively.

2. **Data Conversion:**

   If your dataset is in `.npy` format, you can convert it to `.hdf5` using the `npy_to_hdf5.py` script

3. **Training and Evaluation:**

   You can utilize the code within the `Aurora_architecture.ipynb` notebook to train and evaluate the model. Ensure that your data paths are correctly set within the notebook.

## Acknowledgements

This project would not have been possible without the valuable contributions and support of my colleagues, Alberto Morselli and Giovanni Girardin. Their insights, dedication, and collaborative efforts were instrumental in the success of Aurora. Thank you for sharing this journey and bringing your expertise to the table. Together, we have achieved something remarkable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

*Note: Ensure that all paths and filenames mentioned in the commands match your local directory structure.*
