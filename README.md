# scSpecies

## Introduction
`scSpecies` is a deep learning model designed to align network architectures for single-cell RNA sequencing datasets across multiple species. 
The model builds on conditional variational autoencoders and transfer learning to
establish a direct correspondence between cells of multiple datasets. 

![Architecture](/figures/scSpecies_model_architecture.jpeg)

The model offers the following functionalities:

- **Align and vizualize Latent Representaions:** Align latent representations of datasets from different species. The influence of experimental batch effects is corrected for in the latent representation.
- **Transfer Cell Labels and Information:** After training cell labels or information like disease background can be transferred based on proximity of cells in the latent space.
- **Differential Gene Expression Analysis:** Aids in identifying differentially expressed genes among biologically similar cells across species.
- **Aligned Cell Atlas Creation:** Assists in creating an aligned cell atlas that spans multiple species.
- **Relevance Score Computation:** Assigns and compares the relevance scores of genes between cell types of different species, providing insights into similarities and differences for their biological significance.

![Atlas](/figures/multiple_species.jpeg)

## Prerequisites

This repository requires Python 3.9 [(Download here)](https://www.python.org/downloads/). Please ensure that you have this version installed on your system before proceeding with the setup. We used version 3.9.16.

## Setting Up the Environment

Follow these steps to set up your environment and start using `scSpecies`:

1. **Clone the Repository:** Open your desired folder and run
   ```bash
   git clone https://github.com/cschaech/scSpecies.git
   cd scSpecies

2. **Create and Activate a Virtual Environment:**
   1. Using a virtual environment is recommended to avoid package conflicts.
      For Unix-based systems (Linux/macOS):
       ```bash
      python3.9 -m venv scSpecies
      source scSpecies/bin/activate

   2. For Windows:
      ```bash
      python3.9 -m venv scSpecies
      .\scSpecies\Scripts\activate

3. **Install Required Packages:**
   Install the dependencies listed in `requirements.txt`.
   ```bash
   pip3 install -r requirements.txt

4. **Install PyTorch with CUDA:**
   Using CUDA significantly speeds up computations; if CUDA 11.8 is set up, run
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Demo usage

To get started with `scSpecies`, please refer to the `tutorial.ipynb` Jupyter notebook. This tutorial provides a comprehensive guide on how to use the tool, including data preparation, model training, and analysis.  

## Content of this repository

