# scSpecies

## Introduction
`scSpecies` is a deep learning model designed to align network architectures for datasets across multiple species. 
The model builds on conditional variational autoencoder and transfer learning to
establish a direct correspondence between cells of multiple of single-cell RNA sequencing datasets. 


The model offers the following functionalities:

- **Align and vizualize Latent Representaions:** Align latent representations of datasets from different species. The influence of experimental batch effects is corrected for in this representation.
- **Transferring Cell Labels and Information:** Transfer labels and information between the datasets.
- **Differential Gene Expression Analysis:** Aids in identifying differentially expressed genes among biologically similar cells across species.
- **Aligned Cell Atlas Creation:** Assists in creating an aligned cell atlas that spans multiple species.
- **Relevance Score Computation:** Computes and compares the relevance scores of genes between species, providing insights into their biological significance.

## Prerequisites

This project requires Python 3.9.16. Please ensure that you have this version installed on your system before proceeding with the setup.

## Setting Up the Environment

Follow these steps to set up your environment and start using `scSpecies`:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/[your-username]/scSpecies.git
   cd scSpecies

2. **Create and Activate a Virtual Environment:**
   1. Using a virtual environment is recommended to avoid package conflicts.
      For Unix-based systems (Linux/macOS):
       ```bash
      python3.9 -m venv venv
      source venv/bin/activate

   2. For Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate

3. **Install Required Packages:**
   Install the dependencies listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt

## Demo usage

To get started with `scSpecies`, please refer to the `tutorial.ipynb` Jupyter notebook. This tutorial provides a comprehensive guide on how to use the tool, including data preparation, model training, and analysis.  
