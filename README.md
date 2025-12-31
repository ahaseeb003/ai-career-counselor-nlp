

# 🤖 AI Career Counselor (BERT Notebook)

This repository hosts a complete, end-to-end **Jupyter Notebook (`.ipynb`)** that fine-tunes a **BERT** model for career guidance. By running this single notebook, you can load data, train a deep learning model, evaluate its performance, and interact with a conversational AI career coach.

## 📂 Project Overview

This project solves the problem of automated career counseling using Natural Language Processing (NLP). Instead of simple keyword matching, it uses a transformer-based model to understand the *context* of a user's skills and interests to predict the best-fit job role from **54 distinct career classes**.

The notebook handles the entire pipeline:

1. **Data Preprocessing:** Cleaning and tokenization.
2. **Model Training:** Fine-tuning `bert-base-uncased`.
3. **Evaluation:** Generating confusion matrices and accuracy reports.
4. **Inference:** An interactive chat loop to test the model immediately.

## 📊 The Dataset

This project utilizes the **Career Guidance QA Dataset**.

* **Source:** [Hugging Face - Pradeep016/career-guidance-qa-dataset](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset/tree/main)
* **Format:** The notebook is designed to accept the `.csv` version of this dataset.
* **Content:** It contains pairs of user queries (skills/interests) and career labels (roles).

## 🚀 How to Use (The Notebook)

The entire project is contained within `Career_Counselor.ipynb`. Follow these steps:

### 1. Open in Google Colab

Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/) for the best experience (free GPU access is recommended).

### 2. Prepare the Data

Download the `train.csv` (or the full dataset CSV) from the [Hugging Face link](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset/tree/main) mentioned above.

### 3. Run the Cells

Execute the notebook cells in order:

* **Cell 1:** Installs necessary libraries (`transformers`, `datasets`, `torch`).
* **Cell 2:** Defines the `CareerCounselingBERT` class (the core logic).
* **Cell 3:** Uploads your downloaded CSV file. processing handles file renaming automatically.
* **Cell 4:** Trains the model. (Note: The notebook is configured for ~30 epochs to achieve high accuracy).
* **Cell 5:** Evaluates the model, exports the artifacts, and launches an **Interactive Chat**.

## 🧠 Model Performance

Based on training logs included in the notebook:

* **Test Accuracy:** 100.00%
* **Training Loss:** < 0.03
* **Classes Supported:** 54

## 📦 Exporting the Model

The notebook includes a feature in **Cell 5** that automatically:

1. Saves the trained model weights (`pytorch_model.bin`).
2. Saves the tokenizer and label encoders.
3. Generates a `run_model.py` script for local use.
4. Zips everything into a downloadable file (`career_model_final.zip`).

## 🛠️ Requirements

* Python 3.x
* Transformers (Hugging Face)
* PyTorch
* Scikit-Learn
* Pandas / NumPy

## 🤝 Credits

* **Dataset:** [Pradeep016 on Hugging Face](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset/tree/main)
* **Base Model:** BERT by Google Research

---

**Developed by:** Abdul Haseeb
**Degree:** BS Artificial Intelligence
**Focus:** NLP & Agentic AI & ML
