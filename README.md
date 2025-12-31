
# 🤖 AI Career Counselor (NLP-Powered)

An intelligent Career Counseling System built on **Deep Learning (BERT)**. This project leverages advanced Natural Language Processing (NLP) to analyze user interests and recommend the most suitable career path from 54 distinct professions, wrapped in a conversational "Generative AI" style interface.

## 🌟 Project Overview

This project addresses the challenge of personalized career guidance. Unlike basic rule-based systems, this model understands the *semantic context* of a user's description (e.g., distinguishing "I like design" in an artistic context vs. an engineering context).

It uses a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model to classify user inputs and provides human-like, context-aware advice.

### Key Features

* **🧠 Advanced NLP Core:** Fine-tuned `bert-base-uncased` model optimized for multi-class sequence classification.
* **🎯 High Precision:** Achieved **100% Test Accuracy** on the validation set after 30 epochs of training.
* **💬 LLM-Style Experience:** Features a "Generative Wrapper" that turns raw classification labels into conversational, helpful advice.
* **📊 Robust Evaluation:** validated using Confusion Matrices and Classification Reports to ensure no bias across 54 classes.
* **⚡ efficient Inference:** Optimized for local deployment; runs offline without external API dependencies.

## 🛠️ Technical Architecture

| Component | Technology | Description |
| --- | --- | --- |
| **Model Architecture** | BERT (Base Uncased) | Pre-trained Transformer fine-tuned for 54-class classification. |
| **Training Library** | Hugging Face Transformers | Used `Trainer` API with AdamW optimizer and linear learning rate scheduler. |
| **Dataset** | Custom Career QA Dataset | 1,600+ labeled samples cleaned and stratified into Train/Val/Test splits. |
| **Tokenizer** | WordPiece | BERT's native sub-word tokenizer with dynamic padding (`max_length=128`). |
| **Interface** | Python Script | Interactive CLI with confidence scoring and "Expert System" responses. |

## 🚀 How It Works

1. **Input Processing:** The user inputs a natural language query (e.g., *"I love working with data and predicting future trends"*).
2. **Tokenization:** The text is normalized, tokenized, and converted into input IDs and attention masks.
3. **Inference:** The BERT model processes the sequence and outputs logits for 54 career classes.
4. **Generative Response:** The predicted label (e.g., "Data Scientist") is mapped to a predefined "persona" response to mimic an LLM interaction.

## 📊 Performance Metrics

The model was trained with the following hyperparameter configuration:

* **Epochs:** 30
* **Batch Size:** 16
* **Learning Rate:** 3e-5
* **Warmup Steps:** 200

### Training Results

* **Training Loss:** 0.0287 (Converged)
* **Test Accuracy:** **100.00%**
* **Precision/Recall:** 1.00 across all supported classes.

*(Detailed classification reports and confusion matrices are available in the `logs/` directory)*

## 💻 Installation & Usage

### 1. Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ai-career-counselor.git
cd ai-career-counselor
pip install -r requirements.txt

```

### 2. Run the Model

Execute the inference script to start the interactive session:

```bash
python run_model.py

```

### 3. Example Interaction

```text
Describe your interests: I enjoy solving complex logic puzzles and writing code.
💡 Prediction: Software Engineer (Confidence: 99.8%)
🤖 AI Advice: "Based on your inputs, Software Engineering is the ideal path. 
   You seem to enjoy logic and building systems. I recommend starting with 
   Python or Java and building a GitHub portfolio."

```

## 📂 Project Structure

```
├── career_counselor_final/      # Exported Model Artifacts
│   ├── pytorch_model.bin        # Trained Weights
│   ├── config.json              # Model Configuration
│   ├── vocab.txt                # Tokenizer Vocabulary
│   ├── label_encoder.pkl        # Label Mappings
│   └── run_model.py             # Inference Script
├── dataset/                     # Training Data (CSV)
├── notebooks/                   # Jupyter/Colab Training Notebooks
├── requirements.txt             # Python Dependencies
└── README.md                    # Project Documentation

```

## 🔮 Future Improvements

* **Web Interface:** Deploying the model using Streamlit for a user-friendly web app.
* **RAG Integration:** Connecting the model to a live job market database to recommend *actual* open positions.
* **Resume Parsing:** Adding functionality to upload and analyze PDF resumes directly.

---

**Developed by:** Abdul Haseeb
**Degree:** BS Artificial Intelligence
**Focus:** NLP & Agentic AI & ML
