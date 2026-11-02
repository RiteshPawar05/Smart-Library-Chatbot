# 📚 Smart Library Chatbot (Digital Librarian)
*A Hybrid Deep Learning Architecture for an Intelligent Library Search System with NLP*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-green.svg)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Overview
Traditional library catalogs rely on strict keyword matching (like TF-IDF) and lack conversational memory. **Smart Library Chatbot** bridges this gap by replacing legacy methods with a modern, multi-stage Deep Learning pipeline. It understands the *semantic meaning* of queries, classifies user intent, generates natural human-like responses using RAG, and maintains stateful conversation history for follow-up questions.

Developed by **Jayesh Patil (C233)** and **Ritesh Pawar (C276)** as part of the **B.Tech (AIML) Semester V** curriculum (Deep Learning & Natural Language Processing).

---

## 🏗️ Core Architecture & Pipeline

1. **Phase 1: Advanced Preprocessing**
   - Dual-pipeline approach: Light query preprocessing (spell correction via TextBlob, abbreviation expansion like "dsa" $\rightarrow$ "data structures and algorithms") for semantic search, and standard NLP pipelines (lemmatization, stopword removal) for intent classification.
2. **Phase 2: Semantic Retriever (Sentence-BERT)**
   - Uses `all-MiniLM-L6-v2` SBERT embeddings and cosine similarity instead of TF-IDF to map user intent directly to book topics, summaries, and descriptions.
3. **Phase 3: Neural Intent Classifier (TensorFlow / Keras)**
   - A custom deep learning classification model utilizing an Embedding layer and `GlobalAveragePooling1D` to accurately route queries (e.g., finding topics, locations, or requesting summaries). Trained on an augmented dataset of over 450 examples.
4. **Phase 4: RAG & Stateful Memory Pipeline (T5 + Gradio)**
   - **Generative AI (T5-small):** Acts as the generator in a Retrieval-Augmented Generation (RAG) setup, crafting concise, context-aware answers from retrieved book details.
   - **Stateful Memory (`gr.State`):** Persists recommendation context across conversation turns, enabling seamless follow-up interaction (e.g., typing `"1"` to request details about the first recommended book).

---

## 🛠️ Tech Stack
- **Deep Learning / NLP:** TensorFlow, Keras, Hugging Face Transformers (`t5-small`, `Sentence-Transformers`), NLTK, TextBlob
- **Data Manipulation:** Pandas, NumPy
- **Interface & UI:** Gradio (Soft Theme, custom sidebar layout with state management)
- **Environment:** Google Colab / Python

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed along with the required libraries:
```bash
pip install tensorflow transformers sentence-transformers gradio pandas nltk textblob
