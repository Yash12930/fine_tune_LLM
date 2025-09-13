---
title: Student Email Assistant
emoji: ✉️
colorFrom: gray
colorTo: indigo
sdk: streamlit
sdk_version: "1.33.0"
app_file: app.py
pinned: false
---

# AI Agent for Student & Professional Communication

* **Name**: Yash Sunil Choudhary
* **University**: Indian Institute of Technology, Guwahati
* **Department**: Mathematics and Computing

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yash12930/model)

## Overview

[cite_start]This project is an AI agent designed to automate the daily task of drafting replies to messages, specifically tailored for student and professional academic contexts[cite: 4]. [cite_start]The agent follows a **Planner-Executor** design pattern to structure its reasoning and execution flow, ensuring coherent and contextually appropriate responses[cite: 15].

[cite_start]The core of the agent is a fine-tuned `TinyLlama-1.1B-Chat-v1.0` model, which has been specialized for this task using **Parameter-Efficient Fine-Tuning (PEFT)** with LoRA[cite: 8].

## 📜 Features

* [cite_start]**Task Automation**: Automates the task of drafting emails and messages for common university and professional scenarios[cite: 4].
* [cite_start]**Fine-Tuned Model**: Utilizes a specialized LoRA fine-tuned `TinyLlama` model for improved response quality and relevance[cite: 6, 8].
* [cite_start]**Multi-Agent Architecture**: Implements a Planner-Executor agent design for structured reasoning and task execution[cite: 15].
* [cite_start]**Interactive UI**: A web-based user interface built with Streamlit for easy interaction[cite: 17].

## 🏗️ Architecture

[cite_start]The agent follows a **Planner-Executor** design pattern to structure its reasoning and execution flow[cite: 1].

1.  **Planner Agent**: A Python function that acts as the "thinking" component. [cite_start]It receives the user's goal and any incoming message and creates a detailed, structured prompt for the model[cite: 1].
2.  **Executor Agent**: A Python function that acts as the "doing" component. [cite_start]It takes the structured prompt from the Planner and executes the task by calling the specialized tool[cite: 1].
3.  **Specialized Tool (Fine-Tuned Model)**: The core tool is the `TinyLlama-1.1B-Chat-v1.0` model fine-tuned using LoRA. [cite_start]This model is an expert at generating coherent responses for student-related conversations[cite: 1].


## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Hugging Face Transformers, PEFT, and Datasets**
* **Streamlit** for the user interface
* **Git & Git LFS** for version control

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Yash12930/fine_tune_LLM.git](https://github.com/Yash12930/fine_tune_LLM.git)
    cd fine_tune_LLM
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up your Hugging Face Token:**
    You need a Hugging Face token with **WRITE** permissions to run the scripts.
    ```bash
    export HF_TOKEN='your_hugging_face_write_token_here'
    ```

## 🚀 Usage

The project is split into two main parts: fine-tuning the model and running the Streamlit application.

### Fine-Tuning the Model
The script `fine_tune.py` handles the entire PEFT process.
```bash
python fine_tune.py
```
This script will:
1.  Download the `databricks/databricks-dolly-15k` dataset.
2.  Filter it for examples relevant to student/professional communication.
3.  Load the base `TinyLlama-1.1B-Chat-v1.0` model with 4-bit quantization.
4.  Fine-tune the model using LoRA on the filtered dataset.
5.  Save the resulting adapter weights to the `./tiny-conversational-assistant-final` directory.

### Running the Streamlit App
To start the web interface, run the `app.py` script:
```bash
streamlit run app.py
```
This will launch a local web server where you can interact with the fine-tuned agent.

## 📊 Data Science Report

### Fine-Tuning Setup
* [cite_start]**Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was chosen for its excellent balance of performance and efficiency, making it possible to fine-tune quickly on free hardware like a Google Colab T4 GPU[cite: 1].
* **Dataset**: The `databricks/databricks-dolly-15k` dataset was used as the source. [cite_start]It was filtered down to **517 high-quality examples** relevant to academic and professional communication using a keyword search[cite: 2].
* **Method**: **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** was used. [cite_start]This method freezes the base model's weights and injects trainable rank-decomposition matrices into the Transformer architecture, drastically reducing the number of parameters that need to be trained[cite: 8].

### Fine-Tuning Results
The model was trained for one epoch. [cite_start]The training loss shows that the model successfully learned from the specialized dataset[cite: 26].


### Evaluation and Outcomes
[cite_start]A qualitative evaluation was performed by comparing the outputs of the **fine-tuned model** against the **original base model** on a set of 10 test cases[cite: 28]. [cite_start]The full comparison can be found in `INTERACTION_LOG.md`[cite: 29].

**Key Findings:**
* **Improved Relevance**: The fine-tuned model consistently produces more relevant and specific responses. [cite_start]In **Test Case #1**, it correctly generates a structured request for feedback, whereas the base model hallucinates a previous submission[cite: 3].
* **Reduced Nonsense**: The fine-tuned model avoids the logical errors often made by the base model. [cite_start]In **Test Case #2**, the fine-tuned model politely declines an invitation, while the base model's response is confusing and contradictory, telling the inviter to "reconsider the invitation"[cite: 3].
* **Better Formatting and Style**: The fine-tuned model is better at generating clean, ready-to-use text. [cite_start]In **Test Case #10**, it produces a concise message, while the base model includes extra placeholders[cite: 3].

[cite_start]The primary goal of fine-tuning—**task specialization**—was clearly achieved, leading to improved reliability and adapted style for the target use case[cite: 12].

## ✨ Demo Screenshot
Here is a screenshot of the Streamlit application in action.


## ✅ Deliverables Checklist

* [cite_start][x] **Source Code**: All source code is included in this repository[cite: 23].
* [cite_start][x] **Architecture Document**: `ARCHITECTURE.md` is present in the repository[cite: 24].
* [cite_start][x] **Data Science Report**: This README serves as the data science report, covering fine-tuning and evaluation[cite: 25].
* [cite_start][x] **Interaction Logs**: The full chat history with me (the AI assistant) is included as `INTERACTION_LOG.md`[cite: 29].
* [x] **Demo Screenshots**: A screenshot of the UI is included above.