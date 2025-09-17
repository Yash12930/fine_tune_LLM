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

This project is an AI agent designed to automate the daily task of drafting replies to messages, specifically tailored for student and professional academic contexts.

The core of the agent is a fine-tuned `TinyLlama-1.1B-Chat-v1.0` model, which has been specialized for this task using **Parameter-Efficient Fine-Tuning (PEFT)** with LoRA.

The model has been hosted on https://huggingface.co/spaces/Yash12930/model 
Feel free to check it out!

To view the chat log I used the Gemini LLM to help me out.
here is the log: https://g.co/gemini/share/533cbe096690

## 📜 Features

* **Task Automation**: Automates the task of drafting emails and messages for common university and professional scenarios.
* **Fine-Tuned Model**: Utilizes a specialized LoRA fine-tuned `TinyLlama` model for improved response quality and relevance.
* **Interactive UI**: A web-based user interface built with Streamlit for easy interaction.

## 🏗️ Architecture

The application is designed as a **single-agent system**, optimized for simplicity, readability, and maintainability. It follows a clear two-step logical flow:

1. **Prompt Construction (Planner Function)**  
   - Takes the user's **goal** (e.g., “request an extension”) and any **incoming message** (context).  
   - Formats these inputs into a structured prompt using the fine-tuned training format:  

     ```
     ### Human: <user input>
     ### Assistant:
     ```
   - This ensures consistency between **training and inference**, improving the quality of responses.

2. **Model Execution (Executor Function)**  
   - Sends the structured prompt to the fine-tuned **TinyLlama model**.  
   - Streams the generated response back to the user interface **in real time**, ensuring responsiveness and a natural conversational flow.

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
* **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was chosen for its excellent balance of performance and efficiency, making it possible to fine-tune quickly on free hardware of a Google Colab T4 GPU.
* **Dataset**: The `databricks/databricks-dolly-15k` dataset was used as the source. It was filtered down to **517 high-quality examples** relevant to academic and professional communication using a keyword search.
* **Method**: **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** was used. This method freezes the base model's weights and injects trainable rank-decomposition matrices into the Transformer architecture, drastically reducing the number of parameters that need to be trained.

### Fine-Tuning Results
The model was trained for one epoch. The training loss shows that the model successfully learned from the specialized dataset.


### Evaluation and Outcomes
A qualitative evaluation was performed by comparing the outputs of the **fine-tuned model** against the **original base model** on a set of 10 test cases. The full comparison can be found in `INTERACTION_LOG.md`.

**Key Findings:**
* **Improved Relevance**: The fine-tuned model consistently produces more relevant and specific responses. In **Test Case #1**, it correctly generates a structured request for feedback, whereas the base model hallucinates a previous submission.
* **Reduced Nonsense**: The fine-tuned model avoids the logical errors often made by the base model. In **Test Case #2**, the fine-tuned model politely declines an invitation, while the base model's response is confusing and contradictory, telling the inviter to "reconsider the invitation".
* **Better Formatting and Style**: The fine-tuned model is better at generating clean, ready-to-use text. In **Test Case #10**, it produces a concise message, while the base model includes extra placeholders.

The primary goal of fine-tuning—**task specialization**—was clearly achieved, leading to improved reliability and adapted style for the target use case.

## ✅ Deliverables Checklist

* [x] **Source Code**: All source code is included in this repository.
* [x] **Architecture Document**: `ARCHITECTURE.md` is present in the repository.
* [x] **Data Science Report**: This README serves as the data science report, covering fine-tuning and evaluation.
* [x] **Interaction Logs**: The full chat history with me (the AI assistant) is included as `INTERACTION_LOG.md`.
* [x] **Demo Screenshots**: A screenshot of the UI is included above.



Note: To view the jupyter notebooks, you have to download them, since github does not render interactive elements, and the interactive elements cause github to output invalid notebook. All the relevant codes are included in the .py files.
