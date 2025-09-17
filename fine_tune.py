import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


class FineTuner:
    def __init__(self):
        # --- 1. Configuration ---
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

        self.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.final_model_path = "./tiny-conversational-assistant-final"

        self.dataset = None
        self.filtered_dataset = None
        self.processed_dataset = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.model = None
        self.trainer = None

    # --- 2. Load and Filter Dataset ---
    def load_and_filter_dataset(self):
        print("Loading and filtering the Dolly-15k dataset...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", token=self.hf_token)

        keywords = [
            "email", "letter", "note", "message", "subject:", "draft a reply", "write a response", "respond to",
            "professor", "teacher", "TA", "university", "academic", "college", "lecture", "course", "exam", "thesis",
            "professional", "colleague", "manager", "interview", "application", "job", "internship", "recommendation",
            "assignment", "deadline", "reschedule", "request for", "follow up", "feedback", "inquiry", "apologize", "decline"
        ]

        def is_relevant(example):
            text_to_check = (example['instruction'] + " " + example['category']).lower()
            return any(keyword in text_to_check for keyword in keywords)

        filtered_dataset = dataset.filter(is_relevant)

        if len(filtered_dataset) > 1000:
            filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(1000))

        print(f"✅ Created a targeted dataset with {len(filtered_dataset)} relevant examples.")
        self.filtered_dataset = filtered_dataset

    # --- 3. Format and Tokenize Dataset ---
    def process_dataset(self):
        def format_dolly_prompt(example):
            instruction = example["instruction"]
            context = example["context"]
            response = example["response"]
            if context:
                return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        def create_text_column(example):
            return {"text": format_dolly_prompt(example)}

        processed_dataset = self.filtered_dataset.map(
            create_text_column,
            remove_columns=list(self.filtered_dataset.features)
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=1024)

        tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        print("✅ Dataset formatted and tokenized.")

        self.processed_dataset = processed_dataset
        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset

    # --- 4. Load Model with 4-bit Quantization ---
    def load_model(self):
        print("Loading base model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        print("✅ Base model loaded.")

        # --- 5. Configure LoRA ---
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        self.model = get_peft_model(base_model, lora_config)
        print("✅ PEFT model configured with LoRA.")

    # --- 6. Set Up Trainer ---
    def setup_trainer(self):
        training_args = TrainingArguments(
            output_dir="./tiny-conversational-assistant-training",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=1,
            save_strategy="epoch",
            logging_steps=10,
            fp16=True,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

    # --- 7. Start Fine-Tuning ---
    def train(self):
        print("\nStarting fine-tuning...")
        self.trainer.train()
        print("✅ Fine-tuning finished!")

    # --- 8. Save the Final Model ---
    def save_model(self):
        self.trainer.save_model(self.final_model_path)
        print(f"✅ Model adapters saved to {self.final_model_path}")
        print("You can now zip this folder for easy distribution if needed.")

    # --- Run All Steps ---
    def run(self):
        self.load_and_filter_dataset()
        self.process_dataset()
        self.load_model()
        self.setup_trainer()
        self.train()
        self.save_model()


if __name__ == "__main__":
    fine_tuner = FineTuner()
    fine_tuner.run()
