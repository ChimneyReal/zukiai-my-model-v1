import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os

# --- 1. Configuration Constants ---

# You must replace this with the model you chose.
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
# Replace with the name of the dataset you want to use. We use 'timdettmers/openassistant-guanaco'
# which is a common instruction-tuning dataset, as a default.
DATASET_ID = "timdettmers/openassistant-guanaco"
# Replace with your desired output directory and model name
OUTPUT_DIR = "./results/mistral-7b-qlora-sft"
NEW_MODEL_NAME = "mistral-7b-custom-instruction-adapter"

# --- 2. QLoRA and Quantization Configuration ---

# Load model in 4-bit (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Highly recommended for fine-tuning
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for stability on Ampere+ GPUs
    bnb_4bit_use_double_quant=False, # Standard setting
)

# LoRA Configuration
# r (rank) is the size of the update matrices. Lower means fewer trainable parameters.
# target_modules specifies which layers to inject LoRA into. 'all-linear' is a good default.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32, # Scaling factor, usually 2*r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# --- 3. Training Arguments ---

# These define the core training process settings (learning rate, epochs, logging, etc.)
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit", # Optimizer optimized for QLoRA
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # Use bfloat16 if GPU supports it for better numerical stability
    max_grad_norm=0.3,
    max_steps=-1, # -1 means determined by epochs
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard", # Essential for tracking progress
    push_to_hub=True, # Enables pushing the adapter to Hugging Face Hub
    hub_model_id=NEW_MODEL_NAME, # Name for the model adapter on the Hub
)

# --- 4. Load Model, Tokenizer, and Data ---

def load_components():
    """Loads the model, tokenizer, and dataset."""
    print(f"Loading model: {BASE_MODEL_ID}")
    
    # 4a. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps layers to available devices
        trust_remote_code=True,
    )
    model.config.use_cache = False # Required for gradient checkpointing
    model.config.pretraining_tp = 1 # Recommended for Mistral models
    
    # 4b. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    # Important: Set padding token for Causal LMs
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right"  # Fixes issue with Llama/Mistral tokenizers
    
    # 4c. Load Dataset
    print(f"Loading dataset: {DATASET_ID}")
    # The dataset MUST have a column named 'text' containing the instruction-response pair.
    # The SFTTrainer handles the tokenization and formatting.
    dataset = load_dataset(DATASET_ID, split="train")

    return model, tokenizer, dataset

# --- 5. Initialize and Run Trainer ---

def run_fine_tuning():
    """Initializes and runs the SFTTrainer."""
    model, tokenizer, dataset = load_components()

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text", # The column in the dataset containing the training data
        max_seq_length=512, # Maximum sequence length for training
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False, # Set to True if your dataset samples are very short for efficiency
    )

    print("Starting training process...")
    trainer.train()

    print("Training complete. Saving final adapter...")
    # Save the adapter weights locally and push to the Hub
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Optional: Merge and push the final model (for easy deployment later)
    # merged_model = model.merge_and_unload()
    # merged_model.push_to_hub(f"{NEW_MODEL_NAME}-merged")
    # tokenizer.push_to_hub(f"{NEW_MODEL_NAME}-merged")

    print(f"Fine-tuning adapter saved to {OUTPUT_DIR} and pushed to Hugging Face Hub.")
    print("Next Step: Run the script using a compatible environment (e.g., CUDA-enabled GPU).")

if __name__ == "__main__":
    # Ensure you are logged into Hugging Face and have a token set up in your environment
    # Use: huggingface-cli login
    
    # Check for bfloat16 compatibility and adjust if necessary
    if not torch.cuda.is_bf16_supported() and training_arguments.bf16:
        print("Warning: bfloat16 is not supported on this GPU. Switching to fp16.")
        training_arguments.bf16 = False
        training_arguments.fp16 = True
    
    run_fine_tuning()
