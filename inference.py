import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import os

# --- 1. Configuration Constants (Matching train.py) ---
# The original model you started with
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
# The local directory where the adapter weights were saved
ADAPTER_PATH = "./results/mistral-7b-qlora-sft"

# --- 2. QLoRA and Quantization Configuration ---

# We must use the same quantization settings as training to load the model correctly.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 3. Load Base Model and Adapter ---

def load_fine_tuned_model():
    """Loads the base model and attaches the fine-tuned LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_ID}")

    # 3a. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3b. Load Base Model in 4-bit (QLoRA)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # Ensure model is in evaluation mode
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    
    # 3c. Load the LoRA adapter weights onto the base model
    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: Adapter path not found at {ADAPTER_PATH}. Did training complete?")
        return None, None
        
    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    # Set the model to evaluation mode
    model.eval()
    
    # You can optionally merge the weights for easier deployment, but it's not strictly necessary for inference
    # model = model.merge_and_unload() 

    print("Model and adapter loaded successfully!")
    return model, tokenizer

# --- 4. Run Inference ---

def generate_response(model, tokenizer, prompt: str):
    """Generates a response from the fine-tuned model."""
    if model is None:
        print("Cannot run generation, model failed to load.")
        return

    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the correct device (GPU if available)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    # Generate the output text
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200, # Controls the maximum length of the generated response
            do_sample=True, # Use sampling for more creative responses
            temperature=0.7, # Controls randomness (lower is more predictable)
            top_p=0.9, # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the output and clean up
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the final output for a cleaner look
    response = decoded_output.replace(prompt, "").strip()
    return response

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is required to run inference on this QLoRA model.")
    else:
        fine_tuned_model, tokenizer = load_fine_tuned_model()
        
        if fine_tuned_model and tokenizer:
            # Example prompts to test your model
            test_prompt = "Tell me a short, fun fact about space and then write a simple Python function to calculate the area of a circle."
            
            print("\n--- Running Inference ---")
            print(f"Prompt: {test_prompt}")
            
            generated_text = generate_response(fine_tuned_model, tokenizer, test_prompt)
            
            print("\n--- Generated Response ---")
            print(generated_text)
            print("---------------------------\n")

            print("Next Step: Run this script using a CUDA-enabled GPU to see your fine-tuned model in action!")
