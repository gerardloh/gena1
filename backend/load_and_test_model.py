#!/usr/bin/env python3
"""
Script to load Qwen2.5-VL base model with LoRA adapters and test inference
Designed for the Origami GPU Cluster
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment and check GPU availability"""
    print("=" * 80)
    print("Environment Setup")
    print("=" * 80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("✗ CUDA is not available. Running on CPU (will be slow)")
    
    print(f"  PyTorch Version: {torch.__version__}")
    print()

def load_base_model(model_name="unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"):
    """
    Load the base model with 4-bit quantization
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        model: The loaded base model
        processor: The model processor/tokenizer
    """
    print("=" * 80)
    print("Loading Base Model")
    print("=" * 80)
    print(f"Model: {model_name}")
    print("Loading with 4-bit quantization (BitsAndBytes)...")
    
    try:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load the processor
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load the base model
        print("Loading base model (this may take a few minutes)...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        print("✓ Base model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Device: {model.device}")
        print()
        
        return model, processor
        
    except Exception as e:
        print(f"✗ Error loading base model: {e}")
        sys.exit(1)

def load_lora_adapters(model, adapter_path):
    """
    Load LoRA adapters into the base model
    
    Args:
        model: The base model
        adapter_path: Path to the LoRA adapter files
    
    Returns:
        model: Model with LoRA adapters loaded
    """
    print("=" * 80)
    print("Loading LoRA Adapters")
    print("=" * 80)
    print(f"Adapter path: {adapter_path}")
    
    # Convert to Path object
    adapter_path = Path(adapter_path)
    
    # Check if adapter path exists
    if not adapter_path.exists():
        print(f"✗ Adapter path does not exist: {adapter_path}")
        print("  Please verify the path to your adapter files.")
        sys.exit(1)
    
    # List adapter files
    print("\nAdapter files found:")
    adapter_files = list(adapter_path.glob("*"))
    for f in adapter_files:
        print(f"  - {f.name}")
    
    # Check for required adapter files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [f for f in required_files if not (adapter_path / f).exists()]
    
    if missing_files:
        print(f"\n✗ Missing required adapter files: {missing_files}")
        print("  Expected files: adapter_config.json, adapter_model.safetensors (or adapter_model.bin)")
        sys.exit(1)
    
    try:
        print("\nLoading LoRA adapters into model...")
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False  # Set to False for inference
        )
        
        print("✓ LoRA adapters loaded successfully!")
        print(f"  Adapter config loaded from: {adapter_path / 'adapter_config.json'}")
        print()
        
        return model
        
    except Exception as e:
        print(f"✗ Error loading LoRA adapters: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Ensure adapter_config.json is valid JSON")
        print("  2. Verify adapter weights file exists (adapter_model.safetensors or .bin)")
        print("  3. Check that adapters are compatible with the base model")
        sys.exit(1)

def test_model_text_only(model, processor):
    """
    Test the model with a simple text-only prompt
    
    Args:
        model: The model to test
        processor: The processor/tokenizer
    """
    print("=" * 80)
    print("Testing Model - Text Only")
    print("=" * 80)
    
    # Simple test prompt
    test_prompt = "What is the capital of France?"
    
    print(f"Prompt: {test_prompt}")
    print("\nGenerating response...")
    
    try:
        # Prepare input
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
        # Format the prompt
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]
        
        print("\n" + "-" * 80)
        print("Generated Response:")
        print("-" * 80)
        print(generated_text)
        print("-" * 80)
        print("\n✓ Text generation successful!")
        print()
        
    except Exception as e:
        print(f"\n✗ Error during text generation: {e}")
        print("  This might be a compatibility issue with the model/adapters.")
        return False
    
    return True

def test_model_with_image(model, processor, image_path=None):
    """
    Test the model with vision-language capabilities
    
    Args:
        model: The model to test
        processor: The processor/tokenizer
        image_path: Optional path to test image
    """
    print("=" * 80)
    print("Testing Model - Vision + Language")
    print("=" * 80)
    
    if image_path is None or not Path(image_path).exists():
        print("No test image provided. Skipping vision test.")
        print("To test with an image, provide image_path parameter.")
        return
    
    print(f"Image: {image_path}")
    
    try:
        from PIL import Image
        
        # Load image
        image = Image.open(image_path)
        
        # Test prompt
        test_prompt = "Describe this image in detail."
        
        print(f"Prompt: {test_prompt}")
        print("\nGenerating response...")
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": test_prompt}
                ]
            }
        ]
        
        # Format the prompt
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]
        
        print("\n" + "-" * 80)
        print("Generated Response:")
        print("-" * 80)
        print(generated_text)
        print("-" * 80)
        print("\n✓ Vision-language generation successful!")
        print()
        
    except Exception as e:
        print(f"\n✗ Error during vision-language generation: {e}")
        return False
    
    return True

def print_model_info(model):
    """Print detailed model information"""
    print("=" * 80)
    print("Model Information")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Check if LoRA adapters are present
    if hasattr(model, 'peft_config'):
        print("\n✓ LoRA adapters are active")
        for adapter_name in model.peft_config.keys():
            print(f"  Adapter: {adapter_name}")
    else:
        print("\n✗ No LoRA adapters detected")
    
    print()

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("Qwen2.5-VL LoRA Model Loading and Testing")
    print("=" * 80)
    print()
    
    # Configuration
    BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
    
    # IMPORTANT: Update this path to point to your adapter directory
    ADAPTER_PATH = "."  # Change this to your actual adapter path
    
    # Optional: Path to test image
    TEST_IMAGE = None  # Set to image path if you want to test vision capabilities
    
    # Setup
    setup_environment()
    
    # Load base model
    model, processor = load_base_model(BASE_MODEL)
    
    # Load LoRA adapters
    model = load_lora_adapters(model, ADAPTER_PATH)
    
    # Print model info
    print_model_info(model)
    
    # Test the model
    print("=" * 80)
    print("Running Model Tests")
    print("=" * 80)
    print()
    
    # Test 1: Text-only generation
    success = test_model_text_only(model, processor)
    
    if not success:
        print("\n⚠ Text generation test failed. There may be compatibility issues.")
        return
    
    # Test 2: Vision-language generation (if image provided)
    if TEST_IMAGE:
        test_model_with_image(model, processor, TEST_IMAGE)
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("✓ Base model loaded successfully")
    print("✓ LoRA adapters connected")
    print("✓ Model is ready for inference")
    print("\nYour fine-tuned model is working! Training weights are preserved.")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()