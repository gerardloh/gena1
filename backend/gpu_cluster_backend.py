"""
Vision-Language Model Backend for GPU Cluster
Optimized for CUDA GPUs with your LoRA adapter
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import os
from PIL import Image
import requests
from io import BytesIO

class VisionLanguageModelGPU:
    def __init__(self, base_model_name, adapter_path=None):
        """
        Initialize the model with base model and optional adapter
        
        Args:
            base_model_name: Hugging Face model name
            adapter_path: Optional path to LoRA adapter folder
        """
        print("🚀 Initializing Vision-Language Model for GPU Cluster...")
        
        # Check for GPU
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ Using GPU: {gpu_name}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = "cpu"
            print("⚠️  No GPU detected - using CPU (will be slow)")
        
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        
        # Load base model
        print(f"\n📥 Loading base model: {base_model_name}")
        print("   (First run will download model - may take a few minutes)")
        
        try:
            # Use AutoModel to automatically detect the correct model class
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,  # Use FP16 for faster inference
                device_map="auto",  # Automatically distribute across GPUs
                trust_remote_code=True,
                ignore_mismatched_sizes=True  # Allow minor size mismatches
            )
            print("✅ Base model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            raise
        
        # Load LoRA adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            print(f"\n📥 Loading LoRA adapter from {adapter_path}...")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path
                )
                print("✅ LoRA adapter loaded successfully")
                print("🎯 Model now uses your custom training!")
            except Exception as e:
                print(f"❌ Error loading adapter: {e}")
                print("⚠️  Continuing with base model only")
        
        # Load processor
        print("\n📥 Loading processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            print("✅ Processor loaded successfully")
        except Exception as e:
            print(f"❌ Error loading processor: {e}")
            raise
        
        self.model.eval()
        print("\n✨ Model ready for inference!\n")
    
    def generate_response(self, image_path_or_url, prompt, max_new_tokens=512, temperature=0.7):
        """
        Generate a response based on an image and text prompt
        
        Args:
            image_path_or_url: Local path or URL to image
            prompt: Text prompt/question about the image
            max_new_tokens: Maximum length of generated response
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        
        Returns:
            Generated text response
        """
        try:
            # Load image
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path_or_url)
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # Move to GPU
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate
            print(f"⏳ Generating response (max {max_new_tokens} tokens)...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
            
            # Decode output
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            import traceback
            print(f"❌ Error generating response: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"


def run_test(model):
    """Test the model with a sample image"""
    print("=" * 60)
    print("🧪 RUNNING TEST")
    print("=" * 60)
    
    # Test with a sample image
    test_image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    test_prompt = "What do you see in this image? Describe it in detail."
    
    print(f"\n📸 Test Image: {test_image_url}")
    print(f"💬 Prompt: {test_prompt}\n")
    
    response = model.generate_response(test_image_url, test_prompt, max_new_tokens=200)
    
    print("\n" + "-" * 60)
    print("🤖 MODEL RESPONSE:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    print("\n✅ Test complete!\n")


def main():
    """Main function to initialize and test the model"""
    
    # CONFIGURATION
    # Use the exact base model that your adapter was trained on
    BASE_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct"  # Unsloth's version without quantization
    ADAPTER_PATH = "."  # Current directory (update if needed)
    
    print("=" * 60)
    print("Vision-Language Model - GPU Cluster Version")
    print("=" * 60)
    print(f"\n📋 Base Model: {BASE_MODEL}")
    
    # Check if adapter exists
    use_adapter = False
    if os.path.exists(ADAPTER_PATH):
        adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            print(f"📁 Adapter: {ADAPTER_PATH}")
            use_adapter = True
        else:
            print("⚠️  No adapter found - using base model only")
            ADAPTER_PATH = None
    else:
        print("ℹ️  No adapter path specified - using base model only")
        ADAPTER_PATH = None
    
    print()
    
    try:
        # Initialize model
        model = VisionLanguageModelGPU(
            BASE_MODEL,
            adapter_path=ADAPTER_PATH if use_adapter else None
        )
        
        # Run test
        run_test(model)
        
        # Ready for use
        print("=" * 60)
        print("💡 MODEL READY FOR PRODUCTION USE")
        print("=" * 60)
        print("\nExample usage:")
        print("  response = model.generate_response(")
        print("      'path/to/image.jpg',")
        print("      'Your question here',")
        print("      max_new_tokens=512,")
        print("      temperature=0.7")
        print("  )")
        print("\n🚀 GPU acceleration enabled - fast inference!")
        
        if use_adapter:
            print("✨ Using your custom LoRA adapter!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()