"""
MLX Vision-Language Model Backend with LoRA Adapter
Optimized for Apple Silicon M1/M2/M3 Macs
Uses MLX for efficient inference with quantization support
"""

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import os
from PIL import Image
import requests
from io import BytesIO

class VisionLanguageModelMLX:
    def __init__(self, model_name="qwen2-vl-7b-instruct", adapter_path=None):
        """
        Initialize the MLX-based vision-language model
        
        Args:
            model_name: Base model to use (will be quantized automatically)
            adapter_path: Optional path to LoRA adapter folder
        """
        print("🚀 Initializing MLX Vision-Language Model for Apple Silicon...")
        print("✅ Using MLX (optimized for M1/M2/M3)")
        
        self.adapter_path = adapter_path
        self.model_name = model_name
        
        # Load model with MLX (automatically handles quantization)
        print(f"\n📥 Loading model: {model_name}")
        print("   (MLX will quantize automatically for efficient inference)")
        print("   (First run will download model - may take a few minutes)")
        
        try:
            # Load the base model with MLX
            self.model, self.processor = load(
                model_name,
                quantize=True  # Enable quantization for M1
            )
            print("✅ Model loaded successfully with MLX quantization")
            
            # Load LoRA adapter if provided
            if adapter_path and os.path.exists(adapter_path):
                print(f"\n📥 Loading LoRA adapter from {adapter_path}...")
                # Note: MLX LoRA loading may require converting adapter format
                # For now, we'll note this limitation
                print("⚠️  Note: MLX LoRA adapter loading is limited.")
                print("   The base model will work, but custom adapter may need conversion.")
                print("   See: https://github.com/ml-explore/mlx-examples for details")
            
            print("\n✨ Model ready for inference!\n")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Trying alternative approach...")
            # Fallback to standard loading
            try:
                self.model, self.processor = load(model_name)
                print("✅ Model loaded (without quantization)")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                raise
    
    def generate_response(self, image_path_or_url, prompt, max_tokens=100, temperature=0.7):
        """
        Generate a response based on an image and text prompt
        
        Args:
            image_path_or_url: Local path or URL to image
            prompt: Text prompt/question about the image
            max_tokens: Maximum length of generated response
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        
        Returns:
            Generated text response
        """
        try:
            # Load image
            if image_path_or_url.startswith('http'):
                print("Downloading image from URL...")
                response = requests.get(image_path_or_url)
                image = Image.open(BytesIO(response.content))
                # Save temporarily for MLX-VLM
                temp_path = "/tmp/temp_image.jpg"
                image.save(temp_path)
                image_path_or_url = temp_path
                print(f"Image saved to {temp_path}")
            
            print(f"Calling MLX generate with image: {image_path_or_url}")
            print(f"Prompt: {prompt}")
            
            # Generate response using MLX
            # Note: MLX-VLM expects (model, processor, prompt, image) order
            output = generate(
                self.model,
                self.processor,
                prompt,  # Prompt comes BEFORE image in MLX-VLM!
                image_path_or_url,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=True  # Enable verbose for debugging
            )
            
            print(f"Raw output type: {type(output)}")
            print(f"Raw output: {output}")
            
            return output
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Full error traceback:\n{error_details}")
            return f"Error generating response: {str(e)}"


def run_test(model):
    """Test the model with a sample image"""
    print("=" * 60)
    print("🧪 RUNNING TEST")
    print("=" * 60)
    
    # Download a test image
    test_image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    test_prompt = "What do you see in this image? Describe it briefly."
    
    print(f"\n📸 Test Image: {test_image_url}")
    print(f"💬 Prompt: {test_prompt}")
    print("\n⏳ Downloading image and generating response...\n")
    
    try:
        # Download and save image
        import requests
        from PIL import Image
        from io import BytesIO
        
        response = requests.get(test_image_url)
        image = Image.open(BytesIO(response.content))
        temp_path = "/tmp/test_image.jpg"
        image.save(temp_path)
        print(f"✅ Image saved to {temp_path}")
        
        # Generate response
        print("⏳ Generating response (may take 30-60 seconds on M1)...\n")
        output = model.generate_response(temp_path, test_prompt, max_tokens=100)
        
        print("-" * 60)
        print("🤖 MODEL RESPONSE:")
        print("-" * 60)
        print(output)
        print("-" * 60)
        print("\n✅ Test complete!\n")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to initialize and test the model"""
    
    # CONFIGURATION
    ADAPTER_PATH = "./path/to/your/adapter/folder"  # Update this if you have adapters
    
    # MLX model selection - use full Hugging Face path
    # MLX-VLM compatible Qwen2-VL models:
    MODEL_NAME = "mlx-community/Qwen2-VL-7B-Instruct-4bit"  # 4-bit quantized (recommended for M1)
    # Alternative options:
    # MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"  # Smaller, faster
    # MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Full precision (needs more RAM)
    
    print("=" * 60)
    print("MLX Vision-Language Model for Apple Silicon")
    print("=" * 60)
    print(f"\n📋 Model: {MODEL_NAME}")
    print("🍎 Optimized for M1/M2/M3 Macs")
    print("⚡ Using quantization for efficient inference\n")
    
    # Check if adapter path exists (optional)
    use_adapter = False
    if os.path.exists(ADAPTER_PATH):
        adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            print(f"📁 Found adapter at: {ADAPTER_PATH}")
            use_adapter = True
        else:
            print(f"⚠️  Adapter path exists but no adapter_model.safetensors found")
            ADAPTER_PATH = None
    else:
        print("ℹ️  No adapter path specified - using base model only")
        ADAPTER_PATH = None
    
    try:
        # Initialize model
        model = VisionLanguageModelMLX(
            model_name=MODEL_NAME,
            adapter_path=ADAPTER_PATH if use_adapter else None
        )
        
        # Run test
        run_test(model)
        
        # Ready for use
        print("\n" + "=" * 60)
        print("💡 MODEL READY - You can now use it in your application!")
        print("=" * 60)
        print("\nExample usage:")
        print("  response = model.generate_response(")
        print("      'path/to/image.jpg',")
        print("      'Your question here',")
        print("      max_tokens=100,")
        print("      temperature=0.7")
        print("  )")
        print("\n🚀 Inference will be fast and memory-efficient with MLX!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure mlx and mlx-vlm are installed:")
        print("   pip install mlx mlx-vlm")
        print("2. Check if the model name is correct")
        print("3. Ensure you have internet connection for first download")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()