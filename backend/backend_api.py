"""
Flask Backend API for Qwen2.5-VL Chatbot
Handles model inference with text and image inputs
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import io
import base64
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for model and processor
model = None
processor = None
device = None

# Configuration
BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
ADAPTER_PATH = "."  # Update this to your adapter path
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

def load_model_and_processor():
    """Load the base model and LoRA adapters"""
    global model, processor, device
    
    logger.info("Loading model and processor...")
    
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
        
        # Load base model
        logger.info("Loading base model (this may take a few minutes)...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Load LoRA adapters
        logger.info(f"Loading LoRA adapters from {ADAPTER_PATH}...")
        
        # Check if adapter path exists
        if not os.path.exists(ADAPTER_PATH):
            logger.error(f"Adapter path does not exist: {ADAPTER_PATH}")
            logger.error("LoRA adapters will NOT be loaded!")
            logger.info("✓ Base model loaded (WITHOUT LoRA adapters)")
            model = base_model
        else:
            # Check for adapter files
            adapter_files = os.listdir(ADAPTER_PATH)
            logger.info(f"Files in adapter directory: {adapter_files}")
            
            if 'adapter_config.json' not in adapter_files:
                logger.error("adapter_config.json not found in adapter directory!")
                logger.error("LoRA adapters will NOT be loaded!")
                model = base_model
            else:
                model = PeftModel.from_pretrained(
                    base_model,
                    ADAPTER_PATH,
                    is_trainable=False
                )
                logger.info("✓ LoRA adapters loaded successfully!")
                
                # Verify adapters are active
                if hasattr(model, 'peft_config'):
                    logger.info("✓ PEFT adapters are ACTIVE")
                    for adapter_name in model.peft_config.keys():
                        logger.info(f"  - Active adapter: {adapter_name}")
                else:
                    logger.warning("⚠ No PEFT config found - adapters may not be active!")
        
        logger.info("✓ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_response(text_input, image_input=None):
    """
    Generate response from the model
    
    Args:
        text_input: Text prompt from user
        image_input: PIL Image object (optional)
    
    Returns:
        Generated text response
    """
    try:
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Resize image if too large (to prevent OOM)
        if image_input is not None:
            max_size = 1024
            width, height = image_input.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                image_input = image_input.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Prepare messages
        if image_input is not None:
            # Vision + Language input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_input}
                    ]
                }
            ]
        else:
            # Text-only input
            messages = [
                {"role": "user", "content": text_input}
            ]
        
        # Format the prompt
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        if image_input is not None:
            inputs = processor(
                text=[text],
                images=[image_input],
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with memory-efficient settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(MAX_NEW_TOKENS, 256),  # Limit tokens for images
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        
        # Decode
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]
        
        # Clear GPU cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract only the assistant's response
        # Remove the input prompt from the output
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "").strip()
        elif "assistant\n" in generated_text:
            # Handle case where format is: system\n...\nuser\n...\nassistant\n...
            response = generated_text.split("assistant\n")[-1].strip()
        else:
            response = generated_text
        
        return response
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU out of memory: {e}")
        # Try to recover
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise Exception("GPU out of memory. Please try with a smaller image or text-only message.")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    Accepts: JSON with 'message' (text) and optional 'image' (base64)
    Returns: JSON with 'response' (text)
    """
    try:
        data = request.json
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        text_input = data['message']
        image_input = None
        
        # Process image if provided
        if 'image' in data and data['image']:
            try:
                # Decode base64 image
                image_data = data['image']
                
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                image_bytes = base64.b64decode(image_data)
                image_input = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                logger.info(f"Received image: {image_input.size}")
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return jsonify({'error': 'Invalid image data'}), 400
        
        logger.info(f"Processing request: text='{text_input[:50]}...', has_image={image_input is not None}")
        
        # Generate response
        response = generate_response(text_input, image_input)
        
        logger.info(f"Generated response: '{response[:100]}...'")
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current model configuration"""
    return jsonify({
        'base_model': BASE_MODEL,
        'adapter_path': ADAPTER_PATH,
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'device': str(device) if device else 'unknown'
    })

@app.route('/config', methods=['POST'])
def update_config():
    """Update generation parameters"""
    global MAX_NEW_TOKENS, TEMPERATURE, TOP_P
    
    try:
        data = request.json
        
        if 'max_new_tokens' in data:
            MAX_NEW_TOKENS = int(data['max_new_tokens'])
        
        if 'temperature' in data:
            TEMPERATURE = float(data['temperature'])
        
        if 'top_p' in data:
            TOP_P = float(data['top_p'])
        
        return jsonify({
            'message': 'Configuration updated',
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE,
            'top_p': TOP_P
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Flask server...")
    
    if load_model_and_processor():
        logger.info("Model loaded successfully. Starting server...")
        # Run on all interfaces so it's accessible from other machines
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Server not started.")
        exit(1)