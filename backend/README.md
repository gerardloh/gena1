---
base_model: ''
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/qwen2.5-vl-7b-instruct-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
### Framework versions

- PEFT 0.17.1




# Qwen2.5-VL LoRA Model Loading and Testing

This repository contains scripts to load the Qwen2.5-VL-7B base model with your trained LoRA adapters and test the model to ensure your training weights are preserved.

## Overview

- **Base Model**: `unsloth/qwen2.5-vl-7b-instruct-bnb-4bit` (4-bit quantized)
- **LoRA Adapters**: Your fine-tuned adapters from the `backend` directory
- **Testing**: Automated tests to verify the model works correctly

## Files Included

1. **load_and_test_model.py** - Main Python script that:
   - Loads the base model with 4-bit quantization
   - Connects your LoRA adapters
   - Tests text generation
   - (Optional) Tests vision-language generation

2. **run_model_test.sh** - SLURM job submission script for the Origami GPU cluster

3. **setup_and_run.sh** - Simple bash script for local testing (without SLURM)

4. **requirements.txt** - Python dependencies

## Prerequisites

### Python Packages
```bash
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0
peft>=0.9.0
bitsandbytes>=0.42.0
pillow>=10.0.0
safetensors>=0.4.0
```

### System Requirements
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- CUDA 11.8+ or 12.x
- Python 3.8+

## Setup Instructions

### Step 1: Prepare Your Files

1. **Update the adapter path** in `load_and_test_model.py`:
   ```python
   # Line ~353 - Update this path to point to your adapter directory
   ADAPTER_PATH = "./backend"  # Change to your actual path
   ```

2. Your adapter directory should contain:
   - `adapter_config.json`
   - `adapter_model.safetensors` (or `adapter_model.bin`)
   - Other adapter-related files

### Step 2: Install Dependencies

On the Origami cluster or your local machine:

```bash
# Using pip
pip install --break-system-packages -r requirements.txt

# Or if you're in a virtual environment
pip install -r requirements.txt
```

## Running the Script

### Option 1: SLURM Job Submission (Recommended for Origami Cluster)

1. **Make the script executable**:
   ```bash
   chmod +x run_model_test.sh
   ```

2. **Create logs directory**:
   ```bash
   mkdir -p logs
   ```

3. **Submit the job**:
   ```bash
   sbatch run_model_test.sh
   ```

4. **Check job status**:
   ```bash
   squeue -u $USER
   ```

5. **View output**:
   ```bash
   # Real-time monitoring
   tail -f logs/job_<JOBID>.out
   
   # View after completion
   cat logs/job_<JOBID>.out
   ```

### Option 2: Direct Execution (Local or Interactive Node)

1. **Make the script executable**:
   ```bash
   chmod +x setup_and_run.sh
   ```

2. **Run the script**:
   ```bash
   ./setup_and_run.sh
   ```

   Or run directly with Python:
   ```bash
   python3 load_and_test_model.py
   ```

## Configuration Options

### In `load_and_test_model.py`:

1. **Base Model** (Line ~348):
   ```python
   BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
   ```

2. **Adapter Path** (Line ~351):
   ```python
   ADAPTER_PATH = "./backend"  # Update this!
   ```

3. **Test Image** (Line ~354):
   ```python
   TEST_IMAGE = None  # Set to image path to test vision capabilities
   TEST_IMAGE = "./test_image.jpg"  # Example
   ```

### In `run_model_test.sh`:

Adjust SLURM parameters based on your needs:

```bash
#SBATCH --gres=gpu:1              # Number of GPUs (1 is usually enough)
#SBATCH --cpus-per-task=4         # CPU cores
#SBATCH --mem=32G                 # Memory (increase if needed)
#SBATCH --time=01:00:00           # Time limit
```

## Expected Output

When the script runs successfully, you should see:

```
================================================================================
Environment Setup
================================================================================
✓ CUDA is available
  GPU Device: NVIDIA A100 40GB
  Number of GPUs: 1
  CUDA Version: 12.1
  PyTorch Version: 2.x.x

================================================================================
Loading Base Model
================================================================================
Model: unsloth/qwen2.5-vl-7b-instruct-bnb-4bit
Loading with 4-bit quantization (BitsAndBytes)...
Loading processor...
Loading base model (this may take a few minutes)...
✓ Base model loaded successfully!

================================================================================
Loading LoRA Adapters
================================================================================
Adapter path: ./backend

Adapter files found:
  - adapter_config.json
  - adapter_model.safetensors
  - ...

✓ LoRA adapters loaded successfully!

================================================================================
Model Information
================================================================================
Total parameters: 7,615,000,000
Trainable parameters: 4,194,304
Trainable %: 0.06%

✓ LoRA adapters are active

================================================================================
Testing Model - Text Only
================================================================================
Prompt: What is the capital of France?

Generated Response:
--------------------------------------------------------------------------------
The capital of France is Paris.
--------------------------------------------------------------------------------

✓ Text generation successful!

================================================================================
Summary
================================================================================
✓ Base model loaded successfully
✓ LoRA adapters connected
✓ Model is ready for inference

Your fine-tuned model is working! Training weights are preserved.
```

## Troubleshooting

### Issue: "Adapter path does not exist"
**Solution**: Update the `ADAPTER_PATH` variable in `load_and_test_model.py` to point to your actual adapter directory.

### Issue: "Missing required adapter files"
**Solution**: Ensure your adapter directory contains:
- `adapter_config.json`
- `adapter_model.safetensors` (or `adapter_model.bin`)

### Issue: "CUDA out of memory"
**Solution**: 
- Increase memory allocation in SLURM script (`--mem=64G`)
- Reduce batch size or generation length
- Use a node with more GPU memory

### Issue: "Module not found"
**Solution**: Install missing packages:
```bash
pip install --break-system-packages <package_name>
```

### Issue: Model loads but generation fails
**Solution**: 
- Check adapter compatibility with base model
- Verify `adapter_config.json` is valid
- Try with a simpler prompt

## Testing with Images

To test vision-language capabilities:

1. **Add a test image** to your directory

2. **Update the script** (Line ~354 in `load_and_test_model.py`):
   ```python
   TEST_IMAGE = "./path/to/your/test_image.jpg"
   ```

3. **Run the script** - it will automatically test both text-only and vision-language generation

## Advanced Usage

### Custom Prompts

Modify the `test_model_text_only()` function to use your own prompts:

```python
test_prompt = "Your custom prompt here"
```

### Generation Parameters

Adjust generation settings in the `model.generate()` calls:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,      # Maximum length
    do_sample=True,          # Use sampling
    temperature=0.7,         # Randomness (0.1-1.0)
    top_p=0.9,              # Nucleus sampling
    top_k=50,               # Top-k sampling
)
```

### Saving the Merged Model

To save the model with adapters merged (for deployment):

```python
# Merge adapters into base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./merged_model")
processor.save_pretrained("./merged_model")
```

## Additional Resources

- **Origami Cluster Docs**: https://violet.scis.dev/docs/
- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **Unsloth Documentation**: https://github.com/unslothai/unsloth

## Support

If you encounter issues:

1. Check the error logs in `logs/` directory
2. Verify GPU availability with `nvidia-smi`
3. Ensure all dependencies are installed correctly
4. Check that adapter files are in the correct format

## License

This code is provided as-is for educational and research purposes.