# Qwen2.5-VL Chatbot Frontend & Backend

Complete chatbot interface for interacting with your fine-tuned Qwen2.5-VL model with LoRA adapters.

## 🎨 Features

- ✅ **Text + Image Input**: Send text messages with optional image attachments
- ✅ **Vision-Language AI**: Process images and answer questions about them
- ✅ **Real-time Chat**: Modern chat interface with typing indicators
- ✅ **Configurable**: Adjust model parameters (temperature, max tokens)
- ✅ **Beautiful UI**: Gradient design with smooth animations
- ✅ **Easy Deployment**: Simple setup on Origami cluster

## 📁 Files Included

1. **backend_api.py** - Flask API server that runs the model
2. **run_api_server.sh** - SLURM script to deploy API on GPU
3. **chatbot.html** - Standalone web interface (no build needed!)
4. **ChatbotUI.jsx** - React component (optional, for advanced users)

## 🚀 Quick Start

### Step 1: Update the Adapter Path

Edit `backend_api.py` (line 24):
```python
ADAPTER_PATH = "./backend"  # Change to your actual adapter path
```

### Step 2: Start the Backend API

```bash
# Make script executable
chmod +x run_api_server.sh

# Create logs directory
mkdir -p logs

# Submit the job to GPU
sbatch run_api_server.sh

# Check job status
squeue -u $USER

# Get the server URL from logs
tail -f logs/api_<JOBID>.out
# Look for: "Server will be accessible at: http://X.X.X.X:5000"
```

### Step 3: Create SSH Tunnel (Important!)

The GPU server is on an internal network, so you need an SSH tunnel to access it from your computer.

**On your local computer** (NOT on Origami), open a new terminal and run:

```bash
# Replace with your actual Origami login hostname
ssh -L 5000:10.2.1.35:5000 gerard.loh.2022@origami.scis.smu.edu.sg
```

**Important Notes:**
- Keep this terminal window **open** while using the chatbot
- If you get a hostname error, use the same hostname you normally use to SSH into Origami
- Replace `10.2.1.35` with your actual GPU node IP from Step 2
- The tunnel maps your local `localhost:5000` to the GPU server

### Step 4: Open the Frontend

1. **Download `chatbot.html`** to your local computer
2. **Open it in a web browser** (Chrome, Firefox, Edge, Safari)
3. **Update the API URL** in Settings:
   - Click the settings icon (⚙️)
   - Change API URL to: `http://localhost:5000`
   - Click "Save Settings"

### Step 5: Start Chatting!

- Type a message and press Enter or click Send
- Upload an image using the image icon (📷)
- Ask questions about the uploaded image
- Get responses from your fine-tuned model

## 📋 Backend API Details

### Endpoints

**POST /chat**
- Send messages to the model
- Request body:
  ```json
  {
    "message": "What's in this image?",
    "image": "data:image/jpeg;base64,/9j/4AAQ..." // optional
  }
  ```
- Response:
  ```json
  {
    "response": "This image shows a cat...",
    "timestamp": "2025-11-05T12:34:56"
  }
  ```

**GET /health**
- Check if server is running
- Response:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda"
  }
  ```

**GET /config**
- Get current model settings

**POST /config**
- Update model parameters
- Request body:
  ```json
  {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }
  ```

### Resource Requirements

- **GPU**: 1 GPU (required)
- **RAM**: 32GB
- **CPUs**: 4 cores
- **Time**: Up to 24 hours

## 🎨 Frontend Options

### Option 1: Standalone HTML (Recommended)

**File**: `chatbot.html`

**Pros**:
- ✅ No installation needed
- ✅ Works immediately in any browser
- ✅ Easy to customize
- ✅ Can run from USB drive or local file

**Usage**:
1. Download the HTML file
2. Double-click to open in browser
3. Update API URL in settings
4. Start chatting!

### Option 2: React Component

**File**: `ChatbotUI.jsx`

**Pros**:
- ✅ Better for integration into existing React apps
- ✅ More maintainable for large projects

**Setup**:
```bash
# In your React project:
npm install lucide-react

# Copy ChatbotUI.jsx to your src/components folder
# Import and use:
import ChatbotUI from './components/ChatbotUI';
function App() {
  return <ChatbotUI />;
}
```

## 🔧 Configuration

### Backend Configuration

Edit `backend_api.py`:

```python
# Model settings (lines 22-28)
BASE_MODEL = "unsloth/qwen2.5-vl-7b-instruct-bnb-4bit"
ADAPTER_PATH = "./backend"  # Your adapter path

# Generation settings
MAX_NEW_TOKENS = 512    # Maximum response length
TEMPERATURE = 0.7        # Randomness (0.1-1.0)
TOP_P = 0.9             # Nucleus sampling
```

### Frontend Configuration

In the web interface:
1. Click Settings icon (⚙️)
2. Adjust:
   - **API URL**: Backend server address
   - **Max Tokens**: Response length (128-2048)
   - **Temperature**: Response randomness (0.1-2.0)

## 📝 Usage Examples

### Text-Only Chat

```
You: What is the capital of France?
AI: The capital of France is Paris. It is located in the north-central...
```

### Image Description

```
You: [uploads photo of a cat] What's in this image?
AI: This image shows a tabby cat sitting on a windowsill. The cat appears...
```

### Image + Specific Question

```
You: [uploads chart] What are the key trends shown in this chart?
AI: Based on the chart, there are three main trends: 1) Sales increased...
```

## 🔍 Monitoring

### Check Backend Status

```bash
# View API logs
tail -f logs/api_<JOBID>.out

# Check if model is loaded
curl http://<NODE_IP>:5000/health

# Test the API
curl -X POST http://<NODE_IP>:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Check Resource Usage

```bash
# View GPU usage
srun --account=is469 --partition=student --qos=studentqos \
     --gres=gpu:1 --jobid=<JOBID> nvidia-smi

# View job info
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

## 🛠️ Troubleshooting

### Issue: "Unable to connect to server"

**Symptoms**: Red error message in chat

**Solutions**:

1. **Check if backend is running:**
   ```bash
   squeue -u $USER
   ```

2. **Verify SSH tunnel is active:**
   - Make sure the SSH tunnel terminal is still open
   - Test tunnel: `curl http://localhost:5000/health`
   - If no response, recreate the tunnel

3. **Check API URL in Settings:**
   - Should be `http://localhost:5000` (when using SSH tunnel)
   - NOT `http://10.2.1.35:5000` (internal IP won't work from outside)

4. **Test backend directly on cluster:**
   ```bash
   ssh origami
   curl http://10.2.1.35:5000/health
   ```

5. **Check firewall/network settings**

6. **Look at backend logs for errors:**
   ```bash
   cat logs/api_<JOBID>.err
   ```

### Issue: "Could not resolve hostname" when creating SSH tunnel

**Solutions**:
- Check what hostname you normally use to SSH into Origami
- Try common variations:
  - `origami.scis.smu.edu.sg`
  - `violet.scis.smu.edu.sg`
  - `origami.smu.edu.sg`
- Use the same hostname from your regular SSH login

### Issue: SSH tunnel disconnects

**Solutions**:
- The tunnel will disconnect if the SSH session times out
- Add to your `~/.ssh/config`:
  ```
  Host origami
      ServerAliveInterval 60
      ServerAliveCountMax 3
  ```
- Or use `autossh` for persistent tunnels:
  ```bash
  autossh -M 0 -L 5000:10.2.1.35:5000 user@origami
  ```

### Issue: Backend crashes or stops

**Solutions**:
1. Check error logs:
   ```bash
   cat logs/api_<JOBID>.err
   ```
2. Common causes:
   - Out of GPU memory → Reduce max_new_tokens
   - Job time limit reached → Resubmit job
   - CUDA error → Check PyTorch installation

### Issue: Slow responses

**Causes**:
- Large images (resize to 1024x1024 max)
- High max_new_tokens (reduce to 256-512)
- Multiple users (GPU is shared)

**Solutions**:
1. Reduce image size before uploading
2. Lower max_new_tokens in settings
3. Request dedicated GPU time

### Issue: Model gives wrong/weird answers

**Causes**:
- Wrong adapter path
- Adapters not compatible with base model
- Temperature too high

**Solutions**:
1. Verify ADAPTER_PATH in backend_api.py
2. Check adapter files exist and are correct
3. Lower temperature to 0.3-0.5 for more focused responses

## 🔒 Security Notes

⚠️ **Important**: This setup is for development/research use.

For production deployment:
- Add authentication (API keys, JWT tokens)
- Use HTTPS instead of HTTP
- Implement rate limiting
- Add input validation
- Set up proper CORS policies

## 📊 Performance Tips

### Backend Optimization
- Use batch processing for multiple requests
- Cache common responses
- Monitor GPU memory usage
- Consider using model quantization

### Frontend Optimization
- Compress images before upload
- Implement client-side image resizing
- Add request debouncing
- Cache API responses

## 🎯 Advanced Features

### Custom System Prompts

Edit `backend_api.py` to add system prompts:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant specialized in..."},
    {"role": "user", "content": text_input}
]
```

### Conversation History

Modify frontend to send conversation context:

```javascript
const conversationHistory = messages.map(m => ({
  role: m.role,
  content: m.content
}));

fetch(`${apiUrl}/chat`, {
  body: JSON.stringify({
    message: textToSend,
    image: imageToSend,
    history: conversationHistory  // Add this
  })
});
```

### Multi-Image Support

Backend already supports vision-language, extend to handle multiple images:

```python
images = [Image.open(...) for img in image_list]
inputs = processor(text=[text], images=images, ...)
```

## 📚 Additional Resources

- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **Unsloth**: https://github.com/unslothai/unsloth
- **PEFT/LoRA**: https://huggingface.co/docs/peft
- **Flask**: https://flask.palletsprojects.com/

## 💡 Tips

1. **Start small**: Test with text-only before adding images
2. **Monitor GPU**: Keep an eye on `nvidia-smi` output
3. **Save settings**: Write down working temperature/token values
4. **Backup adapters**: Keep copies of your trained adapters
5. **Log everything**: Backend logs help debug issues

## 📞 Support

If you encounter issues:
1. Check logs first (`logs/api_*.out` and `logs/api_*.err`)
2. Verify all paths are correct
3. Test with simple text messages first
4. Check GPU availability with `nvidia-smi`

---

**Happy Chatting! 🎉**

Your fine-tuned Qwen2.5-VL model is now accessible through a beautiful chat interface!