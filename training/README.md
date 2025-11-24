# Tool-Aware Fine-tuning for Mistral 7B Instruct

**Comprehensive fine-tuning system to make Mistral 7B Instruct better at tool selection and usage.**

Optimized for RTX 4070 Ti (12GB VRAM) using QLoRA (4-bit quantization + LoRA adapters).

## 🎯 Overview

This fine-tuning system teaches Mistral 7B Instruct to:
- **Select the right tool** for each user query
- **Extract correct parameters** for tool calls
- **Provide reasoning** for tool selection

### Available Tools

The system trains on 4 tools matching your agent:

1. **`rag_retrieval`** - Search knowledge base for documents
2. **`web_search`** - Search the internet for current information
3. **`code_execution`** - Execute Python code in sandbox
4. **`direct_chat`** - Direct conversation without external tools

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with 12GB+ VRAM (e.g., RTX 4070 Ti, RTX 3060 12GB, RTX 4080)
- **RAM**: 16GB+ system RAM
- **Storage**: ~20GB free for model and checkpoints

### Software
- Python 3.9+
- CUDA 11.8+ or 12.0+
- Linux (recommended) or Windows with WSL2

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

**Key packages:**
- `transformers` - Model loading and training
- `peft` - LoRA adapters
- `bitsandbytes` - 4-bit quantization
- `torch` - PyTorch framework

### 2. Set Up Environment

Create a `.env` file or export these variables:

```bash
# Required for Mistral model access
export HF_TOKEN="your_huggingface_token_here"

# Database credentials (for query log extraction)
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="cipher_analytics"
export POSTGRES_USER="cipher"
export POSTGRES_PASSWORD="your_password"

# Optional: Weights & Biases for experiment tracking
export USE_WANDB="true"
export WANDB_PROJECT="mistral-tool-aware"
```

**Getting a HuggingFace Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Accept the Mistral model license at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

### 3. Prepare Dataset

```bash
python prepare_dataset.py
```

This will:
- ✅ Extract examples from your query logs (if database accessible)
- ✅ Generate 32+ synthetic examples for each tool
- ✅ Create ambiguous examples requiring careful reasoning
- ✅ Split into train/val/test sets (80/10/10)
- ✅ Format in Mistral instruction format
- ✅ Save to `./data/` directory

**Expected output:**
```
Train: ~80 examples
Val:   ~10 examples
Test:  ~10 examples
```

### 4. Check GPU Memory

Before training, verify your GPU can handle it:

```bash
python monitor_gpu.py check
```

Or estimate memory requirements:

```bash
python monitor_gpu.py estimate \
  --model-size 7.0 \
  --quantization 4bit \
  --lora-r 8 \
  --batch-size 1 \
  --seq-length 2048
```

**Expected:** ~9-11GB for training (comfortably fits 12GB)

### 5. Train the Model

```bash
python train_tool_aware.py
```

**Training time:** ~30-60 minutes for 3 epochs (depends on dataset size)

**What happens:**
1. Loads Mistral 7B Instruct in 4-bit quantization (~4.5GB)
2. Adds LoRA adapters (rank 8, ~200MB trainable params)
3. Trains for 3 epochs with gradient accumulation
4. Saves checkpoints every 100 steps
5. Saves final model to `./output/final_model/`
6. Saves LoRA adapters to `./output/lora_adapters/`

**Monitor training:**
```bash
# In another terminal
python monitor_gpu.py monitor --interval 5 --output gpu_logs.json
```

### 6. Evaluate Results

```bash
python evaluate.py
```

This will:
- ✅ Evaluate base model on test set
- ✅ Evaluate fine-tuned model on test set
- ✅ Compare accuracy, precision, recall, F1
- ✅ Show confusion matrix
- ✅ Calculate per-tool metrics
- ✅ Save detailed results to `./evaluation_results/`

**Example output:**
```
COMPARISON
==========================================
Base Model Accuracy:       45.23%
Fine-tuned Model Accuracy: 87.65%
Improvement:               +42.42% (+93.8%)

Per-Tool F1 Score Comparison:
Tool                 Base         Fine-tuned   Δ
rag_retrieval        0.42         0.89         +0.47
web_search           0.38         0.85         +0.47
code_execution       0.51         0.92         +0.41
direct_chat          0.49         0.84         +0.35
```

## 📁 Project Structure

```
training/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml               # Training configuration
│
├── prepare_dataset.py        # Dataset preparation script
├── train_tool_aware.py       # QLoRA fine-tuning script
├── evaluate.py               # Evaluation script
├── monitor_gpu.py            # GPU monitoring utility
│
├── data/                     # Generated datasets
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── sample_raw.json
│
├── output/                   # Training outputs
│   ├── lora_adapters/       # LoRA adapter weights (~100MB)
│   ├── final_model/         # Full model + adapters
│   └── checkpoint-*/        # Training checkpoints
│
└── evaluation_results/       # Evaluation outputs
    ├── base_model_metrics.json
    ├── finetuned_model_metrics.json
    ├── base_model_results.json
    └── finetuned_model_results.json
```

## ⚙️ Configuration

### Memory-Optimized Settings

The default configuration in `config.yaml` is optimized for 12GB VRAM:

```yaml
# QLoRA: 4-bit quantization
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

# LoRA: Small rank for memory efficiency
lora:
  r: 8              # Rank (trainable: ~4.5M params)
  alpha: 16         # Scaling factor

# Training: Small batch with accumulation
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch = 8

# Memory: Aggressive optimization
optimization:
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
```

### Adjusting for More VRAM

If you have **16GB+ VRAM**, you can:

```yaml
lora:
  r: 16  # Increase rank for better quality

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4  # Same effective batch

model:
  max_seq_length: 4096  # Use full context
```

### Adjusting for Less VRAM

If you have **10GB VRAM**, you can:

```yaml
lora:
  r: 4  # Reduce rank

training:
  gradient_accumulation_steps: 16  # More accumulation

model:
  max_seq_length: 1024  # Shorter sequences
```

## 🔧 Advanced Usage

### Custom Dataset

To use your own dataset instead of query logs:

1. Create JSON files in this format:

```json
[
  {
    "instruction": "Full instruction with tools...",
    "response": "Tool: rag_retrieval\nParameters: {...}\nReasoning: ...",
    "metadata": {
      "source": "custom",
      "tool": "rag_retrieval"
    }
  }
]
```

2. Save as `data/train.json`, `data/val.json`, `data/test.json`

3. Run training directly: `python train_tool_aware.py`

### Multi-Stage Training

For better results, train in stages:

```bash
# Stage 1: Train on synthetic data (learns format)
python train_tool_aware.py  # Uses mostly synthetic examples

# Stage 2: Fine-tune on real data (learns patterns)
# Edit prepare_dataset.py to use more query logs
python prepare_dataset.py
python train_tool_aware.py --output_dir ./output_stage2
```

### Inference with Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./output/lora_adapters")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Generate
prompt = "<s>[INST] What is vLLM? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Merging LoRA Adapters

To merge adapters into base model for faster inference:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./output/lora_adapters")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

## 📊 Monitoring Training

### Weights & Biases

Enable W&B for experiment tracking:

```bash
export USE_WANDB="true"
export WANDB_PROJECT="mistral-tool-aware"
wandb login
python train_tool_aware.py
```

View metrics at: https://wandb.ai/

### TensorBoard

Alternatively, use TensorBoard:

```bash
tensorboard --logdir ./output
```

### GPU Monitoring

Monitor GPU usage during training:

```bash
# Check current status
python monitor_gpu.py check

# Continuous monitoring
python monitor_gpu.py monitor --interval 5

# Save to file
python monitor_gpu.py monitor --interval 5 --output gpu_logs.json
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Symptom:** `torch.cuda.OutOfMemoryError`

**Solutions:**
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to 16
3. Reduce `lora_r` to 4
4. Reduce `max_seq_length` to 1024
5. Close other GPU programs (browser, vLLM server, etc.)

### Slow Training

**Symptom:** <5 steps/second

**Solutions:**
1. Ensure `gradient_checkpointing: true`
2. Use `optim: "paged_adamw_8bit"`
3. Reduce `logging_steps` to 50
4. Disable wandb if not needed

### Poor Accuracy

**Symptom:** Fine-tuned model not better than base

**Solutions:**
1. Check dataset quality with `data/sample_raw.json`
2. Increase training epochs to 5
3. Increase `lora_r` to 16
4. Add more training examples (aim for 200+)
5. Check for class imbalance (equal examples per tool)

### Model Not Loading

**Symptom:** `HfHubHTTPError` or access denied

**Solutions:**
1. Check HF_TOKEN is set: `echo $HF_TOKEN`
2. Accept model license at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
3. Check token has "Read" permission
4. Try: `huggingface-cli login`

## 📈 Performance Expectations

### RTX 4070 Ti (12GB)

| Metric | Value |
|--------|-------|
| **Memory Usage** | 9-11 GB |
| **Training Speed** | ~5-8 steps/sec |
| **Time per Epoch** | ~10-20 min |
| **Total Training** | ~30-60 min |

### Expected Improvements

| Metric | Base Model | After Fine-tuning | Improvement |
|--------|-----------|------------------|-------------|
| **Tool Accuracy** | 40-50% | 80-90% | +40-50% |
| **RAG F1** | 0.35-0.45 | 0.85-0.95 | +0.40-0.50 |
| **Web Search F1** | 0.30-0.40 | 0.80-0.90 | +0.50-0.60 |
| **Code Exec F1** | 0.45-0.55 | 0.90-0.95 | +0.35-0.45 |
| **Chat F1** | 0.40-0.50 | 0.80-0.90 | +0.40-0.50 |

## 🔄 Integration with Existing System

### Option 1: Replace vLLM Model

Replace the base model in your vLLM inference server:

```bash
# Stop current vLLM server
# Start with fine-tuned model
python -m vllm.entrypoints.openai.api_server \
  --model ./training/output/final_model \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16
```

### Option 2: Load Adapters at Runtime

Keep base model, load adapters when needed:

```python
# In services/agent/agent.py
from peft import PeftModel

# Load base model
base_model = load_base_model()

# Load adapters for tool-aware routing
model = PeftModel.from_pretrained(base_model, "./training/output/lora_adapters")
```

### Option 3: Export to GGUF for llama.cpp

For CPU/edge deployment:

```bash
# Merge adapters
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
model = PeftModel.from_pretrained(base, './output/lora_adapters')
merged = model.merge_and_unload()
merged.save_pretrained('./merged_model')
"

# Convert to GGUF (requires llama.cpp)
python llama.cpp/convert.py ./merged_model --outtype f16 --outfile mistral-tool-aware.gguf
```

## 📚 Further Reading

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient fine-tuning
- [Mistral Documentation](https://docs.mistral.ai/) - Model details
- [HuggingFace PEFT](https://huggingface.co/docs/peft) - Parameter-efficient fine-tuning

## 🤝 Contributing

Found a bug or want to improve the training? Contributions welcome!

1. Test your changes with `pytest tests/`
2. Ensure training runs without OOM
3. Document any new configuration options
4. Update this README

## 📄 License

This training code is part of the Aeon-AI project. The fine-tuned model inherits the Mistral license (Apache 2.0).

## 🙏 Acknowledgments

- **Mistral AI** - Base model
- **HuggingFace** - Transformers and PEFT libraries
- **Tim Dettmers** - bitsandbytes quantization
- **Microsoft** - LoRA technique

---

**Need help?** Open an issue with:
- GPU model and VRAM
- Full error message
- Output of `python monitor_gpu.py check`
- Contents of `config.yaml`
