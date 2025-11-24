#!/bin/bash
# Quick Start Script for Tool-Aware Fine-tuning
# Runs the complete pipeline: prepare -> train -> evaluate

set -e  # Exit on error

echo "=========================================="
echo "Tool-Aware Fine-tuning Pipeline"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check GPU
echo -e "${YELLOW}[1/5] Checking GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}Error: CUDA not available in PyTorch${NC}"
    echo "Install PyTorch with CUDA: https://pytorch.org/get-started/locally/"
    exit 1
fi

echo -e "${GREEN}✓ GPU available${NC}"
echo ""

# Check dependencies
echo -e "${YELLOW}[2/5] Checking dependencies...${NC}"
if ! python -c "import transformers, peft, bitsandbytes" 2>/dev/null; then
    echo -e "${RED}Error: Required packages not installed${NC}"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo "You may not be able to download Mistral model."
    echo "Set it with: export HF_TOKEN='your_token'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Prepare dataset
echo -e "${YELLOW}[3/5] Preparing dataset...${NC}"
python prepare_dataset.py

if [ ! -f "data/train.json" ]; then
    echo -e "${RED}Error: Dataset preparation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset ready${NC}"
echo ""

# Estimate memory
echo -e "${YELLOW}[4/5] Estimating memory requirements...${NC}"
python monitor_gpu.py estimate \
  --model-size 7.0 \
  --quantization 4bit \
  --lora-r 8 \
  --batch-size 1 \
  --seq-length 2048

echo ""
read -p "Proceed with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Train
echo -e "${YELLOW}[5/5] Training model...${NC}"
echo "This will take 30-60 minutes. Monitor GPU with:"
echo "  python monitor_gpu.py monitor --interval 5"
echo ""

python train_tool_aware.py

if [ ! -d "output/lora_adapters" ]; then
    echo -e "${RED}Error: Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training complete${NC}"
echo ""

# Evaluate
echo -e "${YELLOW}[BONUS] Evaluating results...${NC}"
python evaluate.py

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Pipeline complete!${NC}"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Model: ./output/final_model/"
echo "  - Adapters: ./output/lora_adapters/"
echo "  - Evaluation: ./evaluation_results/"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results"
echo "  2. Test model with inference script"
echo "  3. Integrate with vLLM server"
echo ""
