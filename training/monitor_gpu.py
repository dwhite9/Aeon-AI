"""
GPU Monitoring Script for Training

Monitors GPU memory usage, utilization, and temperature during training.
Useful for detecting memory issues and optimizing batch sizes.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import argparse

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")


class GPUMonitor:
    """Monitor NVIDIA GPU metrics"""

    def __init__(self):
        """Initialize NVML"""
        if not PYNVML_AVAILABLE:
            raise ImportError("pynvml not installed")

        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]

    def get_metrics(self) -> List[Dict]:
        """
        Get current GPU metrics

        Returns:
            List of metric dicts (one per GPU)
        """
        metrics = []

        for i, handle in enumerate(self.handles):
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total_gb = mem_info.total / 1024**3
            mem_used_gb = mem_info.used / 1024**3
            mem_free_gb = mem_info.free / 1024**3
            mem_percent = (mem_info.used / mem_info.total) * 100

            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
            except pynvml.NVMLError:
                power_w = None

            # Name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            metrics.append({
                "gpu_id": i,
                "name": name,
                "memory_total_gb": round(mem_total_gb, 2),
                "memory_used_gb": round(mem_used_gb, 2),
                "memory_free_gb": round(mem_free_gb, 2),
                "memory_percent": round(mem_percent, 1),
                "gpu_utilization": gpu_util,
                "memory_utilization": mem_util,
                "temperature_c": temp,
                "power_w": round(power_w, 1) if power_w else None,
                "timestamp": datetime.now().isoformat()
            })

        return metrics

    def print_metrics(self, metrics: List[Dict]):
        """Print metrics in a readable format"""
        print("\n" + "=" * 80)
        print(f"GPU Metrics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        for m in metrics:
            print(f"\nGPU {m['gpu_id']}: {m['name']}")
            print(f"  Memory: {m['memory_used_gb']:.2f} / {m['memory_total_gb']:.2f} GB ({m['memory_percent']:.1f}%)")
            print(f"  Utilization: GPU {m['gpu_utilization']}% | Memory {m['memory_utilization']}%")
            print(f"  Temperature: {m['temperature_c']}°C")
            if m['power_w']:
                print(f"  Power: {m['power_w']:.1f} W")

    def close(self):
        """Shutdown NVML"""
        pynvml.nvmlShutdown()


def monitor_continuous(interval: int = 5, output_file: Optional[str] = None):
    """
    Continuously monitor GPU and optionally log to file

    Args:
        interval: Seconds between measurements
        output_file: Optional file to write JSON logs
    """
    monitor = GPUMonitor()
    logs = []

    try:
        print(f"Monitoring GPU every {interval}s. Press Ctrl+C to stop.")

        while True:
            metrics = monitor.get_metrics()
            monitor.print_metrics(metrics)

            # Save to logs
            if output_file:
                logs.extend(metrics)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")

    finally:
        monitor.close()

        # Save logs if requested
        if output_file and logs:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(logs, f, indent=2)

            print(f"Saved {len(logs)} measurements to: {output_path}")


def check_gpu_availability():
    """Check if GPU is available and print info"""
    if not PYNVML_AVAILABLE:
        print("❌ pynvml not available")
        return False

    try:
        monitor = GPUMonitor()
        metrics = monitor.get_metrics()
        monitor.print_metrics(metrics)
        monitor.close()
        return True
    except Exception as e:
        print(f"❌ Error accessing GPU: {e}")
        return False


def estimate_training_memory(
    model_size_b: float = 7.0,
    quantization: str = "4bit",
    lora_r: int = 8,
    batch_size: int = 1,
    seq_length: int = 2048
):
    """
    Estimate memory requirements for training

    Args:
        model_size_b: Model size in billions of parameters
        quantization: "4bit", "8bit", or "16bit"
        lora_r: LoRA rank
        batch_size: Batch size
        seq_length: Sequence length
    """
    print("\n" + "=" * 60)
    print("Memory Estimation for Training")
    print("=" * 60)

    # Model memory
    params = model_size_b * 1e9

    if quantization == "4bit":
        model_mem_gb = (params * 0.5) / 1e9  # 4 bits = 0.5 bytes
        quant_name = "4-bit (QLoRA)"
    elif quantization == "8bit":
        model_mem_gb = params / 1e9  # 8 bits = 1 byte
        quant_name = "8-bit"
    else:  # 16bit
        model_mem_gb = (params * 2) / 1e9  # 16 bits = 2 bytes
        quant_name = "16-bit"

    # LoRA adapter memory (much smaller)
    # Rough estimate: r * (d_model * 2) * num_layers
    lora_params = lora_r * (4096 * 2) * 32  # Mistral has 32 layers, 4096 d_model
    lora_mem_gb = (lora_params * 2) / 1e9  # 16-bit adapters

    # Optimizer states (Adam has 2 states per parameter)
    # With paged_adamw_8bit, these are in 8-bit
    trainable_params = lora_params
    optimizer_mem_gb = (trainable_params * 2 * 1) / 1e9  # 8-bit states

    # Gradients (same size as trainable params, in 16-bit)
    gradient_mem_gb = (trainable_params * 2) / 1e9

    # Activations (depends on batch size and sequence length)
    # Very rough estimate: batch_size * seq_length * hidden_dim * num_layers * bytes_per_activation
    activation_mem_gb = (batch_size * seq_length * 4096 * 32 * 2) / 1e9

    # Total
    total_mem_gb = (
        model_mem_gb +
        lora_mem_gb +
        optimizer_mem_gb +
        gradient_mem_gb +
        activation_mem_gb
    )

    # Print breakdown
    print(f"\nModel: {model_size_b}B parameters, {quant_name}")
    print(f"LoRA: rank {lora_r}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    print("\nMemory Breakdown:")
    print(f"  Model (quantized):  {model_mem_gb:>6.2f} GB")
    print(f"  LoRA adapters:      {lora_mem_gb:>6.2f} GB")
    print(f"  Optimizer states:   {optimizer_mem_gb:>6.2f} GB")
    print(f"  Gradients:          {gradient_mem_gb:>6.2f} GB")
    print(f"  Activations:        {activation_mem_gb:>6.2f} GB")
    print(f"  {'─' * 30}")
    print(f"  Total (estimated):  {total_mem_gb:>6.2f} GB")

    # Check if it fits
    print("\n" + "=" * 60)
    if total_mem_gb <= 10:
        print("✅ Should fit comfortably in 12GB VRAM")
    elif total_mem_gb <= 12:
        print("⚠️  Tight fit in 12GB VRAM (may need gradient checkpointing)")
    else:
        print("❌ Likely won't fit in 12GB VRAM")
        print("\nSuggestions:")
        print("  - Reduce batch_size to 1")
        print("  - Enable gradient_checkpointing")
        print("  - Reduce seq_length")
        print("  - Reduce lora_r")

    print("=" * 60)


def main():
    """Main CLI"""
    parser = argparse.ArgumentParser(description="GPU Monitoring for Training")
    parser.add_argument(
        "command",
        choices=["check", "monitor", "estimate"],
        help="Command to run"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for monitoring logs (JSON)"
    )
    parser.add_argument(
        "--model-size",
        type=float,
        default=7.0,
        help="Model size in billions (default: 7.0)"
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "16bit"],
        default="4bit",
        help="Quantization type (default: 4bit)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)"
    )

    args = parser.parse_args()

    if args.command == "check":
        check_gpu_availability()

    elif args.command == "monitor":
        monitor_continuous(args.interval, args.output)

    elif args.command == "estimate":
        estimate_training_memory(
            model_size_b=args.model_size,
            quantization=args.quantization,
            lora_r=args.lora_r,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )


if __name__ == "__main__":
    main()
