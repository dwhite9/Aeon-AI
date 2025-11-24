"""
Test Inference Script

Quick script to test the fine-tuned model's tool selection.
Useful for manual testing and debugging.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def load_model(base_model_name: str, adapter_path: str = None, hf_token: str = None):
    """Load model with optional adapters"""
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        print(f"Loading adapters: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


TOOL_DEFINITIONS = """You are an AI agent with access to the following tools:

Tool: rag_retrieval
Description: Searches the knowledge base for relevant information. Use when the user asks about specific documents, facts, or information stored in the system.
Parameters: {"query": "string", "limit": "integer (default: 3)"}

Tool: web_search
Description: Searches the internet for current information. Use when user asks about recent events, current data, or info not in the knowledge base.
Parameters: {"query": "string", "num_results": "integer (default: 5)"}

Tool: code_execution
Description: Executes Python code in a secure sandbox. Use when user asks to run code, perform calculations, or test implementations.
Parameters: {"code": "string", "description": "string"}

Tool: direct_chat
Description: Direct conversation with AI without external tools. Use for general chat, creative tasks, code generation when no retrieval/search needed.
Parameters: {"message": "string", "context": "string (optional)"}

User Query: {query}

Analyze the query and determine which tool to use. Respond with:
1. Tool name
2. Parameters (as JSON)
3. Brief reasoning

Format: Tool: <name>
Parameters: <json>
Reasoning: <explanation>"""


def format_prompt(query: str) -> str:
    """Format query as Mistral instruction"""
    instruction = TOOL_DEFINITIONS.format(query=query)
    return f"<s>[INST] {instruction} [/INST]"


def generate(model, tokenizer, query: str, max_tokens: int = 256, temperature: float = 0.1):
    """Generate response for query"""
    prompt = format_prompt(query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only generated part
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


def interactive_mode(model, tokenizer):
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("Interactive Tool Selection Testing")
    print("=" * 60)
    print("Enter queries to test tool selection. Type 'exit' to quit.")
    print("")

    while True:
        query = input("Query: ").strip()

        if query.lower() in ['exit', 'quit', 'q']:
            break

        if not query:
            continue

        print("\nGenerating response...")
        response = generate(model, tokenizer, query)

        print("\n" + "-" * 60)
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print("")


def test_examples(model, tokenizer):
    """Test on predefined examples"""
    examples = [
        ("What is vLLM?", "rag_retrieval"),
        ("What's the latest Python version?", "web_search"),
        ("Calculate fibonacci(10)", "code_execution"),
        ("Explain machine learning", "direct_chat"),
        ("How do I configure the embedding model?", "rag_retrieval"),
        ("Who won the 2024 election?", "web_search"),
        ("Run this: print('hello')", "code_execution"),
        ("Write me a poem", "direct_chat"),
    ]

    print("\n" + "=" * 60)
    print("Testing on Example Queries")
    print("=" * 60)

    correct = 0
    total = len(examples)

    for query, expected_tool in examples:
        print(f"\nQuery: {query}")
        print(f"Expected: {expected_tool}")

        response = generate(model, tokenizer, query)

        # Extract tool from response
        tool = None
        for line in response.split('\n'):
            if line.startswith("Tool:"):
                tool = line.split(":", 1)[1].strip()
                break

        is_correct = tool == expected_tool
        if is_correct:
            correct += 1

        print(f"Predicted: {tool} {'✓' if is_correct else '✗'}")
        print(f"Response: {response[:100]}...")

    print("\n" + "=" * 60)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model inference")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name"
    )
    parser.add_argument(
        "--adapters",
        default="./output/lora_adapters",
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "examples", "single"],
        default="interactive",
        help="Testing mode"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to test (for single mode)"
    )
    parser.add_argument(
        "--no-adapters",
        action="store_true",
        help="Test base model without adapters"
    )

    args = parser.parse_args()

    # Get HF token
    hf_token = os.getenv("HF_TOKEN")

    # Load model
    adapter_path = None if args.no_adapters else args.adapters
    model, tokenizer = load_model(args.base_model, adapter_path, hf_token)

    model_name = "Base Model" if args.no_adapters else "Fine-tuned Model"
    print(f"\n✓ Loaded {model_name}")

    # Run selected mode
    if args.mode == "interactive":
        interactive_mode(model, tokenizer)

    elif args.mode == "examples":
        test_examples(model, tokenizer)

    elif args.mode == "single":
        if not args.query:
            print("Error: --query required for single mode")
            return

        response = generate(model, tokenizer, args.query)
        print("\nQuery:", args.query)
        print("\nResponse:")
        print(response)


if __name__ == "__main__":
    main()
