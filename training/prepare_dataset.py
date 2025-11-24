"""
Dataset Preparation for Tool-Aware Fine-tuning

Converts query logs and synthetic examples into training data for Mistral 7B Instruct.
Formats data for tool-calling fine-tuning with proper instruction format.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")
    psycopg2 = None


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent"""
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class TrainingExample:
    """Single training example for tool-aware fine-tuning"""
    instruction: str  # User query
    available_tools: List[ToolDefinition]  # Tools available
    response: str  # Expected tool call and reasoning
    metadata: Dict[str, Any]  # Source, timestamp, etc.


# Tool definitions matching your agent
TOOL_DEFINITIONS = [
    ToolDefinition(
        name="rag_retrieval",
        description="Searches the knowledge base for relevant information. Use when the user asks about specific documents, facts, or information stored in the system.",
        parameters={
            "query": {"type": "string", "description": "The search query"},
            "limit": {"type": "integer", "description": "Max documents (default: 3)", "default": 3}
        }
    ),
    ToolDefinition(
        name="web_search",
        description="Searches the internet for current information. Use when user asks about recent events, current data, or info not in the knowledge base.",
        parameters={
            "query": {"type": "string", "description": "The search query"},
            "num_results": {"type": "integer", "description": "Number of results (default: 5)", "default": 5}
        }
    ),
    ToolDefinition(
        name="code_execution",
        description="Executes Python code in a secure sandbox. Use when user asks to run code, perform calculations, or test implementations. 30s timeout, safe modules only.",
        parameters={
            "code": {"type": "string", "description": "Python code to execute"},
            "description": {"type": "string", "description": "What the code does", "default": "Code execution"}
        }
    ),
    ToolDefinition(
        name="direct_chat",
        description="Direct conversation with AI without external tools. Use for general chat, creative tasks, code generation when no retrieval/search needed.",
        parameters={
            "message": {"type": "string", "description": "The message"},
            "context": {"type": "string", "description": "Additional context", "optional": True}
        }
    )
]


def format_tool_for_prompt(tool: ToolDefinition) -> str:
    """Format a tool definition for the training prompt"""
    params_str = json.dumps(tool.parameters, indent=2)
    return f"""Tool: {tool.name}
Description: {tool.description}
Parameters: {params_str}"""


def format_training_example_mistral(example: TrainingExample) -> str:
    """
    Format training example in Mistral instruction format

    Uses Mistral's instruction template:
    <s>[INST] {instruction} [/INST] {response}</s>
    """
    # Build tools section
    tools_section = "\n\n".join([format_tool_for_prompt(t) for t in example.available_tools])

    # Full instruction with tools
    full_instruction = f"""You are an AI agent with access to the following tools:

{tools_section}

User Query: {example.instruction}

Analyze the query and determine which tool to use. Respond with:
1. Tool name
2. Parameters (as JSON)
3. Brief reasoning

Format: Tool: <name>
Parameters: <json>
Reasoning: <explanation>"""

    # Return in Mistral format (without tokens, model adds during training)
    return {
        "instruction": full_instruction,
        "response": example.response,
        "metadata": example.metadata
    }


def connect_to_db() -> Any:
    """Connect to PostgreSQL database"""
    if not psycopg2:
        raise ImportError("psycopg2 not installed")

    # Get DB credentials from environment
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "cipher_analytics"),
        "user": os.getenv("POSTGRES_USER", "cipher"),
        "password": os.getenv("POSTGRES_PASSWORD", "")
    }

    return psycopg2.connect(**db_config, cursor_factory=RealDictCursor)


def extract_from_query_logs(limit: int = 1000) -> List[TrainingExample]:
    """
    Extract training examples from query logs

    Args:
        limit: Maximum number of examples to extract

    Returns:
        List of training examples from successful queries
    """
    if not psycopg2:
        print("Warning: Cannot connect to database. Skipping query log extraction.")
        return []

    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # Query successful logs with tool usage
        cursor.execute("""
            SELECT
                query_text,
                tool_used,
                execution_time,
                result_quality,
                timestamp,
                cache_hit
            FROM query_logs
            WHERE success = TRUE
                AND tool_used IS NOT NULL
                AND query_text IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit,))

        rows = cursor.fetchall()
        examples = []

        for row in rows:
            query = row['query_text']
            tool = row['tool_used']

            # Skip if tool not recognized
            if tool not in [t.name for t in TOOL_DEFINITIONS]:
                continue

            # Create response based on tool used
            response = create_response_for_tool(tool, query)

            example = TrainingExample(
                instruction=query,
                available_tools=TOOL_DEFINITIONS,
                response=response,
                metadata={
                    "source": "query_logs",
                    "tool": tool,
                    "execution_time": row['execution_time'],
                    "quality": row['result_quality'],
                    "timestamp": row['timestamp'].isoformat() if row['timestamp'] else None,
                    "cache_hit": row['cache_hit']
                }
            )

            examples.append(example)

        cursor.close()
        conn.close()

        print(f"✓ Extracted {len(examples)} examples from query logs")
        return examples

    except Exception as e:
        print(f"Warning: Failed to extract from database: {e}")
        return []


def create_response_for_tool(tool_name: str, query: str) -> str:
    """
    Create expected response for a given tool and query

    Args:
        tool_name: Name of the tool that should be used
        query: User query

    Returns:
        Formatted response showing tool selection and reasoning
    """
    # Map tool names to response templates
    responses = {
        "rag_retrieval": f"""Tool: rag_retrieval
Parameters: {{"query": "{query}", "limit": 3}}
Reasoning: This query asks about information that would be stored in the knowledge base. RAG retrieval is the best choice to search for relevant documents.""",

        "web_search": f"""Tool: web_search
Parameters: {{"query": "{query}", "num_results": 5}}
Reasoning: This query requires current, up-to-date information from the internet. Web search is needed to get the latest data.""",

        "code_execution": f"""Tool: code_execution
Parameters: {{"code": "# Code extracted from query", "description": "Execute user's code"}}
Reasoning: This query asks to run or test code. Code execution tool will safely execute the Python code in a sandbox.""",

        "direct_chat": f"""Tool: direct_chat
Parameters: {{"message": "{query}"}}
Reasoning: This is a general conversation query that doesn't require document retrieval, web search, or code execution. Direct chat is most appropriate."""
    }

    return responses.get(tool_name, f"Tool: {tool_name}\nParameters: {{}}\nReasoning: Selected based on query context.")


def generate_synthetic_examples() -> List[TrainingExample]:
    """
    Generate synthetic training examples for all tools

    Creates diverse examples covering different use cases for each tool.
    """
    synthetic = []

    # RAG retrieval examples
    rag_queries = [
        ("What is vLLM and how does it work?", "This asks about vLLM documentation which should be in our knowledge base."),
        ("How do I configure the embedding model?", "This is a configuration question likely covered in our docs."),
        ("What are the caching strategies available?", "This asks about system features documented in our knowledge base."),
        ("Tell me about the RAG pipeline implementation", "This asks about technical implementation details in our codebase."),
        ("How does the code execution sandbox work?", "This queries internal system documentation about security features."),
        ("What database schema is used for analytics?", "This asks about system architecture documented internally."),
        ("Show me information about Kubernetes deployment", "This requests infrastructure documentation."),
        ("What models are supported by the system?", "This asks about system capabilities documented in our knowledge base."),
    ]

    for query, reasoning in rag_queries:
        synthetic.append(TrainingExample(
            instruction=query,
            available_tools=TOOL_DEFINITIONS,
            response=f"""Tool: rag_retrieval
Parameters: {{"query": "{query}", "limit": 3}}
Reasoning: {reasoning}""",
            metadata={"source": "synthetic", "category": "rag", "tool": "rag_retrieval"}
        ))

    # Web search examples
    web_queries = [
        ("What's the latest version of Python?", "This asks for current information that changes regularly."),
        ("What are today's top tech news?", "This requires recent, up-to-date information from the web."),
        ("Who won the latest AI competition?", "This is a current events query needing fresh data."),
        ("What's the current best practice for transformer models?", "This asks about evolving best practices in AI."),
        ("What are the recent updates to PyTorch?", "This needs current information about software releases."),
        ("Find recent papers on RAG optimization", "This requires searching for recent research publications."),
        ("What companies are hiring AI engineers now?", "This asks for current job market information."),
        ("What's trending in machine learning this week?", "This needs very recent trend information."),
    ]

    for query, reasoning in web_queries:
        synthetic.append(TrainingExample(
            instruction=query,
            available_tools=TOOL_DEFINITIONS,
            response=f"""Tool: web_search
Parameters: {{"query": "{query}", "num_results": 5}}
Reasoning: {reasoning}""",
            metadata={"source": "synthetic", "category": "web_search", "tool": "web_search"}
        ))

    # Code execution examples
    code_queries = [
        ("Calculate the fibonacci sequence up to 10", "This asks to perform a calculation that requires code execution."),
        ("Run this code: print('Hello World')", "This explicitly asks to execute Python code."),
        ("Test if 17 is prime", "This requires computational testing best done with code."),
        ("What's 2^128?", "This is a calculation that needs code to compute accurately."),
        ("Sort this list: [5,2,8,1,9]", "This asks to perform a data operation requiring code execution."),
        ("Generate random numbers between 1 and 100", "This needs code to generate random values."),
        ("What's the sum of squares from 1 to 50?", "This is a mathematical calculation requiring code."),
        ("Parse this JSON and extract the name field", "This requires code to process structured data."),
    ]

    for query, reasoning in code_queries:
        synthetic.append(TrainingExample(
            instruction=query,
            available_tools=TOOL_DEFINITIONS,
            response=f"""Tool: code_execution
Parameters: {{"code": "# Code to handle the query", "description": "Execute calculation"}}
Reasoning: {reasoning}""",
            metadata={"source": "synthetic", "category": "code_execution", "tool": "code_execution"}
        ))

    # Direct chat examples
    chat_queries = [
        ("Explain what machine learning is", "This is a general knowledge explanation that doesn't need external tools."),
        ("Write me a poem about AI", "This is a creative writing task that direct chat handles well."),
        ("How would you implement a binary search?", "This asks for code explanation/generation without needing execution."),
        ("What's the difference between supervised and unsupervised learning?", "This is a conceptual question that LLM can answer directly."),
        ("Help me understand recursion with an example", "This is a teaching task that doesn't require external resources."),
        ("Translate this to French: Hello, how are you?", "This is a language task the LLM can handle directly."),
        ("Give me tips for writing clean code", "This asks for general advice that doesn't need retrieval."),
        ("What's your opinion on functional programming?", "This asks for analysis and discussion, not facts."),
    ]

    for query, reasoning in chat_queries:
        synthetic.append(TrainingExample(
            instruction=query,
            available_tools=TOOL_DEFINITIONS,
            response=f"""Tool: direct_chat
Parameters: {{"message": "{query}"}}
Reasoning: {reasoning}""",
            metadata={"source": "synthetic", "category": "direct_chat", "tool": "direct_chat"}
        ))

    print(f"✓ Generated {len(synthetic)} synthetic examples")
    return synthetic


def create_ambiguous_examples() -> List[TrainingExample]:
    """
    Create examples that require careful reasoning about tool selection

    These are harder examples where multiple tools might seem applicable.
    """
    ambiguous = []

    # Examples that might confuse between RAG and web search
    ambiguous.append(TrainingExample(
        instruction="What is the system architecture of this project?",
        available_tools=TOOL_DEFINITIONS,
        response="""Tool: rag_retrieval
Parameters: {"query": "system architecture", "limit": 3}
Reasoning: While this could be searched on the web, the query asks specifically about 'this project', indicating internal documentation. RAG retrieval is more appropriate.""",
        metadata={"source": "synthetic", "category": "ambiguous", "tool": "rag_retrieval"}
    ))

    ambiguous.append(TrainingExample(
        instruction="What are best practices for RAG systems?",
        available_tools=TOOL_DEFINITIONS,
        response="""Tool: web_search
Parameters: {"query": "best practices for RAG systems", "num_results": 5}
Reasoning: This asks about general best practices in the field, not specific to this system. Web search will provide broader, more current industry knowledge.""",
        metadata={"source": "synthetic", "category": "ambiguous", "tool": "web_search"}
    ))

    # Examples between code execution and direct chat
    ambiguous.append(TrainingExample(
        instruction="How do I reverse a list in Python?",
        available_tools=TOOL_DEFINITIONS,
        response="""Tool: direct_chat
Parameters: {"message": "How do I reverse a list in Python?"}
Reasoning: This asks 'how to' rather than 'do it'. The user wants an explanation, not execution. Direct chat can explain the methods.""",
        metadata={"source": "synthetic", "category": "ambiguous", "tool": "direct_chat"}
    ))

    ambiguous.append(TrainingExample(
        instruction="Reverse this list for me: [1, 2, 3, 4, 5]",
        available_tools=TOOL_DEFINITIONS,
        response="""Tool: code_execution
Parameters: {"code": "lst = [1, 2, 3, 4, 5]\\nprint(lst[::-1])", "description": "Reverse list"}
Reasoning: The user asks to actually reverse a specific list ('for me'), not explain how. Code execution is needed to perform the task.""",
        metadata={"source": "synthetic", "category": "ambiguous", "tool": "code_execution"}
    ))

    print(f"✓ Generated {len(ambiguous)} ambiguous examples")
    return ambiguous


def split_dataset(examples: List[TrainingExample],
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Split dataset into train/validation/test sets

    Args:
        examples: All training examples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train, val, test) example lists
    """
    # Shuffle examples
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    return train, val, test


def save_dataset(examples: List[TrainingExample], output_path: str, format_type: str = "mistral"):
    """
    Save dataset to JSON file

    Args:
        examples: Training examples to save
        output_path: Path to output JSON file
        format_type: Format type ('mistral' or 'raw')
    """
    if format_type == "mistral":
        formatted = [format_training_example_mistral(ex) for ex in examples]
    else:
        formatted = [asdict(ex) for ex in examples]

    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)

    print(f"✓ Saved {len(examples)} examples to {output_path}")


def main():
    """Main dataset preparation pipeline"""
    print("=" * 60)
    print("Tool-Aware Fine-tuning Dataset Preparation")
    print("=" * 60)

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("\n[1/5] Extracting examples from query logs...")
    query_examples = extract_from_query_logs(limit=1000)

    print("\n[2/5] Generating synthetic examples...")
    synthetic_examples = generate_synthetic_examples()

    print("\n[3/5] Creating ambiguous examples...")
    ambiguous_examples = create_ambiguous_examples()

    # Combine all examples
    all_examples = query_examples + synthetic_examples + ambiguous_examples
    print(f"\n[4/5] Total examples collected: {len(all_examples)}")

    # Distribution by tool
    tool_dist = {}
    for ex in all_examples:
        tool = ex.metadata.get('tool', 'unknown')
        tool_dist[tool] = tool_dist.get(tool, 0) + 1

    print("\nDistribution by tool:")
    for tool, count in sorted(tool_dist.items()):
        print(f"  {tool}: {count}")

    print("\n[5/5] Splitting and saving dataset...")
    train, val, test = split_dataset(all_examples)

    print(f"  Train: {len(train)} examples")
    print(f"  Val:   {len(val)} examples")
    print(f"  Test:  {len(test)} examples")

    # Save in Mistral format
    save_dataset(train, output_dir / "train.json", format_type="mistral")
    save_dataset(val, output_dir / "val.json", format_type="mistral")
    save_dataset(test, output_dir / "test.json", format_type="mistral")

    # Also save raw format for inspection
    save_dataset(all_examples[:10], output_dir / "sample_raw.json", format_type="raw")

    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nDataset saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review sample_raw.json to verify formatting")
    print("  2. Run training: python train_tool_aware.py")
    print("  3. Evaluate results: python evaluate.py")


if __name__ == "__main__":
    main()
