"""
Agent Tools - Tools for Cipher agent to use

Provides RAG retrieval, web search, direct chat, and code execution capabilities.
"""

import httpx
import logging
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Import code execution module
sys.path.append(str(Path(__file__).parent.parent))
from code_exec import CodeExecutor, ExecutionStatus

logger = logging.getLogger(__name__)


class RAGRetrievalToolInput(BaseModel):
    """Input schema for RAG retrieval tool"""
    query: str = Field(description="The search query for retrieving relevant documents")
    limit: int = Field(default=3, description="Maximum number of documents to retrieve")


class RAGRetrievalTool(BaseTool):
    """
    Tool for retrieving relevant documents from the RAG system

    Uses the RAG pipeline with two-tier caching for fast semantic search.
    """
    name: str = "rag_retrieval"
    description: str = (
        "Searches the knowledge base for relevant information. "
        "Use this when the user asks about specific documents, facts, or information "
        "that might be stored in the system. Returns relevant document chunks with context."
    )
    args_schema: type[BaseModel] = RAGRetrievalToolInput

    rag_pipeline: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, limit: int = 3) -> str:
        """Execute RAG retrieval synchronously"""
        raise NotImplementedError("Use async version")

    async def _arun(self, query: str, limit: int = 3) -> str:
        """Execute RAG retrieval"""
        try:
            if not self.rag_pipeline:
                return "RAG pipeline not available. Cannot retrieve documents."

            # Query the RAG pipeline
            results = await self.rag_pipeline.query(
                query=query,
                limit=limit,
                score_threshold=0.7
            )

            if not results["results"]:
                return f"No relevant documents found for query: {query}"

            # Format results
            formatted_results = []
            for idx, result in enumerate(results["results"], 1):
                chunk_text = result.get("text", "")
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})
                filename = metadata.get("filename", "Unknown")

                formatted_results.append(
                    f"[Document {idx}] (Score: {score:.2f}, Source: {filename})\n{chunk_text}"
                )

            context_info = f"Source: {results['source']}\n\n"
            return context_info + "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return f"Error retrieving documents: {str(e)}"


class WebSearchToolInput(BaseModel):
    """Input schema for web search tool"""
    query: str = Field(description="The search query for web search")
    num_results: int = Field(default=5, description="Number of search results to return")


class WebSearchTool(BaseTool):
    """
    Tool for searching the web using SearXNG

    Provides access to current information from the internet.
    """
    name: str = "web_search"
    description: str = (
        "Searches the internet for current information. "
        "Use this when the user asks about recent events, current data, "
        "or information not available in the knowledge base. "
        "Returns web search results with titles, URLs, and snippets."
    )
    args_schema: type[BaseModel] = WebSearchToolInput

    searxng_endpoint: str = Field(default="http://searxng.search-engine:8080")

    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute web search synchronously"""
        raise NotImplementedError("Use async version")

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Execute web search"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Query SearXNG
                response = await client.get(
                    f"{self.searxng_endpoint}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "categories": "general",
                        "language": "en"
                    }
                )

                if response.status_code != 200:
                    return f"Web search failed with status {response.status_code}"

                data = response.json()
                results = data.get("results", [])[:num_results]

                if not results:
                    return f"No web results found for query: {query}"

                # Format results
                formatted_results = []
                for idx, result in enumerate(results, 1):
                    title = result.get("title", "No title")
                    url = result.get("url", "")
                    content = result.get("content", "No description")

                    formatted_results.append(
                        f"[Result {idx}] {title}\n"
                        f"URL: {url}\n"
                        f"{content}"
                    )

                return "\n\n".join(formatted_results)

        except httpx.TimeoutException:
            logger.error("Web search timeout")
            return "Web search timed out. Please try again."
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error performing web search: {str(e)}"


class DirectChatToolInput(BaseModel):
    """Input schema for direct chat tool"""
    message: str = Field(description="The message to send to the LLM")
    context: Optional[str] = Field(default=None, description="Additional context for the LLM")


class DirectChatTool(BaseTool):
    """
    Tool for direct conversation with the LLM

    Used when no retrieval or search is needed, just pure conversation.
    """
    name: str = "direct_chat"
    description: str = (
        "Directly chat with the AI without using any external tools. "
        "Use this for general conversation, creative tasks, code generation, "
        "or when the user's query doesn't require document retrieval or web search."
    )
    args_schema: type[BaseModel] = DirectChatToolInput

    vllm_endpoint: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")

    def _run(self, message: str, context: Optional[str] = None) -> str:
        """Execute direct chat synchronously"""
        raise NotImplementedError("Use async version")

    async def _arun(self, message: str, context: Optional[str] = None) -> str:
        """Execute direct chat with LLM"""
        try:
            # Build prompt
            if context:
                prompt = f"Context: {context}\n\nUser: {message}\n\nAssistant:"
            else:
                prompt = f"User: {message}\n\nAssistant:"

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.vllm_endpoint}/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "stop": ["User:", "\n\nUser:"]
                    }
                )

                if response.status_code != 200:
                    return f"LLM request failed with status {response.status_code}"

                data = response.json()
                text = data["choices"][0]["text"].strip()
                return text

        except Exception as e:
            logger.error(f"Direct chat error: {e}")
            return f"Error in direct chat: {str(e)}"


class CodeExecutionToolInput(BaseModel):
    """Input schema for code execution tool"""
    code: str = Field(description="Python code to execute")
    description: str = Field(default="Code execution", description="Description of what the code does")


class CodeExecutionTool(BaseTool):
    """
    Tool for executing Python code in a sandboxed environment

    Runs code in isolated Kubernetes Jobs with resource limits and timeout.
    """
    name: str = "code_execution"
    description: str = (
        "Executes Python code in a secure, sandboxed environment. "
        "Use this when the user asks you to run code, perform calculations, "
        "or test Python implementations. The code runs with resource limits "
        "and a 30-second timeout. File system, network, and dangerous operations "
        "are blocked for security. Safe modules like math, random, datetime, "
        "and json are available."
    )
    args_schema: type[BaseModel] = CodeExecutionToolInput

    code_executor: Optional[CodeExecutor] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, code: str, description: str = "Code execution") -> str:
        """Execute code synchronously"""
        raise NotImplementedError("Use async version")

    async def _arun(self, code: str, description: str = "Code execution") -> str:
        """Execute Python code in sandboxed environment"""
        try:
            if not self.code_executor:
                return "Code execution is not available (executor not initialized)."

            logger.info(f"Executing code: {description}")

            # Execute code
            result = await self.code_executor.execute(code, description)

            # Format result based on status
            if result.status == ExecutionStatus.VALIDATION_FAILED:
                return f"❌ Code validation failed:\n{result.error}\n\nPlease fix the issues and try again."

            elif result.status == ExecutionStatus.COMPLETED:
                output = result.output.strip()
                if output:
                    return f"✅ Code executed successfully ({result.execution_time:.2f}s):\n\n{output}"
                else:
                    return f"✅ Code executed successfully ({result.execution_time:.2f}s) with no output."

            elif result.status == ExecutionStatus.TIMEOUT:
                return f"⏱️ Execution timeout after {result.execution_time:.1f}s.\nThe code took too long to run. Try optimizing or simplifying it."

            elif result.status == ExecutionStatus.FAILED:
                error_msg = result.error if result.error else result.output
                return f"❌ Execution failed ({result.execution_time:.2f}s):\n\n{error_msg}"

            else:
                return f"❓ Unknown execution status: {result.status}"

        except Exception as e:
            logger.error(f"Code execution error: {e}", exc_info=True)
            return f"Error executing code: {str(e)}"


def create_agent_tools(
    rag_pipeline: Any = None,
    code_executor: Optional[CodeExecutor] = None,
    vllm_endpoint: str = "http://localhost:8000/v1",
    searxng_endpoint: str = "http://searxng.search-engine:8080",
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
) -> List[BaseTool]:
    """
    Create and configure all agent tools

    Args:
        rag_pipeline: RAG pipeline instance for document retrieval
        code_executor: Code executor instance for running Python code
        vllm_endpoint: vLLM server endpoint
        searxng_endpoint: SearXNG server endpoint
        model: Model name for vLLM

    Returns:
        List of configured tools for the agent
    """
    tools = []

    # RAG retrieval tool (only if RAG is available)
    if rag_pipeline:
        rag_tool = RAGRetrievalTool(rag_pipeline=rag_pipeline)
        tools.append(rag_tool)

    # Web search tool
    web_tool = WebSearchTool(searxng_endpoint=searxng_endpoint)
    tools.append(web_tool)

    # Code execution tool (only if executor is available)
    if code_executor:
        code_tool = CodeExecutionTool(code_executor=code_executor)
        tools.append(code_tool)

    # Direct chat tool
    chat_tool = DirectChatTool(vllm_endpoint=vllm_endpoint, model=model)
    tools.append(chat_tool)

    return tools
