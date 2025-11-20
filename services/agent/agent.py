"""
Cipher Agent - LangGraph-based Multi-Tool Agent

Intelligent agent that routes queries to appropriate tools:
- Code execution for running Python code
- RAG retrieval for knowledge base queries
- Web search for current information
- Direct chat for general conversation
"""

import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Sequence
from datetime import datetime
import operator
import httpx

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .tools import create_agent_tools, RAGRetrievalTool, WebSearchTool, DirectChatTool, CodeExecutionTool

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State of the agent during execution

    Tracks messages, tool calls, and execution metadata.
    """
    # Messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Current user query
    query: str

    # Tool outputs
    tool_outputs: Dict[str, Any]

    # Selected tool
    selected_tool: Optional[str]

    # Final response
    response: Optional[str]

    # Metadata
    metadata: Dict[str, Any]


class CipherAgent:
    """
    Cipher - Multi-tool AI Agent using LangGraph

    Routes queries intelligently between code execution, RAG retrieval, web search, and direct chat.
    """

    def __init__(
        self,
        rag_pipeline: Any = None,
        code_executor: Any = None,
        vllm_endpoint: str = "http://192.168.1.100:8000/v1",
        searxng_endpoint: str = "http://searxng.search-engine:8080",
        model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ):
        """
        Initialize Cipher agent

        Args:
            rag_pipeline: RAG pipeline for document retrieval
            code_executor: Code executor for running Python code
            vllm_endpoint: vLLM server endpoint
            searxng_endpoint: SearXNG server endpoint
            model: Model name for vLLM
        """
        self.rag_pipeline = rag_pipeline
        self.code_executor = code_executor
        self.vllm_endpoint = vllm_endpoint
        self.searxng_endpoint = searxng_endpoint
        self.model = model

        # Create tools
        self.tools = create_agent_tools(
            rag_pipeline=rag_pipeline,
            code_executor=code_executor,
            vllm_endpoint=vllm_endpoint,
            searxng_endpoint=searxng_endpoint,
            model=model
        )

        # Map tool names to tools
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Build the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("code_execution", self._code_execution_node)
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("direct_chat", self._direct_chat_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_query,
            {
                "code_execution": "code_execution",
                "rag_retrieval": "rag_retrieval",
                "web_search": "web_search",
                "direct_chat": "direct_chat",
            }
        )

        # All tool nodes go to synthesize
        workflow.add_edge("code_execution", "synthesize")
        workflow.add_edge("rag_retrieval", "synthesize")
        workflow.add_edge("web_search", "synthesize")
        workflow.add_edge("direct_chat", "synthesize")

        # Synthesize ends the workflow
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        """
        Router node - analyzes query and decides which tool to use

        Uses simple keyword matching and heuristics for routing.
        """
        query = state["query"].lower()

        # Initialize metadata
        state["metadata"] = {
            "start_time": datetime.utcnow().isoformat(),
            "routing_decision": None
        }

        # Routing heuristics
        # 1. Code execution keywords
        code_keywords = [
            "run", "execute", "execute code", "calculate", "test",
            "implement", "write code", "python code", "run python",
            "compute", "run this", "execute this"
        ]

        # 2. Web search keywords
        web_keywords = [
            "latest", "recent", "current", "today", "news", "weather",
            "what's new", "updates", "happening now", "breaking",
            "search the web", "search for", "look up online"
        ]

        # 3. RAG retrieval keywords
        rag_keywords = [
            "document", "knowledge base", "what do you know about",
            "in our docs", "according to", "stored information",
            "find in", "search docs", "retrieve"
        ]

        # Determine tool
        if any(keyword in query for keyword in code_keywords) and self.code_executor:
            selected_tool = "code_execution"
        elif any(keyword in query for keyword in web_keywords):
            selected_tool = "web_search"
        elif any(keyword in query for keyword in rag_keywords) and self.rag_pipeline:
            selected_tool = "rag_retrieval"
        elif self.rag_pipeline:
            # Default to RAG if available (most queries benefit from context)
            selected_tool = "rag_retrieval"
        else:
            # Fall back to direct chat
            selected_tool = "direct_chat"

        state["selected_tool"] = selected_tool
        state["metadata"]["routing_decision"] = selected_tool

        logger.info(f"Router selected tool: {selected_tool} for query: {query[:100]}")

        return state

    def _route_query(self, state: AgentState) -> str:
        """Return the selected tool for conditional routing"""
        return state["selected_tool"]

    async def _code_execution_node(self, state: AgentState) -> AgentState:
        """Execute Python code"""
        try:
            logger.info("Executing Python code...")
            tool: CodeExecutionTool = self.tool_map["code_execution"]
            result = await tool._arun(code=state["query"])

            state["tool_outputs"]["code_execution"] = result
            logger.info(f"Code execution completed: {len(result)} chars")

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            state["tool_outputs"]["code_execution"] = f"Error: {str(e)}"

        return state

    async def _rag_retrieval_node(self, state: AgentState) -> AgentState:
        """Execute RAG retrieval"""
        try:
            logger.info("Executing RAG retrieval...")
            tool: RAGRetrievalTool = self.tool_map["rag_retrieval"]
            result = await tool._arun(query=state["query"], limit=3)

            state["tool_outputs"]["rag_retrieval"] = result
            logger.info(f"RAG retrieval completed: {len(result)} chars")

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            state["tool_outputs"]["rag_retrieval"] = f"Error: {str(e)}"

        return state

    async def _web_search_node(self, state: AgentState) -> AgentState:
        """Execute web search"""
        try:
            logger.info("Executing web search...")
            tool: WebSearchTool = self.tool_map["web_search"]
            result = await tool._arun(query=state["query"], num_results=5)

            state["tool_outputs"]["web_search"] = result
            logger.info(f"Web search completed: {len(result)} chars")

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            state["tool_outputs"]["web_search"] = f"Error: {str(e)}"

        return state

    async def _direct_chat_node(self, state: AgentState) -> AgentState:
        """Execute direct chat"""
        try:
            logger.info("Executing direct chat...")
            tool: DirectChatTool = self.tool_map["direct_chat"]

            # Check if we have any context from previous tool outputs
            context = None
            if state.get("tool_outputs"):
                context_parts = []
                for tool_name, output in state["tool_outputs"].items():
                    context_parts.append(f"{tool_name}: {output}")
                context = "\n\n".join(context_parts)

            result = await tool._arun(message=state["query"], context=context)

            state["tool_outputs"]["direct_chat"] = result
            logger.info(f"Direct chat completed: {len(result)} chars")

        except Exception as e:
            logger.error(f"Direct chat failed: {e}")
            state["tool_outputs"]["direct_chat"] = f"Error: {str(e)}"

        return state

    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """
        Synthesize final response from tool outputs

        Combines tool outputs with LLM to create coherent response.
        """
        try:
            logger.info("Synthesizing response...")

            # Get the primary tool output
            selected_tool = state["selected_tool"]
            tool_output = state["tool_outputs"].get(selected_tool, "No output available")

            # For RAG, web search, and code execution, synthesize with LLM
            # For direct chat, use output directly
            if selected_tool in ["rag_retrieval", "web_search", "code_execution"]:
                # Create synthesis prompt
                prompt = self._create_synthesis_prompt(
                    query=state["query"],
                    tool_name=selected_tool,
                    tool_output=tool_output
                )

                # Call LLM to synthesize
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.vllm_endpoint}/completions",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "max_tokens": 1024,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "stop": ["User:", "\n\nUser:", "Context:"]
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        final_response = data["choices"][0]["text"].strip()
                    else:
                        final_response = tool_output  # Fallback to raw output
            else:
                # Direct chat output is already synthesized
                final_response = tool_output

            state["response"] = final_response
            state["metadata"]["end_time"] = datetime.utcnow().isoformat()

            logger.info("Response synthesis completed")

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to tool output
            state["response"] = state["tool_outputs"].get(
                state["selected_tool"],
                "I encountered an error processing your request."
            )

        return state

    def _create_synthesis_prompt(self, query: str, tool_name: str, tool_output: str) -> str:
        """Create prompt for synthesizing final response"""
        if tool_name == "code_execution":
            return f"""You are Cipher, an AI assistant. The user asked you to execute Python code. Here are the results.

User Request: {query}

Execution Results:
{tool_output}

Provide a clear explanation of what was executed and the results. If there were errors, explain them clearly.

Answer: """
        elif tool_name == "rag_retrieval":
            return f"""You are Cipher, an AI assistant. Use the following retrieved documents to answer the user's question.

Retrieved Documents:
{tool_output}

User Question: {query}

Provide a clear, accurate answer based on the documents above. If the documents don't contain relevant information, say so.

Answer: """
        elif tool_name == "web_search":
            return f"""You are Cipher, an AI assistant. Use the following web search results to answer the user's question.

Web Search Results:
{tool_output}

User Question: {query}

Provide a clear, accurate answer based on the search results above. Cite sources where appropriate. If the results don't contain relevant information, say so.

Answer: """
        else:
            # Fallback for direct chat or unknown tools
            return f"""You are Cipher, an AI assistant. Answer the following question directly and helpfully.

User Question: {query}

Context (if available):
{tool_output}

Answer: """

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent with a given query

        Args:
            query: User query to process

        Returns:
            Dictionary containing:
                - response: The final synthesized response
                - metadata: Execution metadata (timing, routing, etc.)
                - tool_outputs: Raw outputs from tools
                - selected_tool: Which tool was used
        """
        logger.info(f"Agent run started for query: {query[:100]}...")

        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "tool_outputs": {},
            "selected_tool": None,
            "response": None,
            "metadata": {}
        }

        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            # Extract results
            result = {
                "response": final_state["response"],
                "metadata": final_state["metadata"],
                "tool_outputs": final_state["tool_outputs"],
                "selected_tool": final_state["selected_tool"]
            }

            logger.info(f"Agent run completed successfully using {result['selected_tool']}")
            return result

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "metadata": {
                    "error": str(e),
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": datetime.utcnow().isoformat()
                },
                "tool_outputs": {},
                "selected_tool": None
            }