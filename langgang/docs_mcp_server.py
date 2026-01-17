"""
Documentation MCP Server for LangChain and LangGraph

This module provides MCP server integration for accessing LangChain and LangGraph
documentation to assist with development and code generation.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentationMCPServer:
    """MCP Server for LangChain and LangGraph documentation access."""
    
    def __init__(self):
        """Initialize the documentation MCP server."""
        self.langchain_docs_url = "https://python.langchain.com/docs"
        self.langgraph_docs_url = "https://langchain-ai.github.io/langgraph"
        
        # Common documentation topics
        self.langchain_topics = [
            "installation",
            "quickstart",
            "modules/model_io",
            "modules/retrieval",
            "modules/chains",
            "modules/agents",
            "modules/memory",
            "modules/callbacks",
        ]
        
        self.langgraph_topics = [
            "concepts",
            "tutorials",
            "how-to-guides",
            "reference",
            "cloud",
        ]
    
    def search_langchain_docs(self, query: str) -> Dict[str, Any]:
        """Search LangChain documentation.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        return {
            "source": "langchain",
            "query": query,
            "base_url": self.langchain_docs_url,
            "relevant_topics": [t for t in self.langchain_topics if query.lower() in t.lower()],
            "message": f"Search LangChain docs for: {query}"
        }
    
    def search_langgraph_docs(self, query: str) -> Dict[str, Any]:
        """Search LangGraph documentation.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        return {
            "source": "langgraph",
            "query": query,
            "base_url": self.langgraph_docs_url,
            "relevant_topics": [t for t in self.langgraph_topics if query.lower() in t.lower()],
            "message": f"Search LangGraph docs for: {query}"
        }
    
    def get_langchain_examples(self, category: str = "all") -> Dict[str, List[str]]:
        """Get LangChain code examples.
        
        Args:
            category: Example category (chains, agents, tools, etc.)
            
        Returns:
            Dictionary with example information
        """
        examples = {
            "chains": [
                "Simple LLM Chain",
                "Sequential Chain",
                "Router Chain",
                "Transform Chain",
            ],
            "agents": [
                "Zero-shot ReAct Agent",
                "Conversational Agent",
                "OpenAI Functions Agent",
                "Structured Chat Agent",
            ],
            "tools": [
                "Custom Tool Creation",
                "Tool Calling",
                "Built-in Tools",
            ],
            "memory": [
                "Conversation Buffer Memory",
                "Conversation Summary Memory",
                "Entity Memory",
            ]
        }
        
        if category == "all":
            return {"examples": examples, "base_url": self.langchain_docs_url}
        else:
            return {
                "examples": {category: examples.get(category, [])},
                "base_url": self.langchain_docs_url
            }
    
    def get_langgraph_examples(self, category: str = "all") -> Dict[str, List[str]]:
        """Get LangGraph code examples.
        
        Args:
            category: Example category (graphs, nodes, edges, etc.)
            
        Returns:
            Dictionary with example information
        """
        examples = {
            "graphs": [
                "Basic State Graph",
                "Conditional Edges",
                "Parallel Execution",
                "Subgraphs",
            ],
            "nodes": [
                "Function Nodes",
                "Tool Nodes",
                "LLM Nodes",
            ],
            "state": [
                "State Schema Definition",
                "State Updates",
                "State Annotations",
            ],
            "persistence": [
                "Checkpointing",
                "Memory Persistence",
                "State Recovery",
            ]
        }
        
        if category == "all":
            return {"examples": examples, "base_url": self.langgraph_docs_url}
        else:
            return {
                "examples": {category: examples.get(category, [])},
                "base_url": self.langgraph_docs_url
            }
    
    def list_available_docs(self) -> Dict[str, Dict[str, Any]]:
        """List all available documentation sources.
        
        Returns:
            Dictionary with documentation sources and their topics
        """
        return {
            "langchain": {
                "url": self.langchain_docs_url,
                "topics": self.langchain_topics,
                "description": "LangChain framework for building LLM applications"
            },
            "langgraph": {
                "url": self.langgraph_docs_url,
                "topics": self.langgraph_topics,
                "description": "LangGraph for building stateful multi-actor applications"
            }
        }


def main():
    """Main entry point for documentation MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = DocumentationMCPServer()
    
    logger.info("Documentation MCP Server initialized")
    logger.info(f"Available documentation sources: {server.list_available_docs()}")


if __name__ == "__main__":
    main()
