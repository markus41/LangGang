"""
LangChain integration for LangGang

This module provides LangChain tools and chains for AI-powered template generation.
"""

from typing import Any, Dict, List
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class TemplateGenerationInput(BaseModel):
    """Input schema for template generation."""
    
    template_type: str = Field(description="Type of template: cookiecutter, copier, or maven")
    template_name: str = Field(description="Name of the template to use")
    output_directory: str = Field(description="Directory where code should be generated")
    context: Dict[str, Any] = Field(description="Template variables and configuration")


class TemplateGenerationTool(BaseTool):
    """LangChain tool for template generation."""
    
    name: str = "generate_template"
    description: str = """
    Generate code from a template using Cookiecutter, Copier, or Maven archetypes.
    Use this tool when you need to scaffold a new project or component.
    """
    args_schema: type[BaseModel] = TemplateGenerationInput
    
    def _run(
        self,
        template_type: str,
        template_name: str,
        output_directory: str,
        context: Dict[str, Any]
    ) -> str:
        """Execute template generation.
        
        Args:
            template_type: Type of template (cookiecutter, copier, maven)
            template_name: Name of the template
            output_directory: Directory for generated code
            context: Template variables
            
        Returns:
            Success message or error description
        """
        from langgang.mcp_server import LangGangMCPServer
        
        server = LangGangMCPServer()
        
        if template_type == "cookiecutter":
            result = server.generate_from_cookiecutter(
                template_name, output_directory, context
            )
        elif template_type == "copier":
            result = server.generate_from_copier(
                template_name, output_directory, context
            )
        elif template_type == "maven":
            result = server.generate_from_maven_archetype(
                template_name, output_directory, context
            )
        else:
            return f"Unknown template type: {template_type}"
        
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Successfully generated code at: {result.get('output', output_directory)}"
    
    async def _arun(
        self,
        template_type: str,
        template_name: str,
        output_directory: str,
        context: Dict[str, Any]
    ) -> str:
        """Async execute template generation.
        
        Note: Currently delegates to synchronous version as template
        generation tools don't have async APIs.
        
        Args:
            template_type: Type of template (cookiecutter, copier, maven)
            template_name: Name of the template
            output_directory: Directory for generated code
            context: Template variables
            
        Returns:
            Success message or error description
        """
        return self._run(template_type, template_name, output_directory, context)


class ListTemplatesTool(BaseTool):
    """LangChain tool for listing available templates."""
    
    name: str = "list_templates"
    description: str = """
    List all available code templates (Cookiecutter, Copier, and Maven archetypes).
    Use this tool to discover what templates are available.
    """
    
    def _run(self) -> str:
        """List available templates."""
        from langgang.mcp_server import LangGangMCPServer
        
        server = LangGangMCPServer()
        templates = server.list_templates()
        
        result = "Available Templates:\n\n"
        for template_type, template_list in templates.items():
            result += f"{template_type.capitalize()}:\n"
            if template_list:
                for template in template_list:
                    result += f"  - {template}\n"
            else:
                result += "  (none)\n"
            result += "\n"
        
        return result


def create_template_generation_prompt() -> ChatPromptTemplate:
    """Create a prompt template for AI-assisted code generation.
    
    Returns:
        ChatPromptTemplate for template generation
    """
    template = """You are an expert software architect helping to generate code scaffolding.

The user wants to create: {project_description}

Available tools:
- list_templates: See what templates are available
- generate_template: Generate code from a template

Please analyze the requirements and:
1. Determine which template type is most appropriate (cookiecutter, copier, or maven)
2. Select or recommend a specific template
3. Gather the necessary context variables
4. Generate the code

Be conversational and ask clarifying questions if needed.

User input: {user_input}
"""
    
    return ChatPromptTemplate.from_template(template)


def get_langgang_tools() -> List[BaseTool]:
    """Get all LangGang LangChain tools.
    
    Returns:
        List of LangChain tools for template generation
    """
    return [
        ListTemplatesTool(),
        TemplateGenerationTool(),
    ]
