"""
LangGang MCP Server

This module implements a Model Context Protocol (MCP) server for AI-powered
code templating using LangChain, LangGraph, Cookiecutter, Copier, and Maven archetypes.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _validate_path(path: str, param_name: str) -> None:
    """Validate path to prevent directory traversal attacks.
    
    Args:
        path: Path to validate
        param_name: Parameter name for error messages
        
    Raises:
        ValueError: If path contains directory traversal sequences
    """
    if ".." in path or path.startswith("/"):
        raise ValueError(f"{param_name} must not contain '..' or start with '/'")


def _sanitize_property_value(value: str) -> str:
    """Sanitize property values for Maven commands.
    
    Args:
        value: Property value to sanitize
        
    Returns:
        Sanitized value
        
    Raises:
        ValueError: If value contains shell metacharacters
    """
    # Only allow alphanumeric, dash, underscore, dot, and forward slash
    if not re.match(r'^[a-zA-Z0-9._/-]+$', value):
        raise ValueError(f"Property value contains invalid characters: {value}")
    return value


class LangGangMCPServer:
    """MCP Server for LangGang template generation."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the MCP server.
        
        Args:
            templates_dir: Directory containing templates. Defaults to ./templates
        """
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "templates"
        self.cookiecutter_dir = self.templates_dir / "cookiecutter"
        self.copier_dir = self.templates_dir / "copier"
        self.maven_dir = self.templates_dir / "maven"
        
    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates.
        
        Returns:
            Dictionary mapping template types to lists of template names
        """
        templates = {
            "cookiecutter": [],
            "copier": [],
            "maven": []
        }
        
        if self.cookiecutter_dir.exists():
            templates["cookiecutter"] = [
                d.name for d in self.cookiecutter_dir.iterdir() if d.is_dir()
            ]
            
        if self.copier_dir.exists():
            templates["copier"] = [
                d.name for d in self.copier_dir.iterdir() if d.is_dir()
            ]
            
        if self.maven_dir.exists():
            templates["maven"] = [
                d.name for d in self.maven_dir.iterdir() if d.is_dir()
            ]
            
        return templates
    
    def generate_from_cookiecutter(
        self,
        template_name: str,
        output_dir: str,
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate code from a Cookiecutter template.
        
        Args:
            template_name: Name of the template
            output_dir: Output directory for generated code
            context: Template variables
            
        Returns:
            Dictionary with generation results
        """
        from cookiecutter.main import cookiecutter
        
        # Validate inputs to prevent path traversal
        try:
            _validate_path(template_name, "template_name")
        except ValueError as e:
            return {"error": str(e)}
        
        template_path = self.cookiecutter_dir / template_name
        if not template_path.exists():
            return {"error": f"Template not found: {template_name}"}
        
        # Ensure template_path is within cookiecutter_dir
        try:
            template_path.resolve().relative_to(self.cookiecutter_dir.resolve())
        except ValueError:
            return {"error": "Invalid template path"}
        
        try:
            result = cookiecutter(
                str(template_path),
                output_dir=output_dir,
                no_input=True,
                extra_context=context
            )
            return {"status": "success", "output": result}
        except Exception as e:
            logger.error(f"Error generating from Cookiecutter: {e}")
            return {"error": str(e)}
    
    def generate_from_copier(
        self,
        template_name: str,
        output_dir: str,
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate code from a Copier template.
        
        Args:
            template_name: Name of the template
            output_dir: Output directory for generated code
            context: Template variables
            
        Returns:
            Dictionary with generation results
        """
        from copier import run_copy
        
        # Validate inputs to prevent path traversal
        try:
            _validate_path(template_name, "template_name")
        except ValueError as e:
            return {"error": str(e)}
        
        template_path = self.copier_dir / template_name
        if not template_path.exists():
            return {"error": f"Template not found: {template_name}"}
        
        # Ensure template_path is within copier_dir
        try:
            template_path.resolve().relative_to(self.copier_dir.resolve())
        except ValueError:
            return {"error": "Invalid template path"}
        
        try:
            run_copy(
                str(template_path),
                output_dir,
                data=context,
                defaults=True,
                overwrite=True
            )
            return {"status": "success", "output": output_dir}
        except Exception as e:
            logger.error(f"Error generating from Copier: {e}")
            return {"error": str(e)}
    
    def generate_from_maven_archetype(
        self,
        archetype_name: str,
        output_dir: str,
        properties: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate code from a Maven archetype.
        
        Args:
            archetype_name: Name of the archetype
            output_dir: Output directory for generated code
            properties: Maven properties (groupId, artifactId, etc.)
            
        Returns:
            Dictionary with generation results
        """
        import subprocess
        
        # Validate inputs to prevent path traversal
        try:
            _validate_path(archetype_name, "archetype_name")
        except ValueError as e:
            return {"error": str(e)}
        
        archetype_path = self.maven_dir / archetype_name
        if not archetype_path.exists():
            return {"error": f"Archetype not found: {archetype_name}"}
        
        # Ensure archetype_path is within maven_dir
        try:
            archetype_path.resolve().relative_to(self.maven_dir.resolve())
        except ValueError:
            return {"error": "Invalid archetype path"}
        
        # Build Maven command with sanitized properties
        cmd = ["mvn", "archetype:generate"]
        cmd.extend(["-DarchetypeGroupId=com.langgang"])
        
        # Sanitize archetype_name for command
        try:
            sanitized_archetype = _sanitize_property_value(archetype_name)
            cmd.extend([f"-DarchetypeArtifactId={sanitized_archetype}"])
        except ValueError as e:
            return {"error": str(e)}
        
        # Sanitize all property values to prevent command injection
        try:
            for key, value in properties.items():
                # Validate key name
                if not re.match(r'^[a-zA-Z0-9._-]+$', key):
                    return {"error": f"Invalid property key: {key}"}
                sanitized_value = _sanitize_property_value(value)
                cmd.append(f"-D{key}={sanitized_value}")
        except ValueError as e:
            return {"error": str(e)}
        
        cmd.append("-DinteractiveMode=false")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {"status": "success", "output": result.stdout}
            else:
                return {"error": result.stderr}
        except Exception as e:
            logger.error(f"Error generating from Maven archetype: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = LangGangMCPServer()
    
    # In a real implementation, this would start the MCP server
    # For now, we'll just log that it's ready
    logger.info("LangGang MCP Server initialized")
    logger.info(f"Available templates: {server.list_templates()}")


if __name__ == "__main__":
    main()
