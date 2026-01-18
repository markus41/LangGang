"""
LangGang MCP Server

This module implements a Model Context Protocol (MCP) server for AI-powered
code templating using LangChain, LangGraph, Cookiecutter, Copier, and Maven archetypes.

Features:
- Multi-template engine support (Cookiecutter, Copier, Maven)
- Rich template metadata and discovery
- Security-first path validation
- Intelligent template search and filtering
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Security Helpers
# ============================================================================

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


# ============================================================================
# Template Metadata Schema
# ============================================================================

class TemplateMetadata:
    """Rich metadata for a template.

    This class represents the standardized metadata schema for all templates,
    enabling intelligent discovery, filtering, and recommendations.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        name: str,
        display_name: str = "",
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
        category: str = "general",
        difficulty: str = "intermediate",
        frameworks: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        compatible_with: Optional[List[str]] = None,
        template_type: str = "cookiecutter",
    ):
        self.schema_version = self.SCHEMA_VERSION
        self.name = name
        self.display_name = display_name or name.replace("-", " ").replace("_", " ").title()
        self.description = description
        self.version = version
        self.author = author
        self.tags = tags or []
        self.category = category
        self.difficulty = difficulty
        self.frameworks = frameworks or []
        self.languages = languages or []
        self.prerequisites = prerequisites or []
        self.features = features or []
        self.compatible_with = compatible_with or []
        self.template_type = template_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "category": self.category,
            "difficulty": self.difficulty,
            "frameworks": self.frameworks,
            "languages": self.languages,
            "prerequisites": self.prerequisites,
            "features": self.features,
            "compatible_with": self.compatible_with,
            "template_type": self.template_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            category=data.get("category", "general"),
            difficulty=data.get("difficulty", "intermediate"),
            frameworks=data.get("frameworks", []),
            languages=data.get("languages", []),
            prerequisites=data.get("prerequisites", []),
            features=data.get("features", []),
            compatible_with=data.get("compatible_with", []),
            template_type=data.get("template_type", "cookiecutter"),
        )

    def matches_query(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        frameworks: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ) -> float:
        """Calculate relevance score for search criteria.

        Args:
            query: Free-text search query
            tags: Tags to match
            category: Category to match
            frameworks: Frameworks to match
            languages: Languages to match
            difficulty: Difficulty level to match

        Returns:
            Relevance score from 0.0 to 1.0
        """
        score = 0.0
        max_score = 0.0

        # Text query matching
        if query:
            max_score += 3.0
            query_lower = query.lower()

            # Name match (highest weight)
            if query_lower in self.name.lower():
                score += 1.5
            elif any(word in self.name.lower() for word in query_lower.split()):
                score += 0.75

            # Description match
            if query_lower in self.description.lower():
                score += 1.0
            elif any(word in self.description.lower() for word in query_lower.split()):
                score += 0.5

            # Tag match
            if any(query_lower in tag.lower() for tag in self.tags):
                score += 0.5

        # Tag matching
        if tags:
            max_score += 1.0
            matching_tags = len(set(t.lower() for t in tags) & set(t.lower() for t in self.tags))
            if matching_tags > 0:
                score += min(1.0, matching_tags / len(tags))

        # Category matching
        if category:
            max_score += 1.0
            if category.lower() == self.category.lower():
                score += 1.0

        # Framework matching
        if frameworks:
            max_score += 1.0
            matching_fw = len(
                set(f.lower() for f in frameworks) & set(f.lower() for f in self.frameworks)
            )
            if matching_fw > 0:
                score += min(1.0, matching_fw / len(frameworks))

        # Language matching
        if languages:
            max_score += 1.0
            matching_lang = len(
                set(l.lower() for l in languages) & set(l.lower() for l in self.languages)
            )
            if matching_lang > 0:
                score += min(1.0, matching_lang / len(languages))

        # Difficulty matching
        if difficulty:
            max_score += 0.5
            if difficulty.lower() == self.difficulty.lower():
                score += 0.5

        # Normalize score
        if max_score > 0:
            return score / max_score
        return 0.0


# ============================================================================
# MCP Server
# ============================================================================

class LangGangMCPServer:
    """MCP Server for LangGang template generation.

    Provides:
    - Template listing and discovery
    - Rich metadata support
    - Multi-engine code generation (Cookiecutter, Copier, Maven)
    - Template search and filtering
    - Variable extraction for templates
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the MCP server.

        Args:
            templates_dir: Directory containing templates. Defaults to ./templates
        """
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "templates"
        self.cookiecutter_dir = self.templates_dir / "cookiecutter"
        self.copier_dir = self.templates_dir / "copier"
        self.maven_dir = self.templates_dir / "maven"

        # Cache for template metadata
        self._metadata_cache: Dict[str, TemplateMetadata] = {}

    # ========================================================================
    # Template Discovery
    # ========================================================================

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

    def list_templates_with_metadata(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all templates with their metadata.

        Returns:
            Dictionary mapping template types to lists of template info with metadata
        """
        result = {}
        templates = self.list_templates()

        for template_type, template_names in templates.items():
            result[template_type] = []
            for name in template_names:
                metadata = self.get_template_metadata(template_type, name)
                result[template_type].append({
                    "name": name,
                    "metadata": metadata
                })

        return result

    def get_template_metadata(
        self,
        template_type: str,
        template_name: str
    ) -> Dict[str, Any]:
        """Get metadata for a specific template.

        Looks for template-meta.json in the template directory.
        Falls back to inferring metadata from template config files.

        Args:
            template_type: Type of template (cookiecutter, copier, maven)
            template_name: Name of the template

        Returns:
            Template metadata dictionary
        """
        cache_key = f"{template_type}/{template_name}"

        # Check cache
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key].to_dict()

        # Get template directory
        if template_type == "cookiecutter":
            template_dir = self.cookiecutter_dir / template_name
        elif template_type == "copier":
            template_dir = self.copier_dir / template_name
        elif template_type == "maven":
            template_dir = self.maven_dir / template_name
        else:
            return TemplateMetadata(name=template_name, template_type=template_type).to_dict()

        if not template_dir.exists():
            return TemplateMetadata(name=template_name, template_type=template_type).to_dict()

        # Try to load template-meta.json
        meta_file = template_dir / "template-meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    data["template_type"] = template_type
                    metadata = TemplateMetadata.from_dict(data)
                    self._metadata_cache[cache_key] = metadata
                    return metadata.to_dict()
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata for {cache_key}: {e}")

        # Infer metadata from template config
        metadata = self._infer_metadata(template_type, template_name, template_dir)
        self._metadata_cache[cache_key] = metadata
        return metadata.to_dict()

    def _infer_metadata(
        self,
        template_type: str,
        template_name: str,
        template_dir: Path
    ) -> TemplateMetadata:
        """Infer metadata from template configuration files.

        Args:
            template_type: Type of template
            template_name: Name of the template
            template_dir: Path to template directory

        Returns:
            Inferred TemplateMetadata
        """
        metadata = TemplateMetadata(
            name=template_name,
            template_type=template_type,
        )

        # Infer from cookiecutter.json
        if template_type == "cookiecutter":
            config_file = template_dir / "cookiecutter.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)

                    # Infer features from boolean options
                    for key, value in config.items():
                        if isinstance(value, str) and value.lower() in ("y", "n", "yes", "no"):
                            if value.lower() in ("y", "yes"):
                                metadata.features.append(key.replace("_", " ").title())

                    # Infer language
                    if "python_version" in config:
                        metadata.languages.append("Python")

                    # Infer frameworks from config keys
                    framework_indicators = {
                        "use_langchain": "LangChain",
                        "use_langgraph": "LangGraph",
                        "use_flask": "Flask",
                        "use_django": "Django",
                        "use_fastapi": "FastAPI",
                    }
                    for key, framework in framework_indicators.items():
                        if key in config:
                            metadata.frameworks.append(framework)

                    # Generate description
                    if not metadata.description:
                        metadata.description = config.get(
                            "description",
                            f"A {template_name.replace('-', ' ').replace('_', ' ')} template"
                        )

                except (json.JSONDecodeError, IOError):
                    pass

        # Infer from copier.yml
        elif template_type == "copier":
            for config_name in ["copier.yml", "copier.yaml"]:
                config_file = template_dir / config_name
                if config_file.exists():
                    try:
                        import yaml
                        with open(config_file, "r") as f:
                            config = yaml.safe_load(f)

                        # Extract description from project_name default
                        if "description" in config:
                            desc_config = config["description"]
                            if isinstance(desc_config, dict):
                                metadata.description = desc_config.get("default", "")

                        # Infer language from python_version
                        if "python_version" in config:
                            metadata.languages.append("Python")

                        # Infer frameworks
                        if "use_langchain" in config:
                            metadata.frameworks.append("LangChain")

                    except Exception:
                        pass
                    break

        # Infer from Maven pom.xml
        elif template_type == "maven":
            pom_file = template_dir / "pom.xml"
            if pom_file.exists():
                metadata.languages.append("Java")
                metadata.frameworks.append("Maven")

                # Basic description
                metadata.description = f"Maven archetype: {template_name}"

        # Infer tags from name
        name_parts = template_name.lower().replace("-", " ").replace("_", " ").split()
        common_tags = ["python", "java", "langchain", "langgraph", "agent", "api", "web"]
        metadata.tags = [part for part in name_parts if part in common_tags]

        return metadata

    def get_template_variables(
        self,
        template_type: str,
        template_name: str
    ) -> List[str]:
        """Get list of required variables for a template.

        Args:
            template_type: Type of template
            template_name: Name of template

        Returns:
            List of variable names required by the template
        """
        variables = []

        if template_type == "cookiecutter":
            config_file = self.cookiecutter_dir / template_name / "cookiecutter.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    # Filter out computed/derived variables (those with {{ }})
                    variables = [
                        k for k, v in config.items()
                        if not (isinstance(v, str) and "{{" in v)
                    ]
                except (json.JSONDecodeError, IOError):
                    pass

        elif template_type == "copier":
            for config_name in ["copier.yml", "copier.yaml"]:
                config_file = self.copier_dir / template_name / config_name
                if config_file.exists():
                    try:
                        import yaml
                        with open(config_file, "r") as f:
                            config = yaml.safe_load(f)
                        # Get non-internal variables (those not starting with _)
                        variables = [
                            k for k in config.keys()
                            if not k.startswith("_")
                        ]
                    except Exception:
                        pass
                    break

        elif template_type == "maven":
            # Common Maven archetype properties
            variables = ["groupId", "artifactId", "version", "package"]

        return variables

    def search_templates(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        frameworks: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        template_type: Optional[str] = None,
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Search templates with filtering and relevance scoring.

        Args:
            query: Free-text search query
            tags: Filter by tags
            category: Filter by category
            frameworks: Filter by frameworks
            languages: Filter by programming languages
            difficulty: Filter by difficulty level
            template_type: Filter by template type (cookiecutter, copier, maven)
            min_score: Minimum relevance score (0.0 to 1.0)

        Returns:
            List of matching templates sorted by relevance
        """
        results = []
        all_templates = self.list_templates()

        for ttype, template_names in all_templates.items():
            # Filter by template type if specified
            if template_type and ttype != template_type:
                continue

            for name in template_names:
                metadata_dict = self.get_template_metadata(ttype, name)
                metadata = TemplateMetadata.from_dict(metadata_dict)

                # Calculate relevance score
                score = metadata.matches_query(
                    query=query,
                    tags=tags,
                    category=category,
                    frameworks=frameworks,
                    languages=languages,
                    difficulty=difficulty,
                )

                # Include if meets minimum score or no criteria specified
                if score >= min_score or (
                    not query and not tags and not category
                    and not frameworks and not languages and not difficulty
                ):
                    results.append({
                        "name": name,
                        "template_type": ttype,
                        "metadata": metadata_dict,
                        "relevance_score": score,
                    })

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results

    # ========================================================================
    # Code Generation
    # ========================================================================

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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = LangGangMCPServer()

    # In a real implementation, this would start the MCP server
    # For now, we'll just log that it's ready
    logger.info("LangGang MCP Server initialized")
    logger.info(f"Available templates: {server.list_templates()}")

    # Demo search functionality
    logger.info("\nSearching for 'python langchain' templates:")
    results = server.search_templates(query="python langchain")
    for r in results:
        logger.info(f"  - {r['name']} (score: {r['relevance_score']:.2f})")


if __name__ == "__main__":
    main()
