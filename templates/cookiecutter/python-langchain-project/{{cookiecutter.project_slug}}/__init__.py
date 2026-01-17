"""
{{ cookiecutter.project_name }}

{{ cookiecutter.description }}
"""

__version__ = "{{ cookiecutter.version }}"

{% if cookiecutter.use_langchain == 'y' %}
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

def create_chain():
    """Create a simple LangChain chain."""
    prompt = ChatPromptTemplate.from_template("Hello {topic}")
    return prompt

{% endif %}

def main():
    """Main entry point."""
    print("{{ cookiecutter.project_name }} initialized!")
    {% if cookiecutter.use_langchain == 'y' %}
    chain = create_chain()
    print("LangChain chain created!")
    {% endif %}


if __name__ == "__main__":
    main()
