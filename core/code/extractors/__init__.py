"""
Language-specific extractors for code parsing.

Each extractor knows how to extract symbols (functions, classes, etc.)
from a specific language's AST.
"""

from .base import BaseExtractor, CodeChunk
from .python import PythonExtractor
from .php import PHPExtractor
from .javascript import JavaScriptExtractor
from .typescript import TypeScriptExtractor
from .html import HTMLExtractor
from .css import CSSExtractor
from .yaml_extractor import YAMLExtractor
from .json_extractor import JSONExtractor
from .gherkin_extractor import GherkinExtractor
from .dart import DartExtractor

# Map file extensions to extractors
EXTRACTOR_MAP = {
    # Python
    ".py": PythonExtractor,

    # PHP and Drupal
    ".php": PHPExtractor,
    ".module": PHPExtractor,
    ".install": PHPExtractor,
    ".theme": PHPExtractor,
    ".inc": PHPExtractor,
    ".profile": PHPExtractor,

    # JavaScript
    ".js": JavaScriptExtractor,
    ".jsx": JavaScriptExtractor,
    ".mjs": JavaScriptExtractor,
    ".cjs": JavaScriptExtractor,

    # TypeScript
    ".ts": TypeScriptExtractor,
    ".tsx": TypeScriptExtractor,

    # HTML and MDX
    ".html": HTMLExtractor,
    ".htm": HTMLExtractor,
    ".vue": HTMLExtractor,
    ".svelte": HTMLExtractor,
    ".mdx": HTMLExtractor,  # MDX is Markdown + JSX

    # CSS
    ".css": CSSExtractor,
    ".scss": CSSExtractor,
    ".sass": CSSExtractor,
    ".less": CSSExtractor,

    # YAML
    ".yml": YAMLExtractor,
    ".yaml": YAMLExtractor,

    # JSON
    ".json": JSONExtractor,

    # Gherkin/Cucumber
    ".feature": GherkinExtractor,

    # Dart
    ".dart": DartExtractor,
}

__all__ = [
    "BaseExtractor",
    "CodeChunk",
    "PythonExtractor",
    "PHPExtractor",
    "JavaScriptExtractor",
    "TypeScriptExtractor",
    "HTMLExtractor",
    "CSSExtractor",
    "YAMLExtractor",
    "JSONExtractor",
    "GherkinExtractor",
    "DartExtractor",
    "EXTRACTOR_MAP",
]
