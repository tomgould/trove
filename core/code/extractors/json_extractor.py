"""
JSON code extractor.

Extracts meaningful chunks from JSON files with special handling for:
- composer.json (PHP dependencies)
- composer.lock (PHP lock file)
- package.json (Node.js dependencies)
- package-lock.json (Node.js lock file)
- tsconfig.json (TypeScript config)
- Generic JSON (top-level keys)
"""

import json
import logging
import re
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


def _strip_jsonc(text: str) -> str:
    """Strip comments and trailing commas from JSONC to produce valid JSON.

    Handles:
    - Single-line comments (// ...)
    - Block comments (/* ... */)
    - Trailing commas before } or ]
    Preserves strings containing // or /* literally.
    """
    result = []
    i = 0
    in_string = False
    length = len(text)

    while i < length:
        char = text[i]

        if in_string:
            result.append(char)
            # Handle escape sequences inside strings
            if char == '\\' and i + 1 < length:
                i += 1
                result.append(text[i])
            elif char == '"':
                in_string = False
            i += 1
            continue

        # Start of string
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        # Single-line comment
        if char == '/' and i + 1 < length and text[i + 1] == '/':
            # Skip to end of line
            while i < length and text[i] != '\n':
                i += 1
            continue

        # Block comment
        if char == '/' and i + 1 < length and text[i + 1] == '*':
            i += 2
            while i < length:
                if text[i] == '*' and i + 1 < length and text[i + 1] == '/':
                    i += 2
                    break
                i += 1
            continue

        result.append(char)
        i += 1

    # Remove trailing commas before } or ]
    cleaned = ''.join(result)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    return cleaned


class JSONExtractor(BaseExtractor):
    """Extract code chunks from JSON files."""

    language = "json"
    tree_sitter_language = "json"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract meaningful chunks from JSON."""
        chunks = []

        # Try to parse JSON, falling back to JSONC stripping for files
        # with comments or trailing commas (tsconfig.json, .eslintrc.json, etc.)
        try:
            data = json.loads(source_code)
        except json.JSONDecodeError:
            try:
                data = json.loads(_strip_jsonc(source_code))
            except json.JSONDecodeError:
                # Fall back to raw text chunk (JSONL, malformed JSON, etc.)
                logger.debug(f"JSON parse failed, indexing as raw text: {file_path}")
                return [self._create_document_chunk(source_code, file_path)]

        if not isinstance(data, dict):
            # Not an object, create a single document chunk
            return [self._create_document_chunk(source_code, file_path)]

        # Detect JSON type and extract accordingly
        json_type = self._detect_json_type(data, file_path)

        if json_type == "composer":
            chunks = self._extract_composer(data, source_code, file_path)
        elif json_type == "composer-lock":
            chunks = self._extract_composer_lock(data, source_code, file_path)
        elif json_type == "package":
            chunks = self._extract_package_json(data, source_code, file_path)
        elif json_type == "package-lock":
            chunks = self._extract_package_lock(data, source_code, file_path)
        elif json_type == "tsconfig":
            chunks = self._extract_tsconfig(data, source_code, file_path)
        else:
            # Generic JSON: extract top-level keys
            chunks = self._extract_generic(data, source_code, file_path)

        return chunks

    def _detect_json_type(self, data: dict, file_path: str) -> str:
        """Detect the type of JSON file."""
        filename = file_path.lower().split("/")[-1]

        # Composer (PHP)
        if filename == "composer.json":
            return "composer"
        if filename == "composer.lock":
            return "composer-lock"

        # Package.json (Node.js)
        if filename == "package.json":
            return "package"
        if filename == "package-lock.json":
            return "package-lock"

        # TypeScript config
        if filename in ("tsconfig.json", "jsconfig.json"):
            return "tsconfig"

        # Detect by content
        if "require" in data and ("autoload" in data or "autoload-dev" in data):
            return "composer"
        if "packages" in data and "_resolved" in str(data.get("packages", {}).get("", {})):
            return "composer-lock"
        if "dependencies" in data and ("scripts" in data or "main" in data):
            return "package"
        if "compilerOptions" in data:
            return "tsconfig"

        return "generic"

    def _extract_composer(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract info from composer.json."""
        chunks = []
        lines = source_code.split("\n")

        # Project metadata chunk
        name = data.get("name", "unnamed")
        description = data.get("description", "")
        php_version = data.get("require", {}).get("php", "")

        # Create main project chunk
        main_info = {
            "name": name,
            "description": description,
            "type": data.get("type", "library"),
            "license": data.get("license", ""),
            "php": php_version,
        }

        chunks.append(CodeChunk(
            content=json.dumps(main_info, indent=2),
            source_file=file_path,
            language=self.language,
            symbol_type="project",
            symbol_name=name,
            line_start=1,
            line_end=len(lines),
            docstring=description,
            extras={
                "json_type": "composer",
                "php_version": php_version,
                "type": data.get("type", "library"),
            },
        ))

        # Extract require dependencies
        require = data.get("require", {})
        if require:
            deps = [f"{pkg}: {ver}" for pkg, ver in require.items() if pkg != "php"]
            content = "require:\n" + "\n".join(f"  - {d}" for d in deps)

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="dependencies",
                symbol_name="require",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "composer",
                    "dependency_type": "require",
                    "packages": list(require.keys()),
                },
            ))

        # Extract require-dev dependencies
        require_dev = data.get("require-dev", {})
        if require_dev:
            deps = [f"{pkg}: {ver}" for pkg, ver in require_dev.items()]
            content = "require-dev:\n" + "\n".join(f"  - {d}" for d in deps)

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="dependencies",
                symbol_name="require-dev",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "composer",
                    "dependency_type": "require-dev",
                    "packages": list(require_dev.keys()),
                },
            ))

        # Extract autoload config
        autoload = data.get("autoload", {})
        if autoload:
            chunks.append(CodeChunk(
                content=json.dumps({"autoload": autoload}, indent=2),
                source_file=file_path,
                language=self.language,
                symbol_type="config",
                symbol_name="autoload",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "composer",
                    "psr4": list(autoload.get("psr-4", {}).keys()),
                },
            ))

        # Extract scripts
        scripts = data.get("scripts", {})
        if scripts:
            chunks.append(CodeChunk(
                content=json.dumps({"scripts": scripts}, indent=2),
                source_file=file_path,
                language=self.language,
                symbol_type="scripts",
                symbol_name="composer-scripts",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "composer",
                    "script_names": list(scripts.keys()),
                },
            ))

        return chunks

    def _extract_composer_lock(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract info from composer.lock."""
        chunks = []
        lines = source_code.split("\n")

        packages = data.get("packages", [])
        packages_dev = data.get("packages-dev", [])

        # Create summary chunk
        pkg_names = [p.get("name", "?") for p in packages[:20]]  # Limit to avoid huge chunks
        dev_pkg_names = [p.get("name", "?") for p in packages_dev[:10]]

        content = f"# Composer Lock\n\n"
        content += f"## Packages ({len(packages)} total)\n"
        content += "\n".join(f"- {name}" for name in pkg_names)
        if len(packages) > 20:
            content += f"\n- ... and {len(packages) - 20} more"

        content += f"\n\n## Dev Packages ({len(packages_dev)} total)\n"
        content += "\n".join(f"- {name}" for name in dev_pkg_names)
        if len(packages_dev) > 10:
            content += f"\n- ... and {len(packages_dev) - 10} more"

        chunks.append(CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="lockfile",
            symbol_name="composer.lock",
            line_start=1,
            line_end=len(lines),
            extras={
                "json_type": "composer-lock",
                "package_count": len(packages),
                "dev_package_count": len(packages_dev),
                "packages": [p.get("name") for p in packages],
            },
        ))

        return chunks

    def _extract_package_json(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract info from package.json."""
        chunks = []
        lines = source_code.split("\n")

        name = data.get("name", "unnamed")
        description = data.get("description", "")
        version = data.get("version", "")

        # Main project chunk
        main_info = {
            "name": name,
            "version": version,
            "description": description,
            "main": data.get("main", ""),
            "type": data.get("type", ""),
        }

        chunks.append(CodeChunk(
            content=json.dumps(main_info, indent=2),
            source_file=file_path,
            language=self.language,
            symbol_type="project",
            symbol_name=name,
            line_start=1,
            line_end=len(lines),
            docstring=description,
            extras={
                "json_type": "package",
                "version": version,
            },
        ))

        # Dependencies
        deps = data.get("dependencies", {})
        if deps:
            dep_list = [f"{pkg}: {ver}" for pkg, ver in deps.items()]
            content = "dependencies:\n" + "\n".join(f"  - {d}" for d in dep_list)

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="dependencies",
                symbol_name="dependencies",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "package",
                    "dependency_type": "dependencies",
                    "packages": list(deps.keys()),
                },
            ))

        # Dev dependencies
        dev_deps = data.get("devDependencies", {})
        if dev_deps:
            dep_list = [f"{pkg}: {ver}" for pkg, ver in dev_deps.items()]
            content = "devDependencies:\n" + "\n".join(f"  - {d}" for d in dep_list)

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="dependencies",
                symbol_name="devDependencies",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "package",
                    "dependency_type": "devDependencies",
                    "packages": list(dev_deps.keys()),
                },
            ))

        # Scripts
        scripts = data.get("scripts", {})
        if scripts:
            chunks.append(CodeChunk(
                content=json.dumps({"scripts": scripts}, indent=2),
                source_file=file_path,
                language=self.language,
                symbol_type="scripts",
                symbol_name="npm-scripts",
                line_start=1,
                line_end=len(lines),
                extras={
                    "json_type": "package",
                    "script_names": list(scripts.keys()),
                },
            ))

        return chunks

    def _extract_package_lock(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract summary from package-lock.json."""
        chunks = []
        lines = source_code.split("\n")

        name = data.get("name", "unnamed")
        packages = data.get("packages", {})

        # Count packages (exclude root "")
        pkg_count = len([k for k in packages.keys() if k])

        content = f"# Package Lock: {name}\n\nTotal packages: {pkg_count}"

        chunks.append(CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="lockfile",
            symbol_name="package-lock.json",
            line_start=1,
            line_end=len(lines),
            extras={
                "json_type": "package-lock",
                "package_count": pkg_count,
            },
        ))

        return chunks

    def _extract_tsconfig(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract info from tsconfig.json."""
        chunks = []
        lines = source_code.split("\n")

        compiler_options = data.get("compilerOptions", {})
        include = data.get("include", [])
        exclude = data.get("exclude", [])

        # Truncate if too long
        content = source_code
        if len(content) > 2000:
            content = json.dumps(data, indent=2)[:2000] + "\n// truncated..."

        chunks.append(CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="config",
            symbol_name=file_path.split("/")[-1],
            line_start=1,
            line_end=len(lines),
            extras={
                "json_type": "tsconfig",
                "target": compiler_options.get("target", ""),
                "module": compiler_options.get("module", ""),
                "strict": compiler_options.get("strict", False),
                "include": include,
                "exclude": exclude,
            },
        ))

        return chunks

    def _extract_generic(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract top-level keys from generic JSON."""
        chunks = []
        lines = source_code.split("\n")

        # For small files, just create one chunk
        if len(source_code) < 1500:
            return [self._create_document_chunk(source_code, file_path)]

        # For larger files, extract top-level keys
        for key, value in data.items():
            content = json.dumps({key: value}, indent=2)

            # Truncate very long content
            if len(content) > 2000:
                content = content[:2000] + "\n// truncated..."

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="key",
                symbol_name=key,
                line_start=1,
                line_end=len(lines),
                extras={"json_type": "generic"},
            ))

        return chunks

    def _create_document_chunk(self, source_code: str, file_path: str) -> CodeChunk:
        """Create a chunk for the entire document."""
        lines = source_code.split("\n")
        content = source_code if len(source_code) < 2000 else source_code[:2000] + "\n// truncated..."

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="document",
            symbol_name=file_path.split("/")[-1],
            line_start=1,
            line_end=len(lines),
        )
