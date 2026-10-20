"""
YAML code extractor.

Extracts meaningful chunks from YAML files with special handling for:
- docker-compose.yml (services)
- OpenAPI/Swagger specs (endpoints)
- Kubernetes manifests (resources)
- GitHub Actions workflows (jobs)
- General YAML (top-level keys)
"""

import logging
import re
from typing import Optional

import yaml

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class _PermissiveYAMLLoader(yaml.SafeLoader):
    """YAML loader that treats unknown tags (e.g. !GetAtt, !Ref, !Sub) as strings.

    CloudFormation, SAM, Helm, and other tools use custom YAML tags that
    yaml.safe_load() rejects. This loader converts them to their string
    representation so the file can still be parsed and indexed.
    """
    pass


def _unknown_tag_constructor(loader, tag_suffix, node):
    """Handle any unknown !Tag by returning its value as a string."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return str(node)


_PermissiveYAMLLoader.add_multi_constructor('!', _unknown_tag_constructor)


class YAMLExtractor(BaseExtractor):
    """Extract code chunks from YAML files."""

    language = "yaml"
    tree_sitter_language = "yaml"

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract meaningful chunks from YAML."""
        chunks = []

        # Try to parse YAML, falling back to permissive loader for files
        # with custom tags (CloudFormation !GetAtt, !Ref, !Sub, etc.)
        try:
            data = yaml.safe_load(source_code)
        except yaml.YAMLError:
            try:
                data = yaml.load(source_code, Loader=_PermissiveYAMLLoader)
            except yaml.YAMLError:
                # Try multi-document YAML (--- separators, e.g. K8s manifests)
                try:
                    docs = list(yaml.safe_load_all(source_code))
                    docs = [d for d in docs if isinstance(d, dict)]
                    if docs:
                        for doc in docs:
                            yaml_type = self._detect_yaml_type(doc, file_path)
                            if yaml_type == "kubernetes":
                                chunks.extend(self._extract_kubernetes(doc, yaml.dump(doc, default_flow_style=False), file_path))
                            else:
                                chunks.extend(self._extract_generic(doc, yaml.dump(doc, default_flow_style=False), file_path))
                        return chunks
                except yaml.YAMLError:
                    pass

                # Final fallback: index as raw text (Helm templates, etc.)
                logger.debug(f"YAML parse failed, indexing as raw text: {file_path}")
                return [self._create_document_chunk(source_code, file_path)]

        if not isinstance(data, dict):
            # Not a mapping, create a single document chunk
            return [self._create_document_chunk(source_code, file_path)]

        # Detect YAML type and extract accordingly
        yaml_type = self._detect_yaml_type(data, file_path)

        if yaml_type == "docker-compose":
            chunks = self._extract_docker_compose(data, source_code, file_path)
        elif yaml_type == "openapi":
            chunks = self._extract_openapi(data, source_code, file_path)
        elif yaml_type == "kubernetes":
            chunks = self._extract_kubernetes(data, source_code, file_path)
        elif yaml_type == "github-actions":
            chunks = self._extract_github_actions(data, source_code, file_path)
        else:
            # Generic YAML: extract top-level keys
            chunks = self._extract_generic(data, source_code, file_path)

        return chunks

    def _detect_yaml_type(self, data: dict, file_path: str) -> str:
        """Detect the type of YAML file."""
        filename = file_path.lower().split("/")[-1]

        # Docker Compose
        if "services" in data and ("version" in data or "networks" in data or "volumes" in data):
            return "docker-compose"
        if "docker-compose" in filename or "compose" in filename:
            return "docker-compose"

        # OpenAPI/Swagger
        if "openapi" in data or "swagger" in data:
            return "openapi"
        if "paths" in data and isinstance(data.get("paths"), dict):
            return "openapi"

        # Kubernetes
        if "apiVersion" in data and "kind" in data:
            return "kubernetes"

        # GitHub Actions
        if "on" in data and ("jobs" in data or "workflow_dispatch" in str(data.get("on", ""))):
            return "github-actions"
        if ".github/workflows" in file_path:
            return "github-actions"

        return "generic"

    def _extract_docker_compose(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract services from docker-compose.yml."""
        chunks = []
        services = data.get("services", {})

        for service_name, service_config in services.items():
            # Find the service in source code
            content, line_start, line_end = self._find_yaml_section(
                source_code, f"  {service_name}:", indent_level=2
            )

            if not content:
                content = yaml.dump({service_name: service_config}, default_flow_style=False)
                line_start, line_end = 1, content.count("\n") + 1

            # Extract useful info
            image = service_config.get("image", "")
            ports = service_config.get("ports", [])
            depends_on = service_config.get("depends_on", [])

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="service",
                symbol_name=service_name,
                line_start=line_start,
                line_end=line_end,
                extras={
                    "yaml_type": "docker-compose",
                    "image": image,
                    "ports": ports,
                    "depends_on": depends_on if isinstance(depends_on, list) else list(depends_on.keys()) if isinstance(depends_on, dict) else [],
                },
            ))

        return chunks

    def _extract_openapi(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract endpoints from OpenAPI/Swagger spec."""
        chunks = []
        paths = data.get("paths", {})

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue

            for method, spec in methods.items():
                if method.startswith("x-") or not isinstance(spec, dict):
                    continue

                # Find in source
                # OpenAPI paths are typically indented
                search_key = f'"{path}"' if " " in path else path
                content, line_start, line_end = self._find_yaml_section(
                    source_code, search_key, indent_level=2
                )

                if not content:
                    content = yaml.dump({method: spec}, default_flow_style=False)
                    line_start, line_end = 1, content.count("\n") + 1

                summary = spec.get("summary", "")
                description = spec.get("description", "")
                operation_id = spec.get("operationId", "")
                tags = spec.get("tags", [])

                symbol_name = f"{method.upper()} {path}"

                chunks.append(CodeChunk(
                    content=content,
                    source_file=file_path,
                    language=self.language,
                    symbol_type="endpoint",
                    symbol_name=symbol_name,
                    line_start=line_start,
                    line_end=line_end,
                    docstring=description or summary,
                    extras={
                        "yaml_type": "openapi",
                        "http_method": method,
                        "path": path,
                        "summary": summary,
                        "operation_id": operation_id,
                        "tags": tags,
                    },
                ))

        return chunks

    def _extract_kubernetes(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract resources from Kubernetes manifest."""
        chunks = []

        kind = data.get("kind", "Unknown")
        api_version = data.get("apiVersion", "")
        metadata = data.get("metadata", {})
        name = metadata.get("name", "unnamed")
        namespace = metadata.get("namespace", "default")

        lines = source_code.split("\n")

        chunks.append(CodeChunk(
            content=source_code,
            source_file=file_path,
            language=self.language,
            symbol_type="resource",
            symbol_name=f"{kind}/{name}",
            line_start=1,
            line_end=len(lines),
            extras={
                "yaml_type": "kubernetes",
                "kind": kind,
                "api_version": api_version,
                "name": name,
                "namespace": namespace,
            },
        ))

        return chunks

    def _extract_github_actions(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract jobs from GitHub Actions workflow."""
        chunks = []
        jobs = data.get("jobs", {})

        # Workflow name
        workflow_name = data.get("name", file_path.split("/")[-1])

        for job_name, job_config in jobs.items():
            content, line_start, line_end = self._find_yaml_section(
                source_code, f"  {job_name}:", indent_level=2
            )

            if not content:
                content = yaml.dump({job_name: job_config}, default_flow_style=False)
                line_start, line_end = 1, content.count("\n") + 1

            runs_on = job_config.get("runs-on", "")
            needs = job_config.get("needs", [])
            if isinstance(needs, str):
                needs = [needs]

            steps = job_config.get("steps", [])
            step_names = [s.get("name", s.get("uses", "unnamed")) for s in steps if isinstance(s, dict)]

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="job",
                symbol_name=job_name,
                line_start=line_start,
                line_end=line_end,
                extras={
                    "yaml_type": "github-actions",
                    "workflow": workflow_name,
                    "runs_on": runs_on,
                    "needs": needs,
                    "steps": step_names,
                },
            ))

        return chunks

    def _extract_generic(
        self,
        data: dict,
        source_code: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Extract top-level keys from generic YAML."""
        chunks = []

        for key, value in data.items():
            content, line_start, line_end = self._find_yaml_section(
                source_code, f"{key}:", indent_level=0
            )

            if not content:
                content = yaml.dump({key: value}, default_flow_style=False)
                line_start, line_end = 1, content.count("\n") + 1

            # Truncate very long content
            if len(content) > 2000:
                content = content[:2000] + "\n# truncated..."

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type="key",
                symbol_name=key,
                line_start=line_start,
                line_end=line_end,
                extras={"yaml_type": "generic"},
            ))

        return chunks

    def _find_yaml_section(
        self,
        source_code: str,
        key: str,
        indent_level: int = 0,
    ) -> tuple[str, int, int]:
        """
        Find a YAML section by its key.

        Returns (content, line_start, line_end).
        """
        lines = source_code.split("\n")
        indent = "  " * indent_level

        start_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith(key.strip()):
                start_line = i
                break

        if start_line is None:
            return "", 0, 0

        # Find the end of this section (next key at same or lower indent level)
        end_line = len(lines)
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            current_indent = len(line) - len(line.lstrip())
            # If we find a line at same or lower indent, section ends
            if current_indent <= base_indent and line.strip() and not line.strip().startswith("#"):
                end_line = i
                break

        content = "\n".join(lines[start_line:end_line])
        return content, start_line + 1, end_line

    def _create_document_chunk(self, source_code: str, file_path: str) -> CodeChunk:
        """Create a chunk for the entire document."""
        lines = source_code.split("\n")
        content = source_code if len(source_code) < 2000 else source_code[:2000] + "\n# truncated..."

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="document",
            symbol_name=file_path.split("/")[-1],
            line_start=1,
            line_end=len(lines),
        )
