"""
Gherkin/Cucumber feature file extractor.

Extracts meaningful chunks from .feature files including:
- Features with their descriptions
- Scenarios and Scenario Outlines
- Background sections
- Tags and metadata
"""

import logging
import re
from typing import Optional

from .base import BaseExtractor, CodeChunk

logger = logging.getLogger(__name__)


class GherkinExtractor(BaseExtractor):
    """Extract code chunks from Gherkin feature files."""

    language = "gherkin"
    tree_sitter_language = None  # No tree-sitter support, use regex parsing

    # Gherkin keywords
    FEATURE_PATTERN = re.compile(r'^Feature:\s*(.+)$', re.MULTILINE)
    SCENARIO_PATTERN = re.compile(r'^(\s*)(Scenario Outline|Scenario|Example):\s*(.+)$', re.MULTILINE)
    BACKGROUND_PATTERN = re.compile(r'^(\s*)Background:\s*(.*)$', re.MULTILINE)
    TAG_PATTERN = re.compile(r'@[\w-]+')

    def extract(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Extract features and scenarios from Gherkin file."""
        chunks = []
        lines = source_code.split('\n')

        # Extract feature-level info
        feature_match = self.FEATURE_PATTERN.search(source_code)
        feature_name = feature_match.group(1).strip() if feature_match else file_path.split('/')[-1]

        # Get feature-level tags
        feature_tags = []
        if feature_match:
            # Look for tags before Feature:
            feature_start = feature_match.start()
            pre_feature = source_code[:feature_start]
            feature_tags = self.TAG_PATTERN.findall(pre_feature)

        # Get feature description (lines between Feature: and first Scenario/Background)
        feature_description = self._extract_feature_description(source_code)

        # Create feature chunk
        chunks.append(CodeChunk(
            content=source_code[:1500] if len(source_code) > 1500 else source_code,
            source_file=file_path,
            language=self.language,
            symbol_type="feature",
            symbol_name=feature_name,
            line_start=1,
            line_end=len(lines),
            docstring=feature_description,
            extras={
                "tags": feature_tags,
                "scenario_count": len(self.SCENARIO_PATTERN.findall(source_code)),
            },
        ))

        # Extract individual scenarios
        scenario_chunks = self._extract_scenarios(source_code, file_path, feature_name)
        chunks.extend(scenario_chunks)

        # Extract background if present
        background_chunk = self._extract_background(source_code, file_path, feature_name)
        if background_chunk:
            chunks.append(background_chunk)

        return chunks

    def _extract_feature_description(self, source_code: str) -> str:
        """Extract the description text after Feature: line."""
        lines = source_code.split('\n')
        description_lines = []
        in_description = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('Feature:'):
                in_description = True
                continue

            if in_description:
                # Stop at Background, Scenario, or tags
                if (stripped.startswith(('Background:', 'Scenario:', 'Scenario Outline:', 'Example:', '@'))
                    or stripped.startswith('#')):
                    break
                if stripped:
                    description_lines.append(stripped)

        return ' '.join(description_lines)

    def _extract_scenarios(
        self,
        source_code: str,
        file_path: str,
        feature_name: str,
    ) -> list[CodeChunk]:
        """Extract individual scenarios as chunks."""
        chunks = []
        lines = source_code.split('\n')

        # Find all scenario starts
        scenario_starts = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('Scenario:', 'Scenario Outline:', 'Example:')):
                # Get scenario type and name
                if stripped.startswith('Scenario Outline:'):
                    scenario_type = 'scenario_outline'
                    scenario_name = stripped[17:].strip()
                elif stripped.startswith('Scenario:'):
                    scenario_type = 'scenario'
                    scenario_name = stripped[9:].strip()
                else:  # Example:
                    scenario_type = 'example'
                    scenario_name = stripped[8:].strip()

                # Get tags from preceding lines
                tags = []
                for j in range(i - 1, max(0, i - 5), -1):
                    tag_line = lines[j].strip()
                    if tag_line.startswith('@'):
                        tags.extend(self.TAG_PATTERN.findall(tag_line))
                    elif tag_line and not tag_line.startswith('#'):
                        break

                scenario_starts.append({
                    'line': i,
                    'name': scenario_name,
                    'type': scenario_type,
                    'tags': tags,
                })

        # Extract each scenario's content
        for idx, scenario in enumerate(scenario_starts):
            start_line = scenario['line']

            # Find end (next scenario or end of file)
            if idx + 1 < len(scenario_starts):
                # End before next scenario's tags
                end_line = scenario_starts[idx + 1]['line']
                # Back up to exclude tags of next scenario
                while end_line > start_line and lines[end_line - 1].strip().startswith('@'):
                    end_line -= 1
            else:
                end_line = len(lines)

            # Extract content
            content_lines = lines[start_line:end_line]
            content = '\n'.join(content_lines).strip()

            # Extract steps
            steps = self._extract_steps(content)

            chunks.append(CodeChunk(
                content=content,
                source_file=file_path,
                language=self.language,
                symbol_type=scenario['type'],
                symbol_name=scenario['name'],
                line_start=start_line + 1,
                line_end=end_line,
                extras={
                    "feature": feature_name,
                    "tags": scenario['tags'],
                    "steps": steps,
                },
            ))

        return chunks

    def _extract_background(
        self,
        source_code: str,
        file_path: str,
        feature_name: str,
    ) -> Optional[CodeChunk]:
        """Extract Background section if present."""
        lines = source_code.split('\n')

        background_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('Background:'):
                background_start = i
                break

        if background_start is None:
            return None

        # Find end (first Scenario or end)
        background_end = len(lines)
        for i in range(background_start + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith(('Scenario:', 'Scenario Outline:', 'Example:', '@')):
                background_end = i
                break

        content_lines = lines[background_start:background_end]
        content = '\n'.join(content_lines).strip()
        steps = self._extract_steps(content)

        return CodeChunk(
            content=content,
            source_file=file_path,
            language=self.language,
            symbol_type="background",
            symbol_name="Background",
            line_start=background_start + 1,
            line_end=background_end,
            extras={
                "feature": feature_name,
                "steps": steps,
            },
        )

    def _extract_steps(self, content: str) -> list[str]:
        """Extract Given/When/Then steps from content."""
        steps = []
        step_pattern = re.compile(r'^\s*(Given|When|Then|And|But)\s+(.+)$', re.MULTILINE)

        for match in step_pattern.finditer(content):
            keyword = match.group(1)
            step_text = match.group(2).strip()
            steps.append(f"{keyword} {step_text}")

        return steps
