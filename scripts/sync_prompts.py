#!/usr/bin/env python3
"""
Sync new/trending prompts from the prompts.chat API into the local
prompts client documents directory for Trove indexing.

Usage:
    python scripts/sync_prompts.py                  # Sync newest + popular
    python scripts/sync_prompts.py --pages 3        # Fetch 3 pages per sort
    python scripts/sync_prompts.py --dry-run        # Preview without writing
    python scripts/sync_prompts.py --reindex        # Reindex after sync
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

# Add project root to path
TROVE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TROVE_ROOT))

from clients.prompts.api_client import PromptsAPIClient, PromptResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sync-prompts")

PROMPTS_DIR = TROVE_ROOT / "clients" / "prompts" / "documents" / "prompts_chat" / "prompts"


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80]


def prompt_to_markdown(prompt: PromptResult) -> str:
    """Convert a PromptResult into a markdown document matching existing format."""
    lines = [f"# {prompt.title}", ""]
    if prompt.description:
        lines.append(prompt.description)
        lines.append("")
    lines.append("### Content")
    lines.append("")
    if prompt.content:
        lines.append(prompt.content)
    lines.append("")
    if prompt.tags:
        lines.append(f"Tags: {', '.join(prompt.tags)}")
    if prompt.category:
        lines.append(f"Category: {prompt.category}")
    if prompt.type != "TEXT":
        lines.append(f"Type: {prompt.type}")
    if prompt.vote_count:
        lines.append(f"Votes: {prompt.vote_count}")
    return "\n".join(lines)


def get_existing_ids(prompts_dir: Path) -> set[str]:
    """Scan existing files for prompt IDs (ID prefix in filename)."""
    ids = set()
    if not prompts_dir.exists():
        return ids
    for f in prompts_dir.glob("*.md"):
        # Filenames are like: cmj1zb0yc000ivl0rc9ka596e_job-interviewer.md
        parts = f.stem.split("_", 1)
        if parts:
            ids.add(parts[0])
    return ids


def sync_prompts(
    pages: int = 2,
    per_page: int = 24,
    dry_run: bool = False,
) -> list[Path]:
    """Fetch newest and popular prompts, save new ones to disk.

    Returns list of newly created file paths.
    """
    api = PromptsAPIClient()
    existing_ids = get_existing_ids(PROMPTS_DIR)
    logger.info(f"Found {len(existing_ids)} existing prompts on disk")

    new_prompts: dict[str, PromptResult] = {}

    for sort_order in ("newest", "popular"):
        for page in range(1, pages + 1):
            try:
                resp = api.search_prompts(sort=sort_order, page=page, limit=per_page)
                for p in resp.prompts:
                    if p.id and p.id not in existing_ids and p.id not in new_prompts:
                        new_prompts[p.id] = p
                logger.info(f"Fetched {sort_order} page {page}: {len(resp.prompts)} prompts")
            except Exception as e:
                logger.warning(f"Failed to fetch {sort_order} page {page}: {e}")

    if not new_prompts:
        logger.info("No new prompts to sync")
        return []

    logger.info(f"Found {len(new_prompts)} new prompts to sync")

    if dry_run:
        for pid, p in new_prompts.items():
            logger.info(f"  [DRY RUN] Would save: {pid}_{slugify(p.title)}.md — {p.title}")
        return []

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for pid, p in new_prompts.items():
        filename = f"{pid}_{slugify(p.title)}.md"
        filepath = PROMPTS_DIR / filename
        content = prompt_to_markdown(p)
        filepath.write_text(content, encoding="utf-8")
        created.append(filepath)
        logger.info(f"  Saved: {filename}")

    logger.info(f"Synced {len(created)} new prompts")
    return created


def main():
    parser = argparse.ArgumentParser(description="Sync prompts from prompts.chat API")
    parser.add_argument("--pages", type=int, default=2, help="Pages to fetch per sort order (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--reindex", action="store_true", help="Run incremental reindex after sync")
    args = parser.parse_args()

    created = sync_prompts(pages=args.pages, dry_run=args.dry_run)

    if args.reindex and created and not args.dry_run:
        logger.info("Running incremental reindex for prompts client...")
        subprocess.run(
            [sys.executable, str(TROVE_ROOT / "scripts" / "ingest.py"), "--client", "prompts"],
            cwd=str(TROVE_ROOT),
        )


if __name__ == "__main__":
    main()
