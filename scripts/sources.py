#!/usr/bin/env python3
"""
CLI script for managing source connectors.

Usage:
    python scripts/sources.py connect github --client myproject --owner myorg
    python scripts/sources.py connect github --client myproject --repo owner/repo
    python scripts/sources.py crawl --client myproject --url https://docs.example.com
    python scripts/sources.py sync --client myproject
    python scripts/sources.py list --client myproject
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client import load_client, discover_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_connect_github(args):
    """Connect a GitHub source to a client."""
    from core.sources.github_models import GitHubSourceConfig
    from core.sources.github_connector import GitHubSourceConnector

    client = load_client(args.client)

    # Build config from CLI args
    config_dict = {}
    if args.repo and "/" in args.repo:
        parts = args.repo.split("/", 1)
        config_dict["owner"] = parts[0]
        config_dict["repo"] = parts[1]
    elif args.owner:
        config_dict["owner"] = args.owner
    else:
        logger.error("Either --owner or --repo (as owner/repo) is required")
        return

    if args.token_env:
        config_dict["token_env"] = args.token_env

    if args.include:
        config_dict["include_repos"] = args.include
    if args.exclude:
        config_dict["exclude_repos"] = args.exclude

    config_dict["include_forks"] = args.include_forks
    config_dict["include_private"] = not args.public_only

    config = GitHubSourceConfig.from_dict(config_dict)
    connector = GitHubSourceConnector(config)

    logger.info("Connecting GitHub source: %s", config.owner)
    sources_dir = str(client.sources_path)
    result = connector.sync(sources_dir)

    print(f"\nGitHub sync complete:")
    print(f"  Files added:   {result.files_added}")
    print(f"  Files updated: {result.files_updated}")
    if result.errors:
        print(f"  Errors:        {len(result.errors)}")
        for err in result.errors:
            print(f"    - {err}")

    if result.total_changes > 0:
        print(f"\nSource files stored at: {client.sources_path / 'github'}")
        print("Run 'python scripts/ingest.py --client %s --rebuild' to index." % args.client)


def cmd_crawl_web(args):
    """Crawl a website and store as source documents."""
    from core.sources.web_crawler import WebSourceConfig, WebSourceConnector

    client = load_client(args.client)

    config = WebSourceConfig(
        url=args.url,
        mode=args.mode,
        stay_in_path=not args.no_stay_in_path,
        max_pages=args.max_pages,
        wait_time=args.wait_time,
        timeout=args.timeout,
    )
    connector = WebSourceConnector(config)

    logger.info("Crawling %s (mode=%s, max_pages=%d)", args.url, args.mode, args.max_pages)
    sources_dir = str(client.sources_path)
    result = connector.sync(sources_dir)

    print(f"\nWeb crawl complete:")
    print(f"  Pages added:   {result.files_added}")
    print(f"  Pages updated: {result.files_updated}")
    if result.errors:
        print(f"  Errors:        {len(result.errors)}")
        for err in result.errors:
            print(f"    - {err}")

    if result.total_changes > 0:
        print(f"\nSource files stored at: {client.sources_path / 'web' / config.domain}")
        print("Run 'python scripts/ingest.py --client %s --rebuild' to index." % args.client)


def cmd_sync(args):
    """Sync all configured sources for a client."""
    from scripts.ingest import sync_client_sources

    if args.all:
        client_names = discover_clients()
        if not client_names:
            logger.error("No clients found")
            return
        for name in client_names:
            print(f"\n--- {name} ---")
            sync_client_sources(name)
    elif args.client:
        sync_client_sources(args.client)
    else:
        logger.error("Either --client or --all is required")


def cmd_list(args):
    """List configured sources for a client."""
    client = load_client(args.client)
    sources_config = client.sources_config
    sources_path = client.sources_path

    print(f"\nSources for {client.name} ({client.machine_name}):")
    print(f"  Sources directory: {sources_path}")
    print(f"  Directory exists: {sources_path.exists()}")

    # Show configured sources from client.yaml
    if sources_config:
        github_sources = sources_config.get("github", [])
        web_sources = sources_config.get("web", [])

        if github_sources:
            print(f"\n  GitHub sources ({len(github_sources)}):")
            for i, gh in enumerate(github_sources, 1):
                owner = gh.get("owner", "?")
                repo = gh.get("repo", "")
                sync = gh.get("sync_on_ingest", False)
                if repo:
                    print(f"    {i}. {owner}/{repo} (sync_on_ingest={sync})")
                else:
                    print(f"    {i}. {owner}/* (sync_on_ingest={sync})")

        if web_sources:
            print(f"\n  Web sources ({len(web_sources)}):")
            for i, web in enumerate(web_sources, 1):
                url = web.get("url", "?")
                mode = web.get("mode", "crawl")
                max_pages = web.get("max_pages", 500)
                print(f"    {i}. {url} (mode={mode}, max_pages={max_pages})")
    else:
        print("\n  No sources configured in client.yaml")

    # Show files on disk
    if sources_path.exists():
        github_dir = sources_path / "github"
        web_dir = sources_path / "web"

        if github_dir.exists():
            repos = [d for d in github_dir.rglob("*/.git") if d.is_dir()]
            if repos:
                print(f"\n  GitHub repos on disk ({len(repos)}):")
                for repo_git in sorted(repos):
                    repo_dir = repo_git.parent
                    rel = repo_dir.relative_to(github_dir)
                    print(f"    - {rel}")

        if web_dir.exists():
            domains = [d for d in web_dir.iterdir() if d.is_dir()]
            if domains:
                print(f"\n  Web domains on disk ({len(domains)}):")
                for domain_dir in sorted(domains):
                    md_count = len(list(domain_dir.glob("*.md")))
                    print(f"    - {domain_dir.name} ({md_count} pages)")


def main():
    parser = argparse.ArgumentParser(
        description="Trove Source Connector CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- connect command ---
    connect_parser = subparsers.add_parser("connect", help="Connect a new source")
    connect_sub = connect_parser.add_subparsers(dest="source_type", help="Source type")

    # connect github
    gh_parser = connect_sub.add_parser("github", help="Connect GitHub repos")
    gh_parser.add_argument("--client", "-c", required=True, help="Client name")
    gh_parser.add_argument("--owner", "-o", help="GitHub user or org name")
    gh_parser.add_argument("--repo", "-r", help="Single repo (owner/repo format)")
    gh_parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Env var for GitHub token (default: GITHUB_TOKEN)")
    gh_parser.add_argument("--include", nargs="+", help="Only include these repos")
    gh_parser.add_argument("--exclude", nargs="+", help="Exclude these repos")
    gh_parser.add_argument("--include-forks", action="store_true", help="Include forked repos")
    gh_parser.add_argument("--public-only", action="store_true", help="Only include public repos")

    # --- crawl command ---
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website")
    crawl_parser.add_argument("--client", "-c", required=True, help="Client name")
    crawl_parser.add_argument("--url", "-u", required=True, help="URL to crawl")
    crawl_parser.add_argument("--mode", "-m", default="crawl", choices=["crawl", "docs", "swagger"], help="Crawl mode")
    crawl_parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to crawl")
    crawl_parser.add_argument("--wait-time", type=float, default=1.0, help="Delay between requests (seconds)")
    crawl_parser.add_argument("--timeout", type=int, default=30, help="HTTP request timeout (seconds)")
    crawl_parser.add_argument("--no-stay-in-path", action="store_true", help="Don't restrict to starting URL path")

    # --- sync command ---
    sync_parser = subparsers.add_parser("sync", help="Sync configured sources")
    sync_parser.add_argument("--client", "-c", help="Client name")
    sync_parser.add_argument("--all", "-a", action="store_true", help="Sync all clients")

    # --- list command ---
    list_parser = subparsers.add_parser("list", help="List configured sources")
    list_parser.add_argument("--client", "-c", required=True, help="Client name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "connect":
        if args.source_type == "github":
            cmd_connect_github(args)
        else:
            connect_parser.print_help()
    elif args.command == "crawl":
        cmd_crawl_web(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
