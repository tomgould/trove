"""Handler functions for source connector tools."""

from mcp.types import TextContent

from core.client import load_client

from mcp_server.shared import CLIENTS_DIR


def handle_connect_github(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    owner = arguments.get("owner")
    repo = arguments.get("repo", "")
    token_env = arguments.get("token_env", "GITHUB_TOKEN")
    include_repos = arguments.get("include_repos", [])
    exclude_repos = arguments.get("exclude_repos", [])

    if not client_name or not owner:
        return [TextContent(type="text", text="Error: client_name and owner are required")]

    try:
        client = load_client(client_name, clients_dir=CLIENTS_DIR)
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading client '{client_name}': {e}")]

    from core.sources.github_connector import GitHubSourceConnector
    from core.sources.github_models import GitHubSourceConfig

    config = GitHubSourceConfig(
        owner=owner,
        repo=repo,
        token_env=token_env,
        include_repos=include_repos,
        exclude_repos=exclude_repos,
    )
    connector = GitHubSourceConnector(config)

    sources_dir = str(client.sources_path)
    result = connector.sync(sources_dir)

    output = f"GitHub source connected for **{client_name}**\n\n"
    output += f"- Owner: {owner}\n"
    if repo:
        output += f"- Repo: {repo}\n"
    output += f"- Files added: {result.files_added}\n"
    output += f"- Files updated: {result.files_updated}\n"
    if result.errors:
        output += f"- Errors ({len(result.errors)}):\n"
        for err in result.errors[:10]:
            output += f"  - {err}\n"
    output += f"\nSource files at: `{sources_dir}/github/{owner}/`"
    return [TextContent(type="text", text=output)]


def handle_crawl_web(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    url = arguments.get("url")
    max_pages = arguments.get("max_pages", 100)
    stay_in_path = arguments.get("stay_in_path", True)

    if not client_name or not url:
        return [TextContent(type="text", text="Error: client_name and url are required")]

    try:
        client = load_client(client_name, clients_dir=CLIENTS_DIR)
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading client '{client_name}': {e}")]

    from core.sources.web_crawler import WebSourceConfig, WebSourceConnector

    config = WebSourceConfig(
        url=url,
        max_pages=max_pages,
        stay_in_path=stay_in_path,
    )
    connector = WebSourceConnector(config)

    sources_dir = str(client.sources_path)
    result = connector.sync(sources_dir)

    output = f"Web crawl complete for **{client_name}**\n\n"
    output += f"- URL: {url}\n"
    output += f"- Pages added: {result.files_added}\n"
    output += f"- Pages updated: {result.files_updated}\n"
    if result.errors:
        output += f"- Errors ({len(result.errors)}):\n"
        for err in result.errors[:10]:
            output += f"  - {err}\n"
    output += f"\nCrawled pages at: `{sources_dir}/web/{config.domain}/`"
    return [TextContent(type="text", text=output)]


def handle_sync_sources(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]

    try:
        client = load_client(client_name, clients_dir=CLIENTS_DIR)
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading client '{client_name}': {e}")]

    sources_config = client.sources_config
    if not sources_config:
        return [TextContent(type="text", text=f"No sources configured for client '{client_name}'")]

    sources_dir = str(client.sources_path)
    all_results = []

    # Sync GitHub sources
    github_sources = sources_config.get("github", [])
    if github_sources:
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig

        for gh_config in github_sources:
            config = GitHubSourceConfig.from_dict(gh_config)
            connector = GitHubSourceConnector(config)
            result = connector.sync(sources_dir)
            all_results.append(("github", config.owner, result))

    # Sync web sources
    web_sources = sources_config.get("web", [])
    if web_sources:
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector

        for web_config in web_sources:
            config = WebSourceConfig.from_dict(web_config)
            connector = WebSourceConnector(config)
            result = connector.sync(sources_dir)
            all_results.append(("web", config.domain, result))

    output = f"Source sync complete for **{client_name}**\n\n"
    total_added = 0
    total_updated = 0
    total_errors = 0
    for source_type, source_id, result in all_results:
        total_added += result.files_added
        total_updated += result.files_updated
        total_errors += len(result.errors)
        status = "OK" if result.success else f"{len(result.errors)} errors"
        output += f"- [{source_type}] {source_id}: +{result.files_added} added, ~{result.files_updated} updated ({status})\n"

    output += f"\n**Totals:** {total_added} added, {total_updated} updated, {total_errors} errors"
    return [TextContent(type="text", text=output)]


def handle_list_sources(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]

    try:
        client = load_client(client_name, clients_dir=CLIENTS_DIR)
    except Exception as e:
        return [TextContent(type="text", text=f"Error loading client '{client_name}': {e}")]

    sources_config = client.sources_config
    sources_dir = client.sources_path

    output = f"Sources for **{client_name}**:\n\n"

    github_sources = sources_config.get("github", [])
    web_sources = sources_config.get("web", [])

    if not github_sources and not web_sources:
        # Check if sources dir has content even without config
        if sources_dir.exists():
            github_dir = sources_dir / "github"
            web_dir = sources_dir / "web"
            if github_dir.exists():
                repos = [d.name for d in github_dir.iterdir() if d.is_dir()]
                output += f"**GitHub** (unconfigured): {len(repos)} owner directories\n"
            if web_dir.exists():
                domains = [d.name for d in web_dir.iterdir() if d.is_dir()]
                output += f"**Web** (unconfigured): {len(domains)} domains\n"
            if not github_dir.exists() and not web_dir.exists():
                output += "No sources configured or found.\n"
        else:
            output += "No sources configured.\n"
    else:
        if github_sources:
            output += "**GitHub Sources:**\n"
            for gh in github_sources:
                owner = gh.get("owner", "?")
                repo = gh.get("repo", "")
                has_token = bool(gh.get("token_env"))
                repo_dir = sources_dir / "github" / owner
                synced = repo_dir.exists() and any(repo_dir.iterdir()) if repo_dir.exists() else False
                status = "synced" if synced else "not synced"
                desc = f"{owner}/{repo}" if repo else f"{owner} (all repos)"
                output += f"  - {desc} [{status}] {'(auth)' if has_token else '(public)'}\n"

        if web_sources:
            output += "\n**Web Sources:**\n"
            for web in web_sources:
                url = web.get("url", "?")
                mode = web.get("mode", "crawl")
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace("www.", "")
                web_dir = sources_dir / "web" / domain
                page_count = len(list(web_dir.glob("*.md"))) if web_dir.exists() else 0
                output += f"  - {url} [mode: {mode}, pages: {page_count}]\n"

    return [TextContent(type="text", text=output)]
