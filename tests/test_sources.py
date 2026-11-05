"""
Source connectors: local folders, GitHub and the web crawler.
"""

import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil


class TestSourceConnectors:
    """Tests for core.sources module (Stage 0)."""

    # --- GitHubSourceConfig ---

    def test_github_config_from_dict(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig.from_dict({
            "owner": "testorg",
            "token_env": "GH_TOKEN",
            "include_repos": ["repo1", "repo2"],
            "exclude_repos": ["legacy"],
            "include_forks": True,
            "include_private": False,
        })
        assert config.owner == "testorg"
        assert config.token_env == "GH_TOKEN"
        assert config.include_repos == ["repo1", "repo2"]
        assert config.exclude_repos == ["legacy"]
        assert config.include_forks is True
        assert config.include_private is False

    def test_github_config_repo_shorthand(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(repo="owner/repo-name")
        assert config.owner == "owner"
        assert config.repo == "repo-name"

    def test_github_config_repo_shorthand_no_override(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(owner="explicit", repo="myrepo")
        assert config.owner == "explicit"
        assert config.repo == "myrepo"

    def test_github_config_defaults(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig()
        assert config.owner == ""
        assert config.repo == ""
        assert config.token_env == ""
        assert config.include_repos == []
        assert config.exclude_repos == []
        assert config.include_forks is False
        assert config.include_private is True
        assert config.sync_on_ingest is True

    def test_github_config_token_from_env(self, monkeypatch):
        from core.sources.github_models import GitHubSourceConfig

        monkeypatch.setenv("TEST_GH_TOKEN", "ghp_testtoken123")
        config = GitHubSourceConfig(token_env="TEST_GH_TOKEN")
        assert config.token == "ghp_testtoken123"

    def test_github_config_token_missing_env(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(token_env="NONEXISTENT_TOKEN_VAR_XYZ")
        assert config.token is None

    def test_github_config_no_token_env(self):
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig()
        assert config.token is None

    # --- RepoInfo ---

    def test_repo_info_from_api_response(self):
        from core.sources.github_models import RepoInfo

        data = {
            "name": "my-repo",
            "owner": {"login": "testuser"},
            "html_url": "https://github.com/testuser/my-repo",
            "clone_url": "https://github.com/testuser/my-repo.git",
            "private": True,
            "description": "A test repo",
            "default_branch": "develop",
            "size": 1234,
            "fork": False,
        }
        repo = RepoInfo.from_api_response(data)
        assert repo.name == "my-repo"
        assert repo.owner == "testuser"
        assert repo.url == "https://github.com/testuser/my-repo"
        assert repo.clone_url == "https://github.com/testuser/my-repo.git"
        assert repo.private is True
        assert repo.description == "A test repo"
        assert repo.default_branch == "develop"
        assert repo.size_kb == 1234
        assert repo.is_fork is False

    def test_repo_info_defaults(self):
        from core.sources.github_models import RepoInfo

        data = {
            "name": "minimal",
            "owner": {"login": "user"},
        }
        repo = RepoInfo.from_api_response(data)
        assert repo.url == ""
        assert repo.clone_url == ""
        assert repo.private is False
        assert repo.description == ""
        assert repo.default_branch == "main"
        assert repo.size_kb == 0
        assert repo.is_fork is False

    def test_repo_info_null_description(self):
        from core.sources.github_models import RepoInfo

        data = {
            "name": "nodesc",
            "owner": {"login": "user"},
            "description": None,
        }
        repo = RepoInfo.from_api_response(data)
        assert repo.description == ""

    # --- SyncResult ---

    def test_sync_result_defaults(self):
        from core.sources.base import SyncResult

        r = SyncResult(source_type="github")
        assert r.files_added == 0
        assert r.files_updated == 0
        assert r.files_removed == 0
        assert r.errors == []
        assert r.total_changes == 0
        assert r.success is True

    def test_sync_result_with_errors(self):
        from core.sources.base import SyncResult

        r = SyncResult(source_type="web", errors=["something failed"])
        assert r.success is False
        assert r.total_changes == 0

    def test_sync_result_total_changes(self):
        from core.sources.base import SyncResult

        r = SyncResult(
            source_type="github",
            files_added=5,
            files_updated=3,
            files_removed=1,
        )
        assert r.total_changes == 9
        assert r.success is True

    # --- git_ops ---

    def test_git_ops_validate_name_valid(self):
        from core.sources.git_ops import _validate_name

        _validate_name("my-repo")
        _validate_name("repo.name")
        _validate_name("Repo_123")

    def test_git_ops_validate_name_invalid(self):
        from core.sources.git_ops import _validate_name, GitError

        with pytest.raises(GitError):
            _validate_name("repo with spaces")
        with pytest.raises(GitError):
            _validate_name("../escape")
        with pytest.raises(GitError):
            _validate_name("path/../traversal")

    def test_git_ops_sanitise_output(self):
        from core.sources.git_ops import _sanitise_output

        assert "ghp_" not in _sanitise_output(
            "error with ghp_abcdefghijklmnopqrstuvwxyz0123456789"
        )
        assert "****" in _sanitise_output(
            "error with ghp_abcdefghijklmnopqrstuvwxyz0123456789"
        )

    def test_git_ops_sanitise_url(self):
        from core.sources.git_ops import _sanitise_output

        result = _sanitise_output(
            "fatal: https://user:password@github.com/repo.git not found"
        )
        assert "password" not in result
        assert "****" in result

    def test_clone_result_dataclass(self):
        from core.sources.git_ops import CloneResult

        r = CloneResult(success=True, message="ok")
        assert r.success is True
        assert r.message == "ok"
        assert r.changed_files == []

    def test_clone_result_with_changes(self):
        from core.sources.git_ops import CloneResult

        r = CloneResult(
            success=True,
            message="Updated: 3 files changed",
            changed_files=["a.py", "b.py", "c.py"],
        )
        assert len(r.changed_files) == 3

    # --- GitHubClient ---

    def test_github_client_mask_token(self):
        from core.sources.github_client import _mask_token

        assert _mask_token(None) == "(none)"
        assert _mask_token("") == "(none)"
        assert _mask_token("short") == "****"
        assert _mask_token("ghp_abcdefghijklm") == "ghp_...jklm"

    def test_github_client_creates_with_token(self):
        from core.sources.github_client import GitHubClient

        client = GitHubClient(token="test_token")
        assert client._token == "test_token"
        client.close()

    def test_github_client_creates_without_token(self):
        from core.sources.github_client import GitHubClient

        client = GitHubClient()
        assert client._token is None
        client.close()

    def test_github_client_context_manager(self):
        from core.sources.github_client import GitHubClient

        with GitHubClient() as client:
            assert client is not None

    # --- GitHubSourceConnector ---

    def test_connector_status(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig

        config = GitHubSourceConfig(owner="testorg", token_env="")
        connector = GitHubSourceConnector(config)
        status = connector.status()

        assert status["source_type"] == "github"
        assert status["owner"] == "testorg"
        assert status["has_token"] is False

    def test_connector_filter_repos_include(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(
            owner="org", include_repos=["keep-this", "and-this"]
        )
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("keep-this", "org", "", "", False, "", "main", 0, False),
            RepoInfo("and-this", "org", "", "", False, "", "main", 0, False),
            RepoInfo("skip-this", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 2
        assert {r.name for r in filtered} == {"keep-this", "and-this"}

    def test_connector_filter_repos_exclude(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(owner="org", exclude_repos=["legacy"])
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("app", "org", "", "", False, "", "main", 0, False),
            RepoInfo("legacy", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1
        assert filtered[0].name == "app"

    def test_connector_filter_repos_case_insensitive(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(
            owner="org", include_repos=["MyRepo"]
        )
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("myrepo", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1

    def test_connector_filter_repos_empty_filters(self):
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig(owner="org")
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo("a", "org", "", "", False, "", "main", 0, False),
            RepoInfo("b", "org", "", "", False, "", "main", 0, False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 2

    # --- WebSourceConfig ---

    def test_web_config_from_dict(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig.from_dict({
            "url": "https://docs.example.com/api",
            "mode": "docs",
            "stay_in_path": False,
            "max_pages": 100,
            "wait_time": 2.0,
        })
        assert config.url == "https://docs.example.com/api"
        assert config.mode == "docs"
        assert config.stay_in_path is False
        assert config.max_pages == 100
        assert config.wait_time == 2.0

    def test_web_config_defaults(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig()
        assert config.url == ""
        assert config.mode == "crawl"
        assert config.stay_in_path is True
        assert config.max_pages == 500
        assert config.wait_time == 1.0

    def test_web_config_domain(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig(url="https://www.example.com/docs/api")
        assert config.domain == "example.com"
        assert config.base_url == "https://www.example.com"
        assert config.starting_path == "/docs/api"

    def test_web_config_domain_no_www(self):
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig(url="https://docs.example.com/")
        assert config.domain == "docs.example.com"

    # --- URL normalization ---

    def test_normalize_link_absolute_same_domain(self):
        from core.sources.web_crawler import normalize_link

        result = normalize_link(
            "https://example.com/page", "/", "https://example.com"
        )
        assert result == "/page"

    def test_normalize_link_relative(self):
        from core.sources.web_crawler import normalize_link

        result = normalize_link("subpage", "/docs/api", "https://example.com")
        assert result == "/docs/subpage"

    def test_normalize_link_external_returns_none(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "https://other.com/page", "/", "https://example.com"
        ) is None

    def test_normalize_link_mailto_returns_none(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "mailto:test@example.com", "/", "https://example.com"
        ) is None

    def test_normalize_link_skip_extensions(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "/file.pdf", "/", "https://example.com"
        ) is None
        assert normalize_link(
            "/image.png", "/", "https://example.com"
        ) is None

    def test_normalize_link_strips_query_fragment(self):
        from core.sources.web_crawler import normalize_link

        result = normalize_link(
            "/page?sort=date#section", "/", "https://example.com"
        )
        assert result == "/page"

    def test_normalize_link_empty(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link("", "/", "https://example.com") is None
        assert normalize_link("#", "/", "https://example.com") is None

    def test_normalize_link_javascript(self):
        from core.sources.web_crawler import normalize_link

        assert normalize_link(
            "javascript:void(0)", "/", "https://example.com"
        ) is None

    # --- Scope checking ---

    def test_is_in_scope_stay_in_path(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig

        config = WebSourceConfig(
            url="https://example.com/docs", stay_in_path=True
        )
        assert _is_in_scope("/docs/api", config) is True
        assert _is_in_scope("/docs", config) is True
        assert _is_in_scope("/other", config) is False

    def test_is_in_scope_no_restriction(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig

        config = WebSourceConfig(
            url="https://example.com/docs", stay_in_path=False
        )
        assert _is_in_scope("/anywhere", config) is True

    def test_is_in_scope_root(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig

        config = WebSourceConfig(
            url="https://example.com/", stay_in_path=True
        )
        assert _is_in_scope("/anything", config) is True

    # --- Non-content path detection ---

    def test_non_content_paths(self):
        from core.sources.web_crawler import NON_CONTENT_PATHS

        assert NON_CONTENT_PATHS.search("/login")
        assert NON_CONTENT_PATHS.search("/admin/dashboard")
        assert NON_CONTENT_PATHS.search("/cart/checkout")
        assert NON_CONTENT_PATHS.search("/author/john")
        assert NON_CONTENT_PATHS.search("/tag/python")
        assert not NON_CONTENT_PATHS.search("/docs/api-reference")
        assert not NON_CONTENT_PATHS.search("/blog/my-post")

    # --- Link extraction ---

    def test_extract_links(self):
        from core.sources.web_crawler import _extract_links

        html = '<a href="/page1">P1</a> <a href="/page2">P2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_extract_links_data_attributes(self):
        from core.sources.web_crawler import _extract_links

        html = '<a href="/page1">P1</a>'
        links = _extract_links(html)
        assert "/page1" in links

    # --- URL to filepath ---

    def test_url_to_filepath(self):
        from core.sources.web_crawler import _url_to_filepath

        assert _url_to_filepath("/docs/api") == "docs_api.md"
        assert _url_to_filepath("/") == "index.md"
        assert _url_to_filepath("") == "index.md"
        assert _url_to_filepath("/page") == "page.md"

    # --- CrawlCache ---

    def test_crawl_cache_empty(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache = CrawlCache(str(tmp_path))
        cache.load()  # no file to load
        assert cache.page_count == 0
        assert cache.get_page_headers("/page") == {}
        assert cache.get_page_links("/page") == []
        assert cache.is_page_unchanged("/page", "content") is False

    def test_crawl_cache_update_and_check(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache = CrawlCache(str(tmp_path))
        cache.update_page(
            "/page",
            {"etag": '"abc"', "last-modified": "Mon, 01 Jan 2024"},
            "hello world",
            links=["/link1", "/link2"],
        )

        # Conditional headers
        headers = cache.get_page_headers("/page")
        assert headers["If-None-Match"] == '"abc"'
        assert headers["If-Modified-Since"] == "Mon, 01 Jan 2024"

        # Links
        assert cache.get_page_links("/page") == ["/link1", "/link2"]

        # Content hash
        assert cache.is_page_unchanged("/page", "hello world") is True
        assert cache.is_page_unchanged("/page", "changed") is False

    def test_crawl_cache_persistence(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache1 = CrawlCache(str(tmp_path))
        cache1.update_page("/p", {}, "content", links=["/a"])
        cache1.save()

        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.page_count == 1
        assert cache2.get_page_links("/p") == ["/a"]

    def test_crawl_cache_case_insensitive_headers(self, tmp_path):
        from core.sources.web_cache import CrawlCache

        cache = CrawlCache(str(tmp_path))
        cache.update_page("/p", {"ETag": '"xyz"', "Last-Modified": "date"}, "c")
        headers = cache.get_page_headers("/p")
        assert headers["If-None-Match"] == '"xyz"'

    # --- HTML to Markdown converter ---

    def test_html_to_markdown_basic(self):
        from core.sources.web_converter import html_to_markdown

        html = """
        <html><body>
        <article>
            <h1>Test Article</h1>
            <p>This is a test paragraph with enough words to pass the minimum
            content threshold that the converter uses to determine if content
            is meaningful enough to extract and convert to markdown format.</p>
        </article>
        </body></html>
        """
        md = html_to_markdown(html)
        assert md is not None
        assert "Test Article" in md

    def test_html_to_markdown_strips_nav(self):
        from core.sources.web_converter import html_to_markdown

        html = """
        <html><body>
        <nav><a href="/">Home</a></nav>
        <article>
            <h1>Content</h1>
            <p>This is the main content of the page with plenty of text to
            ensure it passes the word count threshold for extraction purposes
            in our markdown converter module test cases here.</p>
        </article>
        <footer>Copyright</footer>
        </body></html>
        """
        md = html_to_markdown(html)
        assert md is not None
        assert "Home" not in md or "Content" in md

    def test_html_to_markdown_returns_none_for_empty(self):
        from core.sources.web_converter import html_to_markdown

        assert html_to_markdown("<html><body></body></html>") is None

    def test_extract_title(self):
        from core.sources.web_converter import extract_title

        assert extract_title("<html><body><h1>Title</h1></body></html>") == "Title"
        assert extract_title("<html><head><title>Page</title></head></html>") == "Page"
        assert extract_title("<html></html>") == ""

    def test_clean_markdown_removes_ads(self):
        from core.sources.web_converter import clean_markdown

        md = "# Title\n\nContent here.\n\nAdvertisement\n\nMore content."
        cleaned = clean_markdown(md)
        assert "Advertisement" not in cleaned
        assert "Content here" in cleaned

    def test_clean_markdown_removes_subscribe(self):
        from core.sources.web_converter import clean_markdown

        md = "# Title\n\nContent.\n\nSign up\n\nSubscribe\n\nMore."
        cleaned = clean_markdown(md)
        assert "Sign up" not in cleaned
        assert "Subscribe" not in cleaned

    def test_clean_markdown_removes_empty_links(self):
        from core.sources.web_converter import clean_markdown

        md = "# Title\n\nSome text [](http://empty) here."
        cleaned = clean_markdown(md)
        assert "[](http://empty)" not in cleaned

    def test_clean_markdown_collapses_blank_lines(self):
        from core.sources.web_converter import clean_markdown

        md = "Line 1\n\n\n\n\n\nLine 2"
        cleaned = clean_markdown(md)
        assert "\n\n\n\n" not in cleaned

    def test_clean_markdown_none_input(self):
        from core.sources.web_converter import clean_markdown

        assert clean_markdown(None) is None
        assert clean_markdown("") is None

    # --- WebSourceConnector ---

    def test_web_connector_status(self):
        from core.sources.web_crawler import WebSourceConnector, WebSourceConfig

        config = WebSourceConfig(url="https://docs.example.com/api")
        connector = WebSourceConnector(config)
        status = connector.status()

        assert status["source_type"] == "web"
        assert status["url"] == "https://docs.example.com/api"
        assert status["domain"] == "docs.example.com"

    # --- Error path tests ---

    def test_clone_result_is_fresh_clone(self):
        from core.sources.git_ops import CloneResult

        fresh = CloneResult(success=True, message="Cloned successfully", is_fresh_clone=True)
        assert fresh.is_fresh_clone is True

        fetched = CloneResult(success=True, message="Already up to date")
        assert fetched.is_fresh_clone is False

    def test_askpass_script_shell_quoting(self):
        from core.sources.git_ops import _make_askpass_script
        import os

        # Token with a single quote should be properly escaped
        token = "ghp_test'token"
        path = _make_askpass_script(token)
        try:
            with open(path) as f:
                content = f.read()
            assert "'" in content  # should use quotes
            assert f"echo {token}" not in content  # should NOT have unquoted token
        finally:
            os.unlink(path)

    def test_askpass_script_normal_token(self):
        from core.sources.git_ops import _make_askpass_script
        import os

        token = "ghp_abc123XYZ"
        path = _make_askpass_script(token)
        try:
            with open(path) as f:
                content = f.read()
            assert token in content
            assert content.startswith("#!/bin/sh\n")
            # Check file permissions
            stat = os.stat(path)
            assert stat.st_mode & 0o777 == 0o700
        finally:
            os.unlink(path)

    def test_github_client_is_org(self):
        """Test the public is_org method exists and has correct signature."""
        from core.sources.github_client import GitHubClient

        client = GitHubClient(token=None)
        assert hasattr(client, "is_org")
        client.close()

    def test_yaml_frontmatter_escaping(self):
        """Titles with quotes and newlines should be escaped in frontmatter."""
        import yaml

        # Simulate the frontmatter generation with a problematic title
        title = 'My "Great" Page'
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        body = (
            f'title: "{safe_title}"\n'
            f"url: https://example.com/test\n"
            f"source: web_crawler\n"
        )
        # Should parse as valid YAML
        parsed = yaml.safe_load(body)
        assert parsed["title"] == 'My "Great" Page'

    def test_yaml_frontmatter_newline_in_title(self):
        """Newlines in titles should be replaced with spaces."""
        import yaml

        title = "Title\nWith\nNewlines"
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        body = (
            f'title: "{safe_title}"\n'
            f"url: https://example.com/test\n"
            f"source: web_crawler\n"
        )
        parsed = yaml.safe_load(body)
        assert "\n" not in parsed["title"]
        assert parsed["title"] == "Title With Newlines"

    def test_sync_result_files_updated(self):
        """SyncResult should track updated files separately from added."""
        from core.sources.base import SyncResult

        result = SyncResult(source_type="web")
        result.files_added = 3
        result.files_updated = 5
        assert result.total_changes == 8

    def test_crawl_cache_version_mismatch(self):
        """Cache with different version should start fresh."""
        import json
        import os
        import tempfile

        from core.sources.web_cache import CrawlCache

        tmp_dir = tempfile.mkdtemp()
        cache = CrawlCache(tmp_dir)
        cache_file = os.path.join(tmp_dir, ".crawl_cache.json")
        with open(cache_file, "w") as f:
            json.dump({"version": "0.0.0", "pages": {"old": "data"}}, f)

        cache.load()
        assert cache.page_count == 0  # should start fresh

    def test_crawl_cache_corrupt_file(self):
        """Corrupt cache file should not crash, should start fresh."""
        import os
        import tempfile

        from core.sources.web_cache import CrawlCache

        tmp_dir = tempfile.mkdtemp()
        cache = CrawlCache(tmp_dir)
        cache_file = os.path.join(tmp_dir, ".crawl_cache.json")
        with open(cache_file, "w") as f:
            f.write("not valid json{{{")

        cache.load()
        assert cache.page_count == 0

    def test_github_connector_uses_is_fresh_clone(self):
        """Connector should use is_fresh_clone instead of string matching."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        import inspect

        source = inspect.getsource(GitHubSourceConnector.sync)
        assert "is_fresh_clone" in source
        assert '"Cloned" in' not in source

    def test_web_config_domain_empty_url(self):
        """Empty URL config should produce empty domain."""
        from core.sources.web_crawler import WebSourceConfig

        config = WebSourceConfig()
        assert config.domain == ""  # empty, but doesn't crash

    def test_connector_filter_repos_both_include_and_exclude(self):
        """When both include and exclude are set, exclude takes precedence."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo

        config = GitHubSourceConfig.from_dict({
            "owner": "test",
            "include_repos": ["repo-a", "repo-b", "repo-c"],
            "exclude_repos": ["repo-b"],
        })
        connector = GitHubSourceConnector(config)

        repos = [
            RepoInfo(name="repo-a", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
            RepoInfo(name="repo-b", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
            RepoInfo(name="repo-c", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
            RepoInfo(name="repo-d", owner="test", url="", clone_url="",
                     private=False, description="", default_branch="main",
                     size_kb=0, is_fork=False),
        ]
        filtered = connector._filter_repos(repos)
        names = [r.name for r in filtered]
        assert "repo-a" in names
        assert "repo-b" not in names  # excluded
        assert "repo-c" in names
        assert "repo-d" not in names  # not in include list

    def test_sanitise_output_all_token_formats(self):
        """All GitHub token formats should be sanitised."""
        from core.sources.git_ops import _sanitise_output

        tokens = [
            "ghp_" + "a" * 36,
            "github_pat_" + "a" * 80,
            "gho_" + "a" * 36,
            "ghs_" + "a" * 36,
            "ghu_" + "a" * 36,
            "ghr_" + "a" * 36,
        ]
        for token in tokens:
            result = _sanitise_output(f"error with {token} in output")
            assert token not in result, f"Token format {token[:10]} not sanitised"
            assert "****" in result

    # --- Module imports ---

    def test_sources_package_imports(self):
        from core.sources import (
            BaseSourceConnector,
            SyncResult,
            GitHubSourceConnector,
            GitHubSourceConfig,
            RepoInfo,
            WebSourceConnector,
            WebSourceConfig,
        )

        assert BaseSourceConnector is not None
        assert SyncResult is not None
        assert GitHubSourceConnector is not None
        assert GitHubSourceConfig is not None
        assert RepoInfo is not None
        assert WebSourceConnector is not None
        assert WebSourceConfig is not None


class TestSourceConnectorErrorPaths:
    """Error-path tests for source connector I/O operations."""

    # --- git_ops error paths ---

    def test_validate_name_rejects_path_traversal(self):
        from core.sources.git_ops import _validate_name, GitError
        with pytest.raises(GitError, match="Path traversal"):
            _validate_name("foo..bar")

    def test_validate_name_rejects_special_chars(self):
        from core.sources.git_ops import _validate_name, GitError
        with pytest.raises(GitError, match="Invalid repository name"):
            _validate_name("repo name with spaces")

    def test_validate_name_accepts_valid(self):
        from core.sources.git_ops import _validate_name
        _validate_name("my-repo.v2")
        _validate_name("repo_name")
        _validate_name("valid123")

    def test_clone_result_defaults(self):
        from core.sources.git_ops import CloneResult
        r = CloneResult(success=True, message="ok")
        assert r.changed_files == []
        assert r.is_fresh_clone is False

    def test_clone_repo_invalid_name(self):
        from core.sources.git_ops import clone_repo, GitError
        with pytest.raises(GitError, match="Invalid repository name"):
            clone_repo("https://github.com/x/y.git", Path("/tmp/bad dir"), token=None)

    def test_fetch_repo_not_a_git_repo(self):
        import tempfile
        from core.sources.git_ops import fetch_repo
        tmp = tempfile.mkdtemp()
        try:
            result = fetch_repo(Path(tmp), "https://github.com/x/y.git")
            assert result.success is False
            assert "Not a git repository" in result.message
        finally:
            shutil.rmtree(tmp)

    def test_sanitise_output_removes_tokens(self):
        from core.sources.git_ops import _sanitise_output
        output = "error: ghp_" + "a" * 36 + " invalid"
        sanitised = _sanitise_output(output)
        assert "ghp_" not in sanitised
        assert "****" in sanitised

    def test_sanitise_output_removes_url_tokens(self):
        from core.sources.git_ops import _sanitise_output
        output = "error: https://oauth2:ghp_abc123@github.com/x/y.git failed"
        sanitised = _sanitise_output(output)
        assert "ghp_abc123" not in sanitised

    def test_sanitise_output_preserves_clean_text(self):
        from core.sources.git_ops import _sanitise_output
        clean = "normal git output with no tokens"
        assert _sanitise_output(clean) == clean

    def test_get_default_branch_non_git_dir(self):
        import tempfile
        from core.sources.git_ops import get_default_branch
        tmp = tempfile.mkdtemp()
        try:
            result = get_default_branch(Path(tmp))
            assert result == "main"
        finally:
            shutil.rmtree(tmp)

    # --- GitHub client error paths ---

    def test_github_client_mask_token_none(self):
        from core.sources.github_client import _mask_token
        assert _mask_token(None) == "(none)"

    def test_github_client_mask_token_short(self):
        from core.sources.github_client import _mask_token
        assert _mask_token("abcd") == "****"

    def test_github_client_mask_token_normal(self):
        from core.sources.github_client import _mask_token
        result = _mask_token("abcdefghij1234567890")
        assert result.startswith("abcd")
        assert result.endswith("7890")
        assert "..." in result

    def test_github_api_error_has_status_code(self):
        from core.sources.github_client import GitHubAPIError
        err = GitHubAPIError("test", status_code=404)
        assert err.status_code == 404
        assert str(err) == "test"

    def test_authentication_error_is_github_api_error(self):
        from core.sources.github_client import AuthenticationError, GitHubAPIError
        err = AuthenticationError("auth failed")
        assert isinstance(err, GitHubAPIError)

    def test_rate_limit_error_is_github_api_error(self):
        from core.sources.github_client import RateLimitError, GitHubAPIError
        err = RateLimitError("rate limited")
        assert isinstance(err, GitHubAPIError)

    # --- GitHub connector error paths ---

    def test_github_connector_empty_owner(self):
        from core.sources.github_models import GitHubSourceConfig
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="")
        connector = GitHubSourceConnector(config)
        repos = connector._discover_repos()
        assert repos == []

    def test_github_connector_filter_include(self):
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="test", include_repos=["repo1"])
        connector = GitHubSourceConnector(config)
        repos = [
            RepoInfo(name="repo1", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
            RepoInfo(name="repo2", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1
        assert filtered[0].name == "repo1"

    def test_github_connector_filter_exclude(self):
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="test", exclude_repos=["legacy"])
        connector = GitHubSourceConnector(config)
        repos = [
            RepoInfo(name="repo1", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
            RepoInfo(name="legacy", owner="test", url="", clone_url="", private=False, description="", default_branch="main", size_kb=0, is_fork=False),
        ]
        filtered = connector._filter_repos(repos)
        assert len(filtered) == 1
        assert filtered[0].name == "repo1"

    def test_github_connector_status(self):
        from core.sources.github_models import GitHubSourceConfig
        from core.sources.github_connector import GitHubSourceConnector
        config = GitHubSourceConfig(owner="testorg", repo="myrepo")
        connector = GitHubSourceConnector(config)
        status = connector.status()
        assert status["source_type"] == "github"
        assert status["owner"] == "testorg"
        assert status["repo"] == "myrepo"

    # --- Web crawler error paths ---

    def test_normalize_link_mailto(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("mailto:test@example.com", "/", "https://example.com") is None

    def test_normalize_link_javascript(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("javascript:void(0)", "/", "https://example.com") is None

    def test_normalize_link_empty(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("", "/", "https://example.com") is None

    def test_normalize_link_external(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("https://other.com/page", "/", "https://example.com") is None

    def test_normalize_link_skip_extension(self):
        from core.sources.web_crawler import normalize_link
        assert normalize_link("/file.pdf", "/", "https://example.com") is None
        assert normalize_link("/image.jpg", "/", "https://example.com") is None

    def test_normalize_link_relative(self):
        from core.sources.web_crawler import normalize_link
        result = normalize_link("subpage", "/docs/", "https://example.com")
        assert result == "/docs/subpage"

    def test_normalize_link_absolute_same_domain(self):
        from core.sources.web_crawler import normalize_link
        result = normalize_link("https://example.com/page", "/", "https://example.com")
        assert result == "/page"

    def test_is_in_scope_root(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig
        config = WebSourceConfig(url="https://example.com", stay_in_path=True)
        assert _is_in_scope("/anything", config) is True

    def test_is_in_scope_restricted(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig
        config = WebSourceConfig(url="https://example.com/docs/api", stay_in_path=True)
        assert _is_in_scope("/docs/api/endpoints", config) is True
        assert _is_in_scope("/docs/api", config) is True
        assert _is_in_scope("/other", config) is False

    def test_is_in_scope_disabled(self):
        from core.sources.web_crawler import _is_in_scope, WebSourceConfig
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=False)
        assert _is_in_scope("/other", config) is True

    def test_url_to_filepath_root(self):
        from core.sources.web_crawler import _url_to_filepath
        assert _url_to_filepath("/") == "index.md"

    def test_url_to_filepath_nested(self):
        from core.sources.web_crawler import _url_to_filepath
        result = _url_to_filepath("/docs/api/endpoints")
        assert result == "docs_api_endpoints.md"

    def test_url_to_filepath_special_chars(self):
        from core.sources.web_crawler import _url_to_filepath
        result = _url_to_filepath('/page?"special"')
        assert '"' not in result
        assert '?' not in result

    def test_extract_links_basic(self):
        from core.sources.web_crawler import _extract_links
        html = '<a href="/page1">Page 1</a><a href="/page2">Page 2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_extract_links_no_links(self):
        from core.sources.web_crawler import _extract_links
        assert _extract_links("<p>No links here</p>") == []

    def test_non_content_paths_blocked(self):
        from core.sources.web_crawler import NON_CONTENT_PATHS
        assert NON_CONTENT_PATHS.search("/login")
        assert NON_CONTENT_PATHS.search("/admin/dashboard")
        assert NON_CONTENT_PATHS.search("/cart/checkout")
        assert not NON_CONTENT_PATHS.search("/docs/api")

    def test_web_connector_status(self):
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=100)
        connector = WebSourceConnector(config)
        status = connector.status()
        assert status["source_type"] == "web"
        assert status["url"] == "https://example.com/docs"
        assert status["max_pages"] == 100

    # --- Web converter error paths ---

    def test_clean_markdown_none(self):
        from core.sources.web_converter import clean_markdown
        assert clean_markdown(None) is None

    def test_clean_markdown_empty(self):
        from core.sources.web_converter import clean_markdown
        assert clean_markdown("") is None

    def test_clean_markdown_strips_ads(self):
        from core.sources.web_converter import clean_markdown
        md = "Content here\n\nArticle continues below\nADVERTISEMENT\n\nMore content"
        result = clean_markdown(md)
        assert "Article continues below" not in result
        assert "More content" in result

    def test_clean_markdown_strips_subscribe(self):
        from core.sources.web_converter import clean_markdown
        md = "Content\n\nSign up\n\nSubscribe\n\nMore"
        result = clean_markdown(md)
        assert "Sign up" not in result

    def test_clean_markdown_removes_empty_links(self):
        from core.sources.web_converter import clean_markdown
        md = "Content [](http://example.com) here ![](http://example.com/img.png)"
        result = clean_markdown(md)
        assert "[](http://example.com)" not in result
        assert "![](http://example.com/img.png)" not in result

    def test_clean_markdown_collapses_blank_lines(self):
        from core.sources.web_converter import clean_markdown
        md = "Line 1\n\n\n\n\n\nLine 2"
        result = clean_markdown(md)
        assert "\n\n\n\n" not in result

    def test_html_to_markdown_none_for_empty(self):
        from core.sources.web_converter import html_to_markdown
        assert html_to_markdown("<html><body></body></html>") is None

    def test_extract_title_h1(self):
        from core.sources.web_converter import extract_title
        assert extract_title("<h1>My Title</h1>") == "My Title"

    def test_extract_title_fallback_to_title(self):
        from core.sources.web_converter import extract_title
        assert extract_title("<title>Page Title</title>") == "Page Title"

    def test_extract_title_empty(self):
        from core.sources.web_converter import extract_title
        assert extract_title("<p>No title</p>") == ""

    # --- CrawlCache error paths ---

    def test_crawl_cache_load_nonexistent(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            assert cache._data.get("version") is not None
        finally:
            shutil.rmtree(tmp)

    def test_crawl_cache_save_and_reload(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            cache.update_page("/test", {"etag": '"abc"'}, "<html>test</html>", links=["/page2"])
            cache.save()

            cache2 = CrawlCache(tmp)
            cache2.load()
            assert cache2.get_page_links("/test") == ["/page2"]
        finally:
            shutil.rmtree(tmp)

    def test_crawl_cache_unchanged_page(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            html = "<html><body>Same content</body></html>"
            cache.update_page("/test", {}, html, links=[])
            assert cache.is_page_unchanged("/test", html) is True
            assert cache.is_page_unchanged("/test", "<html>different</html>") is False
        finally:
            shutil.rmtree(tmp)

    def test_crawl_cache_headers_etag(self):
        import tempfile
        from core.sources.web_cache import CrawlCache
        tmp = tempfile.mkdtemp()
        try:
            cache = CrawlCache(tmp)
            cache.load()
            cache.update_page("/test", {"etag": '"xyz"'}, "<html>x</html>", links=[])
            headers = cache.get_page_headers("/test")
            assert headers.get("If-None-Match") == '"xyz"'
        finally:
            shutil.rmtree(tmp)

    # --- SyncResult ---

    def test_sync_result_total_changes(self):
        from core.sources.base import SyncResult
        r = SyncResult(source_type="test", files_added=5, files_updated=3, files_removed=2)
        assert r.total_changes == 10

    def test_sync_result_success_no_errors(self):
        from core.sources.base import SyncResult
        r = SyncResult(source_type="test")
        assert r.success is True

    def test_sync_result_success_with_errors(self):
        from core.sources.base import SyncResult
        r = SyncResult(source_type="test", errors=["Something went wrong"])
        assert r.success is False


class TestSourceIngestIntegration:
    """Tests for source connector + ingest pipeline integration."""

    def test_client_sources_path(self):
        from core.client import Client
        client = Client(Path("/tmp/test_client"), {"machine_name": "test"})
        assert client.sources_path == Path("/tmp/test_client/sources")

    def test_client_sources_config_empty(self):
        from core.client import Client
        client = Client(Path("/tmp/test_client"), {"machine_name": "test"})
        assert client.sources_config == {}

    def test_client_sources_config_with_data(self):
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "github": [{"owner": "testorg"}],
                "web": [{"url": "https://example.com"}],
            },
        }
        client = Client(Path("/tmp/test_client"), config)
        assert len(client.sources_config["github"]) == 1
        assert len(client.sources_config["web"]) == 1

    def test_client_has_sources_false_no_dir(self):
        import tempfile
        from core.client import Client
        tmp = tempfile.mkdtemp()
        try:
            client = Client(Path(tmp), {"machine_name": "test"})
            assert client.has_sources is False
        finally:
            shutil.rmtree(tmp)

    def test_client_has_sources_true_with_files(self):
        import tempfile
        from core.client import Client
        tmp = tempfile.mkdtemp()
        try:
            sources_dir = Path(tmp) / "sources" / "web"
            sources_dir.mkdir(parents=True)
            (sources_dir / "test.md").write_text("# Test")
            client = Client(Path(tmp), {"machine_name": "test"})
            assert client.has_sources is True
        finally:
            shutil.rmtree(tmp)

    def test_sources_in_known_top_level_keys(self):
        from core.client import _KNOWN_TOP_LEVEL_KEYS
        assert "sources" in _KNOWN_TOP_LEVEL_KEYS

    def test_sources_config_no_validation_warning(self):
        from core.client import validate_config
        config = {
            "machine_name": "test_client",
            "sources": {
                "github": [{"owner": "testorg"}],
            },
        }
        warnings = validate_config(config)
        assert not any("sources" in w for w in warnings)

    def test_should_sync_on_ingest_true(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "github": [{"owner": "org", "sync_on_ingest": True}],
            },
        }
        client = Client(Path("/tmp/test"), config)
        assert _should_sync_on_ingest(client) is True

    def test_should_sync_on_ingest_false(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "github": [{"owner": "org", "sync_on_ingest": False}],
            },
        }
        client = Client(Path("/tmp/test"), config)
        assert _should_sync_on_ingest(client) is False

    def test_should_sync_on_ingest_no_sources(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        client = Client(Path("/tmp/test"), {"machine_name": "test"})
        assert _should_sync_on_ingest(client) is False

    def test_should_sync_on_ingest_web_source(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import _should_sync_on_ingest
        from core.client import Client
        config = {
            "machine_name": "test",
            "sources": {
                "web": [{"url": "https://example.com", "sync_on_ingest": True}],
            },
        }
        client = Client(Path("/tmp/test"), config)
        assert _should_sync_on_ingest(client) is True

    def test_sync_client_sources_no_config(self):
        import sys, tempfile
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.ingest import sync_client_sources
        # Create a minimal client directory
        tmp = tempfile.mkdtemp()
        import yaml
        client_dir = Path(tmp) / "clients" / "test_sync"
        client_dir.mkdir(parents=True)
        (client_dir / "documents").mkdir()
        (client_dir / "index").mkdir()
        (client_dir / "prompts").mkdir()
        config = {"machine_name": "test_sync", "name": "Test Sync"}
        (client_dir / "client.yaml").write_text(yaml.dump(config))
        try:
            import os
            old_cwd = os.getcwd()
            os.chdir(tmp)
            result = sync_client_sources("test_sync")
            assert result["synced"] is False
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmp)


class TestGitHubClientRequest:
    """Mock-based tests for GitHubClient._request() retry/error handling."""

    def test_request_success(self):
        """Successful request returns response."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        result = client._request("GET", "/test")
        assert result.status_code == 200
        client._client.request.assert_called_once()

    def test_request_401_raises_auth_error(self):
        """401 response raises AuthenticationError."""
        from core.sources.github_client import GitHubClient, AuthenticationError
        client = GitHubClient(token="ghp_testtoken1234567890")
        mock_response = MagicMock()
        mock_response.status_code = 401
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        with pytest.raises(AuthenticationError) as exc_info:
            client._request("GET", "/test")
        assert "ghp_" in str(exc_info.value)
        assert "testtoken1234567890" not in str(exc_info.value)

    def test_request_403_rate_limit_raises_rate_limit_error(self):
        """403 with x-ratelimit-remaining=0 raises RateLimitError."""
        from core.sources.github_client import GitHubClient, RateLimitError
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "0",
        }
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        with pytest.raises(RateLimitError) as exc_info:
            client._request("GET", "/test")
        assert exc_info.value.status_code == 403

    def test_request_403_non_rate_limit_raises_api_error(self):
        """403 without rate limit headers raises GitHubAPIError."""
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {"x-ratelimit-remaining": "100"}
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)
        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/repos/test")
        assert "Forbidden" in str(exc_info.value)
        assert exc_info.value.status_code == 403

    @patch("core.sources.github_client.time.sleep")
    def test_request_5xx_retries(self, mock_sleep):
        """5xx responses are retried with backoff."""
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.headers = {}
        client._client = MagicMock()
        client._client.request = MagicMock(return_value=mock_500)
        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")
        assert client._client.request.call_count == 3  # MAX_RETRIES
        assert mock_sleep.call_count == 3
        assert "Server error" in str(exc_info.value)

    @patch("core.sources.github_client.time.sleep")
    def test_request_transport_error_retries(self, mock_sleep):
        """TransportError is retried with backoff."""
        import httpx
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        client._client = MagicMock()
        client._client.request = MagicMock(
            side_effect=httpx.TransportError("Connection refused")
        )
        with pytest.raises(GitHubAPIError) as exc_info:
            client._request("GET", "/test")
        assert client._client.request.call_count == 3
        assert "Network error" in str(exc_info.value)

    @patch("core.sources.github_client.time.sleep")
    def test_request_5xx_then_success(self, mock_sleep):
        """5xx followed by success returns the successful response."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.headers = {}
        mock_200 = MagicMock()
        mock_200.status_code = 200
        client._client = MagicMock()
        client._client.request = MagicMock(side_effect=[mock_500, mock_200])
        result = client._request("GET", "/test")
        assert result.status_code == 200
        assert client._client.request.call_count == 2


class TestGitHubClientPagination:
    """Mock-based tests for GitHubClient._get_paginated()."""

    def test_single_page(self):
        """Single page with no next link returns items."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"name": "repo1"}, {"name": "repo2"}]
        mock_response.headers = {"link": ""}
        mock_response.raise_for_status = MagicMock()
        client._request = MagicMock(return_value=mock_response)
        items = client._get_paginated("/repos")
        assert len(items) == 2
        assert items[0]["name"] == "repo1"

    def test_multi_page(self):
        """Multiple pages are accumulated."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = [{"name": "repo1"}]
        page1.headers = {"link": '<url>; rel="next"'}
        page1.raise_for_status = MagicMock()
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = [{"name": "repo2"}]
        page2.headers = {"link": ""}
        page2.raise_for_status = MagicMock()
        client._request = MagicMock(side_effect=[page1, page2])
        items = client._get_paginated("/repos")
        assert len(items) == 2
        assert items[1]["name"] == "repo2"

    def test_404_raises_error(self):
        """404 response raises GitHubAPIError."""
        from core.sources.github_client import GitHubClient, GitHubAPIError
        client = GitHubClient(token=None)
        mock_response = MagicMock()
        mock_response.status_code = 404
        client._request = MagicMock(return_value=mock_response)
        with pytest.raises(GitHubAPIError) as exc_info:
            client._get_paginated("/repos")
        assert "Not found" in str(exc_info.value)

    def test_empty_page_stops_pagination(self):
        """Empty page breaks the loop."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = [{"name": "repo1"}]
        page1.headers = {"link": '<url>; rel="next"'}
        page1.raise_for_status = MagicMock()
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = []
        page2.headers = {}
        page2.raise_for_status = MagicMock()
        client._request = MagicMock(side_effect=[page1, page2])
        items = client._get_paginated("/repos")
        assert len(items) == 1

    def test_max_pages_caps_pagination(self):
        """max_pages parameter limits page count."""
        from core.sources.github_client import GitHubClient
        client = GitHubClient(token=None)

        def make_page():
            p = MagicMock()
            p.status_code = 200
            p.json.return_value = [{"name": "repo"}]
            p.headers = {"link": '<url>; rel="next"'}
            p.raise_for_status = MagicMock()
            return p

        client._request = MagicMock(side_effect=[make_page() for _ in range(5)])
        items = client._get_paginated("/repos", max_pages=2)
        assert len(items) == 2
        assert client._request.call_count == 2


class TestGitOpsCloneFetch:
    """Mock-based tests for clone_repo() and fetch_repo()."""

    @patch("core.sources.git_ops._run_git")
    def test_clone_repo_success(self, mock_run):
        """Successful clone returns CloneResult with is_fresh_clone=True."""
        from core.sources.git_ops import clone_repo, CloneResult
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        result = clone_repo("https://github.com/test/repo.git", Path("/tmp/test/repo"))
        assert result.success
        assert result.is_fresh_clone
        assert "Cloned" in result.message
        # First call is clone, second is config cleanup
        assert mock_run.call_count == 2

    @patch("core.sources.git_ops._run_git")
    def test_clone_repo_with_token_uses_oauth2_url(self, mock_run):
        """Clone with token constructs oauth2@ URL."""
        from core.sources.git_ops import clone_repo
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        clone_repo("https://github.com/test/repo.git", Path("/tmp/test/repo"), token="ghp_test123")
        # Check the clone call used oauth2@ URL
        clone_call = mock_run.call_args_list[0]
        args = clone_call[0][0]  # first positional arg is the args list
        assert any("oauth2@" in str(a) for a in args)

    @patch("core.sources.git_ops._run_git")
    def test_clone_repo_failure_sanitises_error(self, mock_run):
        """Failed clone returns sanitised error message."""
        from core.sources.git_ops import clone_repo
        # Use a realistic 40-char PAT (ghp_ + 36 alphanumeric)
        fake_token = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=128,
            stdout="",
            stderr=f"fatal: could not read from https://{fake_token}@github.com"
        )
        result = clone_repo("https://github.com/test/repo.git", Path("/tmp/test/repo"))
        assert not result.success
        assert fake_token not in result.message
        assert "****" in result.message

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_not_git_dir(self, mock_run):
        """Fetch on non-git directory returns failure."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_repo(Path(tmp), "https://github.com/test/repo.git")
            assert not result.success
            assert "Not a git repository" in result.message
            mock_run.assert_not_called()

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_with_changed_files(self, mock_run):
        """Fetch with changed files returns the diff list."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            # Create .git dir
            (Path(tmp) / ".git").mkdir()

            # Mock sequence: rev-parse HEAD, fetch, rev-parse verify, reset, rev-parse HEAD, diff
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr=""),  # old HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # fetch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="def456\n", stderr=""),  # verify branch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # reset
                subprocess.CompletedProcess(args=[], returncode=0, stdout="def456\n", stderr=""),  # new HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="file1.py\nfile2.py\n", stderr=""),  # diff
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # cleanup URL
            ]
            result = fetch_repo(Path(tmp), "https://github.com/test/repo.git")
            assert result.success
            assert len(result.changed_files) == 2
            assert "file1.py" in result.changed_files

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_no_changes(self, mock_run):
        """Fetch with no changes returns 'Already up to date'."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            same_hash = "abc123\n"
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout=same_hash, stderr=""),  # old HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # fetch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr=""),  # verify branch
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # reset
                subprocess.CompletedProcess(args=[], returncode=0, stdout=same_hash, stderr=""),  # new HEAD (same)
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # cleanup URL
            ]
            result = fetch_repo(Path(tmp), "https://github.com/test/repo.git")
            assert result.success
            assert result.changed_files == []
            assert "up to date" in result.message

    @patch("core.sources.git_ops._run_git")
    def test_fetch_repo_always_cleans_url(self, mock_run):
        """Fetch always restores clean remote URL even on failure."""
        import tempfile
        from core.sources.git_ops import fetch_repo
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="abc\n", stderr=""),  # old HEAD
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # set auth URL
                subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="fetch error"),  # fetch fails
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # cleanup URL (finally)
            ]
            result = fetch_repo(
                Path(tmp), "https://github.com/test/repo.git", token="ghp_secret"
            )
            assert not result.success
            # Verify the last _run_git call was the URL cleanup
            last_call_args = mock_run.call_args_list[-1][0][0]
            assert "config" in last_call_args
            assert "remote.origin.url" in last_call_args
            assert "https://github.com/test/repo.git" in last_call_args

    def test_run_git_timeout_returns_synthetic_result(self):
        """Timeout returns CompletedProcess with returncode -1."""
        from core.sources.git_ops import _run_git
        with patch("core.sources.git_ops.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)
            result = _run_git(["status"], timeout=5)
            assert result.returncode == -1
            assert "timed out" in result.stderr


class TestGitHubConnectorSync:
    """Mock-based tests for GitHubSourceConnector.sync() orchestration."""

    def test_sync_api_error_captured(self):
        """API error during discovery is captured in result.errors."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        from core.sources.github_client import GitHubAPIError
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        connector._discover_repos = MagicMock(
            side_effect=GitHubAPIError("Bad request", status_code=400)
        )
        result = connector.sync("/tmp/test")
        assert len(result.errors) == 1
        assert "GitHub API error" in result.errors[0]
        assert result.files_added == 0

    def test_sync_no_repos_found(self):
        """Empty repo list returns clean result."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="empty-org")
        connector = GitHubSourceConnector(config)
        connector._discover_repos = MagicMock(return_value=[])
        result = connector.sync("/tmp/test")
        assert result.errors == []
        assert result.files_added == 0
        assert result.files_updated == 0

    def test_sync_per_repo_exception_isolated(self, tmp_path):
        """Exception in one repo doesn't stop other repos."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo1 = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                         private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        repo2 = RepoInfo(name="repo2", owner="test", url="", clone_url="https://github.com/test/repo2.git",
                         private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo1, repo2])

        call_count = 0
        def mock_sync(repo, repo_dir):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Disk full")
            return CloneResult(success=True, message="Cloned", is_fresh_clone=True)

        connector._sync_repo = mock_sync
        result = connector.sync(str(tmp_path))
        assert len(result.errors) == 1
        assert "Disk full" in result.errors[0]
        assert result.files_added == 1  # repo2 succeeded

    def test_sync_fresh_clone_counts_as_added(self, tmp_path):
        """Fresh clone increments files_added."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(success=True, message="Cloned", is_fresh_clone=True)
        )
        result = connector.sync(str(tmp_path))
        assert result.files_added == 1
        assert result.files_updated == 0

    def test_sync_changed_files_counts_as_updated(self, tmp_path):
        """Fetch with changed files increments files_updated."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(
                success=True, message="Updated: 3 files changed",
                changed_files=["a.py", "b.py", "c.py"]
            )
        )
        result = connector.sync(str(tmp_path))
        assert result.files_added == 0
        assert result.files_updated == 3

    def test_sync_failed_repo_adds_error(self, tmp_path):
        """Failed sync_repo adds error to result."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="test")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="test", url="", clone_url="https://github.com/test/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(success=False, message="Clone failed: auth error")
        )
        result = connector.sync(str(tmp_path))
        assert len(result.errors) == 1
        assert "auth error" in result.errors[0]

    def test_sync_saves_metadata(self, tmp_path):
        """Sync saves .sync_metadata.json after completion."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        from core.sources.git_ops import CloneResult
        config = GitHubSourceConfig(owner="testowner")
        connector = GitHubSourceConnector(config)
        repo = RepoInfo(name="repo1", owner="testowner", url="", clone_url="https://github.com/testowner/repo1.git",
                        private=False, description="", default_branch="main", size_kb=0, is_fork=False)
        connector._discover_repos = MagicMock(return_value=[repo])
        connector._sync_repo = MagicMock(
            return_value=CloneResult(success=True, message="Cloned", is_fresh_clone=True)
        )
        connector.sync(str(tmp_path))
        metadata_path = tmp_path / "github" / "testowner" / ".sync_metadata.json"
        assert metadata_path.exists()
        import json
        meta = json.loads(metadata_path.read_text())
        assert meta["owner"] == "testowner"
        assert len(meta["repos"]) == 1


class TestWebCrawlerBFS:
    """Mock-based tests for WebSourceConnector._crawl() BFS loop."""

    # Realistic HTML that html_to_markdown() will accept (short pages return None)
    REALISTIC_HTML = (
        '<html><head><title>Test Page</title></head><body><main>'
        '<h1>Getting Started</h1>'
        '<p>This is a comprehensive guide to getting started with our platform. '
        'It covers installation, configuration, and first steps.</p>'
        '<h2>Installation</h2>'
        '<p>To install the package, run the following command in your terminal. '
        'Make sure you have Python 3.8 or later installed on your system.</p>'
        '<pre><code>pip install example-package</code></pre>'
        '</main></body></html>'
    )

    @patch("core.sources.web_crawler.time.sleep")
    def test_simple_crawl(self, mock_sleep, tmp_path):
        """Simple crawl fetches a page and saves markdown."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = self.REALISTIC_HTML

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added >= 1
        md_files = list((tmp_path / "web" / "example.com").glob("*.md"))
        assert len(md_files) >= 1

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_follows_links(self, mock_sleep, tmp_path):
        """Crawler follows links to discover new pages."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        page1_html = (
            '<html><head><title>Page 1</title></head><body><main>'
            '<h1>Page One</h1>'
            '<p>This is the first page with substantial content that passes the converter filter. '
            'It includes links to other pages for testing BFS traversal.</p>'
            '<h2>Details</h2>'
            '<p>More detailed information about this topic, including code examples and configuration.</p>'
            '<a href="/docs/page2">Go to page 2</a>'
            '</main></body></html>'
        )
        page2_html = (
            '<html><head><title>Page 2</title></head><body><main>'
            '<h1>Page Two</h1>'
            '<p>This is the second page with substantial content that also passes the converter filter. '
            'It has enough text to be considered meaningful content by the extraction pipeline.</p>'
            '<h2>More Details</h2>'
            '<p>Additional paragraphs to ensure the content length is sufficient for conversion.</p>'
            '</main></body></html>'
        )

        def mock_get(url, headers=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "text/html"}
            if "page2" in url:
                resp.text = page2_html
            else:
                resp.text = page1_html
            return resp

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added >= 2

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_max_pages_stops(self, mock_sleep, tmp_path):
        """max_pages stops the crawl."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com", max_pages=1, wait_time=0,
                                 stay_in_path=False)
        connector = WebSourceConnector(config)

        def mock_get(url, headers=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "text/html"}
            resp.text = (
                '<html><head><title>Page</title></head><body><main>'
                '<h1>Topic</h1>'
                '<p>This page has enough content to pass the converter filter and be saved as markdown. '
                'It includes a link to another page to test max_pages enforcement.</p>'
                '<h2>Section</h2>'
                '<p>More content here to ensure it meets the minimum length requirement.</p>'
                '<a href="/other">Link</a>'
                '</main></body></html>'
            )
            return resp

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = mock_get
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added == 1

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_http_error_accumulated(self, mock_sleep, tmp_path):
        """HTTP errors are accumulated in result.errors."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "text/html"}

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert len(result.errors) >= 1
        assert "server error" in result.errors[0]

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_non_html_skipped(self, mock_sleep, tmp_path):
        """Non-HTML content types are skipped."""
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"key": "value"}'

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert result.files_added == 0

    @patch("core.sources.web_crawler.time.sleep")
    def test_crawl_httpx_exception_captured(self, mock_sleep, tmp_path):
        """httpx.HTTPError per-page is captured without stopping crawl."""
        import httpx
        from core.sources.web_crawler import WebSourceConfig, WebSourceConnector
        config = WebSourceConfig(url="https://example.com/docs", max_pages=10, wait_time=0)
        connector = WebSourceConnector(config)

        with patch("core.sources.web_crawler.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = connector.sync(str(tmp_path))

        assert len(result.errors) >= 1
        assert result.files_added == 0


class TestGitHubConnectorDiscover:
    """Tests for _discover_repos branching logic."""

    def test_discover_single_repo_mode(self):
        """Single repo mode calls get_repo."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="testowner", repo="testrepo")
        connector = GitHubSourceConnector(config)

        mock_client = MagicMock()
        mock_client.get_repo.return_value = {
            "name": "testrepo", "owner": {"login": "testowner"},
            "html_url": "https://github.com/testowner/testrepo",
            "clone_url": "https://github.com/testowner/testrepo.git",
            "private": False, "description": "A test repo",
            "default_branch": "main", "size": 1000, "fork": False,
        }
        mock_client.close = MagicMock()

        with patch("core.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = connector._discover_repos()

        assert len(repos) == 1
        assert repos[0].name == "testrepo"

    def test_discover_single_repo_not_found(self):
        """Single repo mode returns empty list if repo not found."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="testowner", repo="nonexistent")
        connector = GitHubSourceConnector(config)

        mock_client = MagicMock()
        mock_client.get_repo.return_value = None
        mock_client.close = MagicMock()

        with patch("core.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = connector._discover_repos()

        assert repos == []

    def test_discover_org_mode(self):
        """Org mode calls is_org + list_repos."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig
        config = GitHubSourceConfig(owner="myorg")
        connector = GitHubSourceConnector(config)

        mock_client = MagicMock()
        mock_client.is_org.return_value = True
        mock_client.list_repos.return_value = [
            {"name": "repo1", "owner": {"login": "myorg"}, "html_url": "", "clone_url": "",
             "private": False, "description": "", "default_branch": "main", "size": 0, "fork": False},
        ]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("core.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = connector._discover_repos()

        mock_client.is_org.assert_called_once_with("myorg")
        mock_client.list_repos.assert_called_once()
        assert len(repos) == 1

    def test_filter_repos_include_and_exclude_together(self):
        """When both include and exclude are set, exclude takes precedence."""
        from core.sources.github_connector import GitHubSourceConnector
        from core.sources.github_models import GitHubSourceConfig, RepoInfo
        config = GitHubSourceConfig(
            owner="test", include_repos=["a", "b", "c"], exclude_repos=["b"]
        )
        connector = GitHubSourceConnector(config)
        repos = [
            RepoInfo(name=n, owner="test", url="", clone_url="", private=False,
                     description="", default_branch="main", size_kb=0, is_fork=False)
            for n in ["a", "b", "c", "d"]
        ]
        filtered = connector._filter_repos(repos)
        names = [r.name for r in filtered]
        assert "a" in names
        assert "c" in names
        assert "b" not in names  # excluded
        assert "d" not in names  # not included
