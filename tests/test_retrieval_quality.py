"""
Retrieval quality on small synthetic corpora: result count, per-page diversity
and heading matches. Each corpus is invented to reproduce one failure mode.
"""

import pytest


def _index(searcher, chunks):
    embeddings = searcher.embedder.encode([c["content"] for c in chunks]).tolist()
    searcher.index_chunks(chunks, embeddings)


@pytest.fixture
def searcher_factory(tmp_path):
    from core.search import HybridSearcher

    made = []

    def make(**kwargs):
        searcher = HybridSearcher(index_path=str(tmp_path / f"idx{len(made)}"), ephemeral=True,
                                  collection_name=f"quality{len(made)}", cache_ttl=0, **kwargs)
        made.append(searcher)
        return searcher

    yield make
    for searcher in made:
        searcher.close()


ANIMALS = [
    "otter", "badger", "heron", "falcon", "beaver", "marten", "weasel", "plover", "curlew", "lapwing",
    "osprey", "kestrel", "puffin", "gannet", "shrew", "vole", "stoat", "ferret", "hedgehog", "dormouse",
    "squirrel", "rabbit", "hare", "deer", "fox", "wren", "robin", "thrush", "finch", "linnet",
    "swift", "swallow", "martin", "pipit", "wagtail", "dipper", "kingfisher", "egret", "bittern", "crane",
    "stork", "ibis", "spoonbill", "avocet", "dunlin", "knot", "sanderling", "turnstone", "redshank", "godwit",
]


class TestTopK:
    def test_top_k_above_the_adaptive_depth_is_honoured(self, searcher_factory):
        """Regression: with a confident top match, adaptive reranking cut the candidates to its
        depth, so asking for more results than that still returned only the depth."""
        searcher = searcher_factory(adaptive_rerank_depth=15)
        chunks = [
            {"id": f"c{i}", "content": f"Field notes on the {name}: the {name} lives near the river bank.",
             "metadata": {"source_file": f"{name}.md"}}
            for i, name in enumerate(ANIMALS)
        ]
        _index(searcher, chunks)

        results = searcher.search("Field notes on the otter: the otter lives near the river bank.", top_k=40)

        assert searcher._last_search_quality.adaptive_rerank_applied
        assert len(results) == 40

    def test_top_k_above_rerank_top_n_is_honoured(self, searcher_factory):
        """A top_k larger than the candidate pool widens the pool instead of being cut to it."""
        searcher = searcher_factory(adaptive_rerank=False)
        chunks = [
            {"id": f"c{i}", "content": f"The {name} was seen near the river bank this morning.",
             "metadata": {"source_file": f"{name}.md"}}
            for i, name in enumerate(ANIMALS)
        ]
        _index(searcher, chunks)

        results = searcher.search("river bank sightings", top_k=45, rerank_top_n=20)

        assert len(results) == 45

    def test_top_k_is_capped_at_the_maximum(self, searcher_factory):
        from core.search.searcher import MAX_TOP_K

        searcher = searcher_factory(adaptive_rerank=False)
        chunks = [
            {"id": f"c{i}", "content": f"Entry {i}: the {ANIMALS[i % 50]} number {i} by the river bank.",
             "metadata": {"source_file": f"entry{i}.md"}}
            for i in range(MAX_TOP_K + 20)
        ]
        _index(searcher, chunks)

        results = searcher.search("river bank", top_k=MAX_TOP_K + 50)

        assert len(results) == MAX_TOP_K


def _page_chunk(chunk_id, url, text, **meta):
    return {"id": chunk_id, "content": text,
            "metadata": {"source_file": url.rsplit("/", 1)[-1] + ".md", "url": url, **meta}}


class TestLimitPerPage:
    def test_extra_chunks_of_a_page_move_behind_other_pages(self):
        from core.search.searcher import limit_per_page

        chunks = [_page_chunk(i, url, "x") for i, url in
                  enumerate(["https://e.org/a", "https://e.org/a", "https://e.org/a", "https://e.org/b", "https://e.org/a"])]
        ordered = limit_per_page(chunks, 2)
        assert [c["id"] for c in ordered] == [0, 1, 3, 2, 4]

    def test_one_page_alone_keeps_all_its_chunks_in_order(self):
        from core.search.searcher import limit_per_page

        chunks = [_page_chunk(i, "https://e.org/a", "x") for i in range(5)]
        assert limit_per_page(chunks, 2) == chunks

    def test_zero_or_none_turns_it_off(self):
        from core.search.searcher import limit_per_page

        chunks = [_page_chunk(i, "https://e.org/a", "x") for i in range(3)] + [_page_chunk(9, "https://e.org/b", "x")]
        assert limit_per_page(chunks, 0) == chunks
        assert limit_per_page(chunks, None) == chunks

    def test_print_copy_counts_as_its_canonical_page(self):
        from core.search.searcher import _norm_url, page_key

        main = _page_chunk(1, "https://e.org/manual", "x")
        copy = _page_chunk(2, "https://e.org/manual/print/7", "x", canonical_url="https://e.org/manual/")
        known = {_norm_url(c["metadata"]["url"]) for c in (main, copy)}
        assert page_key(copy, known) == page_key(main, known)

    @pytest.mark.parametrize("canonical", [
        "http://e.org/manual",           # other scheme
        "https://E.ORG/manual/",         # host in capitals
        "https://e.org:443/manual#top",  # default port and a fragment
    ])
    def test_canonical_url_variants_still_match(self, canonical):
        from core.search.searcher import _norm_url, page_key

        main = _page_chunk(1, "https://e.org/manual", "x")
        copy = _page_chunk(2, "https://e.org/manual/print/7", "x", canonical_url=canonical)
        known = {_norm_url(c["metadata"]["url"]) for c in (main, copy)}
        assert page_key(copy, known) == page_key(main, known)

    def test_canonical_url_on_another_host_is_ignored(self):
        """A page cannot claim another site's URL and share that page's quota."""
        from core.search.searcher import _norm_url, page_key

        other_site = _page_chunk(1, "https://vendor.example/manual", "x")
        claimant = _page_chunk(2, "https://mirror.example/copy", "x", canonical_url="https://vendor.example/manual")
        known = {_norm_url(c["metadata"]["url"]) for c in (other_site, claimant)}
        assert page_key(claimant, known) != page_key(other_site, known)
        assert page_key(claimant, known) == _norm_url("https://mirror.example/copy")

    def test_canonical_url_of_an_unindexed_page_is_ignored(self):
        """A canonical URL that is no other file's URL does not merge unrelated pages."""
        from core.search.searcher import page_key

        chunk = _page_chunk(1, "https://e.org/one", "x", canonical_url="https://e.org/")
        assert page_key(chunk, {"e.org/one"}) == "e.org/one"

    def test_chunks_without_a_url_group_by_source_file(self):
        from core.search.searcher import page_key

        chunk = {"id": 1, "content": "x", "metadata": {"source_file": "manual.pdf"}}
        assert page_key(chunk) == "manual.pdf"


CHATTY = "https://blog.example/backup-weekend"
CHATTY_PRINT = "https://blog.example/backup-weekend/print/3"
GUIDE = "https://docs.example/storage-guide"


class TestPageDiversityInSearch:
    """One chatty page with many matching chunks (and a print copy) must not fill the top results."""

    def _corpus(self):
        chatty = [
            "Backup weekend tips: back up the desktop, back up the phone, then check every backup twice.",
            "Backup tips for beginners: copy backups to a drive, label the backups, test a backup monthly.",
            "Storing backups: keep one backup at home, one backup offsite, and rotate the backup drives.",
            "Backup variations: mix cloud backups and drive backups so every backup has a second copy.",
        ]
        chunks = [_page_chunk(f"c{i}", CHATTY, t) for i, t in enumerate(chatty)]
        chunks += [_page_chunk(f"p{i}", CHATTY_PRINT, t + " Printed copy.", canonical_url=CHATTY + "/")
                   for i, t in enumerate(chatty[:2])]
        chunks.append(_page_chunk("g0", GUIDE, "Backup schedule for servers: keep daily snapshots for two weeks."))
        chunks.append(_page_chunk("x0", "https://other.example/keyboard", "Keyboard layouts can be changed in the settings menu."))
        return chunks

    def test_other_pages_reach_the_top_results(self, searcher_factory):
        searcher = searcher_factory(max_chunks_per_source=2)
        _index(searcher, self._corpus())

        results = searcher.search("backup tips", top_k=6)

        top3 = [r["metadata"]["url"] for r in results[:3]]
        assert GUIDE in top3, top3
        chatty_group = [u for u in top3 if u in (CHATTY, CHATTY_PRINT)]
        assert len(chatty_group) <= 2
        assert len(results) == 6  # moved chunks still come back, later in the list

    def test_on_by_default_at_two_per_page(self, searcher_factory):
        """With no setting, a page's third matching chunk comes after another page's chunk."""
        searcher = searcher_factory()
        assert searcher.max_chunks_per_source == 2
        _index(searcher, self._corpus())

        results = searcher.search("backup tips", top_k=6)

        first_other = next(i for i, r in enumerate(results) if r["metadata"]["url"] not in (CHATTY, CHATTY_PRINT))
        assert first_other == 2, [r["id"] for r in results]

    def test_off_when_set_to_zero(self, searcher_factory):
        searcher = searcher_factory(max_chunks_per_source=0)
        _index(searcher, self._corpus())

        results = searcher.search("backup tips", top_k=6)

        top3 = [r["metadata"]["url"] for r in results[:3]]
        assert GUIDE not in top3, top3


class TestHeadingMatch:
    def test_share_of_query_words_in_title_and_section(self):
        from core.search.searcher import heading_match

        chunk = {"metadata": {"title": "Recovery guide", "section": "Recovery guide > Firmware > Resets"}}
        assert heading_match("firmware resets", chunk) == 1.0
        assert heading_match("firmware reset timing", chunk) == pytest.approx(2 / 3)
        assert heading_match("keyboard", chunk) == 0.0
        assert heading_match("firmware", {"metadata": {}}) == 0.0

    def test_plurals_match_loosely(self):
        from core.search.searcher import _light_stem

        assert _light_stem("resets") == _light_stem("reset")
        assert _light_stem("batteries") == _light_stem("battery")
        assert _light_stem("switches") == _light_stem("switch")
        assert _light_stem("glass") == "glass"


class TestHeadingBoostInSearch:
    """A terse manual section is found by the topic words in its headings, even when its
    text never repeats them all, ahead of chatty pages that use the words in passing."""

    GUIDE = "https://docs.example/recovery-guide"

    def _corpus(self):
        chunks = [
            _page_chunk("g0", self.GUIDE, "Factory reset: hold the power button for ten seconds before unplugging.",
                        title="Recovery guide", section="Recovery guide > Firmware reset",
                        element_type="MarkdownSection"),
        ]
        chatty = [
            "We flashed the new firmware on every router in the office and threw a small party.",
            "Firmware nights at our office: pizza, music and a stack of routers to update.",
            "Our routers run the latest firmware, and the office loves the faster wifi.",
        ]
        for i, text in enumerate(chatty):
            url = f"https://blog.example/office-notes-{i}"
            chunks.append(_page_chunk(f"r{i}", url, text, title=f"Office notes {i}",
                                      section=f"Office notes {i} > Diary", element_type="MarkdownSection"))
        return chunks

    def _rank_of_guide(self, results):
        urls = [r["metadata"]["url"] for r in results]
        return urls.index(self.GUIDE) + 1 if self.GUIDE in urls else None

    def test_manual_heading_ranks_first(self, searcher_factory):
        searcher = searcher_factory(heading_boost=1.0)
        _index(searcher, self._corpus())

        results = searcher.search("firmware reset", top_k=4)

        assert self._rank_of_guide(results) == 1, [r["id"] for r in results]
        assert results[0]["_heading_match"] == 1.0

    def test_without_the_boost_the_chatty_pages_win(self, searcher_factory):
        searcher = searcher_factory(heading_boost=0)
        _index(searcher, self._corpus())

        results = searcher.search("firmware reset", top_k=4)

        assert self._rank_of_guide(results) != 1, [r["id"] for r in results]


class TestSearchSettings:
    """Numeric search settings from client.yaml: null gives the documented default,
    any other wrong value stops with a clear error."""

    @pytest.mark.parametrize("key,default", [
        ("top_k", 50), ("rerank_top_n", 50), ("adaptive_rerank_depth", 30), ("heading_boost", 1.0),
    ])
    def test_null_gives_the_default(self, key, default):
        from core.client import search_setting

        assert search_setting({key: None}, key) == default
        assert search_setting({}, key) == default

    def test_max_chunks_per_source_defaults_to_two_and_null_means_no_limit(self):
        from core.client import DEFAULT_CONFIG, search_setting

        assert DEFAULT_CONFIG["search"]["max_chunks_per_source"] == 2
        assert search_setting({}, "max_chunks_per_source") == 2
        assert search_setting({"max_chunks_per_source": None}, "max_chunks_per_source") is None
        assert search_setting({"max_chunks_per_source": 0}, "max_chunks_per_source") == 0

    @pytest.mark.parametrize("key,value", [
        ("rerank_top_n", "50"), ("rerank_top_n", 2.5), ("rerank_top_n", True), ("rerank_top_n", 0),
        ("top_k", -1), ("adaptive_rerank_depth", [30]), ("max_chunks_per_source", -1),
        ("max_chunks_per_source", "two"), ("heading_boost", "high"), ("heading_boost", -0.5),
    ])
    def test_wrong_values_raise_a_clear_error(self, key, value):
        from core.client import ConfigError, search_setting

        with pytest.raises(ConfigError, match=f"search.{key}"):
            search_setting({key: value}, key)

    @pytest.mark.parametrize("key", ["top_k", "rerank_top_n", "adaptive_rerank_depth", "max_chunks_per_source", "heading_boost"])
    @pytest.mark.parametrize("literal", [".nan", ".inf", "-.inf"])
    def test_non_finite_values_raise_a_clear_error(self, key, literal):
        """Regression: heading_boost: .nan passed the check (NaN is never below the minimum)."""
        import yaml

        from core.client import ConfigError, search_setting

        config = yaml.safe_load(f"{key}: {literal}")
        with pytest.raises(ConfigError, match=f"search.{key}"):
            search_setting(config, key)

    def test_validate_config_warns_on_a_nan_heading_boost(self):
        import yaml

        from core.client import validate_config

        config = {"machine_name": "demo", "search": yaml.safe_load("heading_boost: .nan")}
        assert any("search.heading_boost" in w for w in validate_config(config))

    def test_valid_values_pass_through(self):
        from core.client import search_setting

        assert search_setting({"heading_boost": 0}, "heading_boost") == 0
        assert search_setting({"heading_boost": 2}, "heading_boost") == 2
        assert search_setting({"rerank_top_n": 80}, "rerank_top_n") == 80


class _FakeClient:
    def __init__(self, search):
        self.name = "Fake"
        self.machine_name = "fake"
        self.has_index = True
        self.search_config = search
        self.llm_config = {}
        self.index_path = "unused-index"
        self.tokenization_config = {}
        self.embedding_config = {}


class _FakeSearcher:
    def __init__(self):
        self.calls = []

    def search_with_quality(self, **kwargs):
        from core.search import SearchQuality, SearchResult

        self.calls.append(kwargs)
        return SearchResult(results=[], quality=SearchQuality())


def _run_cli(monkeypatch, argv, search):
    import scripts.search as cli
    from core.client import DEFAULT_CONFIG

    fake = _FakeSearcher()
    monkeypatch.setattr(cli, "load_client", lambda name: _FakeClient({**DEFAULT_CONFIG["search"], **search}))
    monkeypatch.setattr(cli, "_build_searcher", lambda client, args: fake)
    monkeypatch.setattr("sys.argv", ["search.py", *argv])
    cli.main()
    return fake


class TestCliSettings:
    def test_null_rerank_top_n_uses_the_default(self, monkeypatch):
        """Regression: rerank_top_n: null in client.yaml crashed the CLI with a TypeError."""
        fake = _run_cli(monkeypatch, ["--client", "fake", "query"], {"rerank_top_n": None, "top_k": None})
        assert fake.calls[0]["rerank_top_n"] == 50
        assert fake.calls[0]["top_k"] == 50

    def test_wrong_value_stops_with_a_clear_error(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _run_cli(monkeypatch, ["--client", "fake", "query"], {"rerank_top_n": "many"})
        assert exit_info.value.code == 1
        assert "search.rerank_top_n" in capsys.readouterr().err

    def test_top_k_above_the_maximum_is_lowered_with_a_note(self, monkeypatch, capsys):
        from core.search.searcher import MAX_TOP_K

        fake = _run_cli(monkeypatch, ["--client", "fake", "--top-k", str(MAX_TOP_K + 1), "query"], {})
        assert fake.calls[0]["top_k"] == MAX_TOP_K
        assert f"--top-k is limited to {MAX_TOP_K}" in capsys.readouterr().err

    def test_top_k_within_the_maximum_has_no_note(self, monkeypatch, capsys):
        fake = _run_cli(monkeypatch, ["--client", "fake", "--top-k", "40", "query"], {})
        assert fake.calls[0]["top_k"] == 40
        assert "limited" not in capsys.readouterr().err


class TestMcpSettings:
    def test_null_settings_use_the_defaults(self, monkeypatch, clients_dir, make_client):
        import mcp_server.shared as shared

        cache = shared.SearcherCache()
        monkeypatch.setattr(shared, "_cache", cache)
        monkeypatch.setattr(shared, "CLIENTS_DIR", clients_dir)
        name = make_client("nullsettings", search={
            "adaptive_rerank_depth": None, "heading_boost": None, "max_chunks_per_source": None,
        })
        try:
            _, searcher = shared.load_trove_client(name, require_index=False)
            assert searcher.adaptive_rerank_depth == 30
            assert searcher.heading_boost == 1.0
            assert searcher.max_chunks_per_source is None
        finally:
            cache.close_all()

    def test_wrong_setting_raises_a_clear_error(self, monkeypatch, clients_dir, make_client):
        import mcp_server.shared as shared
        from core.client import ConfigError

        monkeypatch.setattr(shared, "_cache", shared.SearcherCache())
        monkeypatch.setattr(shared, "CLIENTS_DIR", clients_dir)
        name = make_client("badsettings", search={"heading_boost": "loud"})
        with pytest.raises(ConfigError, match="search.heading_boost"):
            shared.load_trove_client(name, require_index=False)


class TestRunAgentSettings:
    def test_wrong_setting_stops_with_a_message_not_a_traceback(self, monkeypatch, capsys):
        import scripts.run_agent as run_agent

        monkeypatch.setattr(run_agent.AgentLoader, "get", lambda self, name: object())
        monkeypatch.setattr(run_agent, "load_client", lambda name: _FakeClient({"heading_boost": float("nan")}))
        monkeypatch.setattr(run_agent, "create_provider_from_config", lambda config: object())
        monkeypatch.setattr("sys.argv", ["run_agent.py", "--agent", "any", "--client", "fake", "a question"])

        with pytest.raises(SystemExit) as exit_info:
            run_agent.main()

        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert "search.heading_boost" in err
        assert "Traceback" not in err

    def test_config_error_outside_searcher_creation_is_reported(self, monkeypatch, capsys):
        """A ConfigError from anywhere in a run is printed as one line, with a non-zero exit."""
        import scripts.run_agent as run_agent
        from core.client import ConfigError

        def fail():
            raise ConfigError("search.top_k in client.yaml must be a whole number >= 1 (or null), got 'x'")

        monkeypatch.setattr(run_agent, "_main", fail)

        with pytest.raises(SystemExit) as exit_info:
            run_agent.main()

        assert exit_info.value.code == 1
        assert capsys.readouterr().err.startswith("Error: search.top_k")
