"""
Markdown ingestion: frontmatter, image stripping, heading breadcrumbs, citations in results.
"""

import datetime
import logging

import pytest

PAGE = """---
title: Feeding a starter
url: https://example.com/guides/starter
date: 2026-10-01
domain: example.com
page_type: guide
scraped_at: 2026-11-01T09:30:00
content_hash: 9f2c
tags: [bread, yeast]
---
# Feeding a starter

A sourdough starter is a culture of wild yeast and lactic acid bacteria that you keep alive with regular feeding.

![A jar of bubbly starter](https://example.com/img/jar.jpg)

## Daily routine

Discard half of the starter, then add equal weights of flour and water. Keep it at room temperature and stir well.

### Flour choice

Rye flour makes a more active starter than white flour, because it carries more wild yeast and nutrients.

## Storage

Keep a mature starter in the fridge and feed it once a week. Let it warm up before baking with it again.
"""


class TestFrontmatter:
    def test_fields_become_metadata_and_block_is_removed(self):
        from core.markdown import frontmatter_metadata, split_frontmatter

        fm, body = split_frontmatter(PAGE)
        assert body.startswith("# Feeding a starter")
        assert "page_type:" not in body
        meta = frontmatter_metadata(fm)
        assert meta == {
            "title": "Feeding a starter",
            "url": "https://example.com/guides/starter",
            "date": "2026-10-01",
            "domain": "example.com",
            "page_type": "guide",
            "scraped_at": "2026-11-01T09:30:00",
            "page_content_hash": "9f2c",
        }

    def test_values_are_scalars_chroma_accepts(self):
        from core.markdown import frontmatter_metadata

        meta = frontmatter_metadata({"tags": 1, "date": datetime.datetime(2026, 1, 2, 3, 4), "domain": ["a", "b"]})
        assert meta == {"date": "2026-01-02T03:04:00", "domain": "a, b"}

    def test_numeric_title_and_url_become_strings(self):
        from core.markdown import frontmatter_metadata

        meta = frontmatter_metadata({"title": 2026, "url": 12345})
        assert meta == {"title": "2026", "url": "12345"}

    def test_url_number_from_yaml_is_a_string(self):
        from core.markdown import frontmatter_metadata, split_frontmatter

        fm, _ = split_frontmatter("---\ntitle: '  Spaced  '\nurl: 12345\n---\nBody")
        assert frontmatter_metadata(fm) == {"title": "Spaced", "url": "12345"}

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True, ["a", "b"], {"k": "v"}, "   "])
    def test_unusable_title_and_url_are_ignored(self, value):
        from core.markdown import frontmatter_metadata

        assert frontmatter_metadata({"title": value, "url": value, "domain": "example.com"}) == {
            "domain": "example.com",
        }

    def test_nan_title_from_yaml_is_ignored(self):
        from core.markdown import frontmatter_metadata, split_frontmatter

        fm, _ = split_frontmatter("---\ntitle: .nan\nurl: https://example.com/a\n---\nBody")
        assert frontmatter_metadata(fm) == {"url": "https://example.com/a"}

    def test_long_title_is_capped(self):
        from core.markdown import MAX_TITLE_CHARS, frontmatter_metadata

        meta = frontmatter_metadata({"title": "x" * 15000})
        assert MAX_TITLE_CHARS == 300
        assert meta["title"] == "x" * 300

    def test_no_frontmatter(self):
        from core.markdown import split_frontmatter

        assert split_frontmatter("# Title\n\nBody") == ({}, "# Title\n\nBody")

    @pytest.mark.parametrize("text", [
        "---\ntitle: [unclosed\n---\n# Body\n",          # invalid YAML
        "---\n- just\n- a list\n---\n# Body\n",          # not a mapping
        "---\ntitle: no closing line\n# Body\n",          # never closed
    ])
    def test_bad_frontmatter_keeps_the_whole_text(self, text, caplog):
        from core.markdown import split_frontmatter

        with caplog.at_level(logging.WARNING, logger="core.markdown"):
            fm, body = split_frontmatter(text, source="bad.md")
        assert fm == {}
        assert body == text
        assert "bad.md" in caplog.text


class TestStripImages:
    def test_inline_image_becomes_alt_text(self):
        from core.markdown import strip_images

        assert strip_images("See ![the chart](c.png) above.") == "See the chart above."
        assert strip_images('See ![chart](c.png "Title") now') == "See chart now"

    def test_image_only_lines_are_dropped(self):
        from core.markdown import strip_images

        text = "Before\n![logo](logo.png)\n  ![a](1.png) ![b](2.png)  \n[![badge](b.svg)](https://ci)\nAfter"
        assert strip_images(text) == "Before\nAfter"

    def test_linked_and_reference_images(self):
        from core.markdown import strip_images

        assert strip_images("Build [![status](b.svg)](https://ci) ok") == "Build status ok"
        assert strip_images("Logo ![brand][logo] here") == "Logo brand here"

    def test_code_blocks_are_untouched(self):
        from core.markdown import strip_images

        text = "```md\n![keep](me.png)\n```"
        assert strip_images(text) == text

    def test_links_are_not_images(self):
        from core.markdown import strip_images

        assert strip_images("Read [the docs](https://x.y)") == "Read [the docs](https://x.y)"


class TestSections:
    def test_breadcrumbs_follow_headings(self):
        from core.markdown import split_sections

        paths = [" > ".join(p) for p, _ in split_sections(PAGE.split("---\n", 2)[2], title="Feeding a starter")]
        assert paths == [
            "Feeding a starter",
            "Feeding a starter > Daily routine",
            "Feeding a starter > Daily routine > Flour choice",
            "Feeding a starter > Storage",
        ]

    def test_headings_in_code_fences_are_ignored(self):
        from core.markdown import split_sections

        text = "# Top\n\n```bash\n# just a comment\n```\n\n## Next\n\nbody"
        assert [p for p, _ in split_sections(text)] == [["Top"], ["Top", "Next"]]

    def test_underlined_headings(self):
        from core.markdown import split_sections

        text = "Main\n====\n\nIntro\n\nPart\n----\n\nMore"
        assert [p for p, _ in split_sections(text)] == [["Main"], ["Main", "Part"]]

    def test_long_section_is_split_and_keeps_its_section(self):
        from core.markdown import chunk_markdown

        text = "# Doc\n\n## Long\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
        chunks = chunk_markdown(text, max_chars=400, overlap=0)
        assert len(chunks) > 2
        assert all(section == "Doc > Long" for _, section in chunks[1:])
        assert all(len(content) <= 400 for content, _ in chunks)

    def test_short_section_merges_into_previous_with_shared_headings(self):
        from core.markdown import chunk_markdown

        text = "# Doc\n\n## Install\n\n" + "Install it like this. " * 8 + "\n\n### Extra\n\npip install x\n"
        chunks = chunk_markdown(text, max_chars=1000)
        assert chunks[-1][1] == "Doc > Install"
        assert "pip install x" in chunks[-1][0]

    def test_first_heading(self):
        from core.markdown import first_heading

        assert first_heading("intro\n\n## Sub\n\n# Real title\n") == "Real title"
        assert first_heading("```\n# no\n```\nTitle\n=====\n") == "Title"
        assert first_heading("no headings") is None


class TestFences:
    """A fence closes only with the same character, at least as many times as it opened."""

    def test_tildes_do_not_close_a_backtick_fence(self):
        from core.markdown import first_heading, split_sections, strip_images

        text = "# Top\n\n```\n~~~\n# not a heading\n![keep](me.png)\n```\n\n## Next\n\nbody"
        assert [p for p, _ in split_sections(text)] == [["Top"], ["Top", "Next"]]
        assert "![keep](me.png)" in strip_images(text)
        assert first_heading("```\n~~~\n# no\n```\n# Yes\n") == "Yes"

    def test_shorter_fence_does_not_close_a_longer_one(self):
        from core.markdown import first_heading, split_sections, strip_images

        text = "# Top\n\n````\n```\n# not a heading\n![keep](me.png)\n````\n\n## Next\n\nbody"
        assert [p for p, _ in split_sections(text)] == [["Top"], ["Top", "Next"]]
        assert "![keep](me.png)" in strip_images(text)
        assert first_heading("````\n```\n# no\n````\n# Yes\n") == "Yes"

    def test_longer_closing_fence_closes(self):
        from core.markdown import split_sections

        text = "# Top\n\n```\ncode\n`````\n\n## Next\n\nbody"
        assert [p for p, _ in split_sections(text)] == [["Top"], ["Top", "Next"]]

    def test_unclosed_fence_runs_to_the_end(self):
        from core.markdown import first_heading, split_sections, strip_images

        text = "# Top\n\nintro\n\n````\n```\n# comment\n## also code\n![keep](me.png)\n"
        sections = split_sections(text)
        assert [p for p, _ in sections] == [["Top"]]
        assert "## also code" in sections[0][1]
        assert "![keep](me.png)" in strip_images(text)
        assert first_heading("````\n```\n# no\n") is None


LONG = "word " * 296


class TestShortSections:
    """No non-empty text may be lost, and a chunk keeps a precise breadcrumb."""

    @staticmethod
    def _find(chunks, needle):
        hits = [(content, section) for content, section in chunks if needle in content]
        assert hits, f"{needle!r} is in no chunk: {chunks}"
        return hits[0]

    def test_short_last_section_that_cannot_merge_survives(self, ingestor, tmp_path):
        from core.markdown import chunk_markdown

        text = "# T\n## Install\n" + LONG + "\n## Contact\nEmail sales@example.com"
        content, section = self._find(chunk_markdown(text, title="T"), "sales@example.com")
        assert section == "T > Contact"
        assert "## Contact" in content
        page = tmp_path / "t.md"
        page.write_text(text)
        assert any("sales@example.com" in c["content"] for c in ingestor.chunk_file(page))

    def test_short_middle_section_survives(self, ingestor, tmp_path):
        from core.markdown import chunk_markdown

        text = "# T\n## Install\n" + LONG + "\n## Contact\nEmail sales@example.com\n## Usage\n" + "word " * 200
        chunks = chunk_markdown(text, title="T")
        content, section = self._find(chunks, "sales@example.com")
        assert section == "T > Usage"  # carried into the next section, which holds most of the text
        assert content.index("## Contact") < content.index("## Usage")
        page = tmp_path / "t.md"
        page.write_text(text)
        assert any("sales@example.com" in c["content"] for c in ingestor.chunk_file(page))

    def test_short_first_section_survives(self, ingestor, tmp_path):
        from core.markdown import chunk_markdown

        text = "# T\n## Intro\nHello there.\n## Install\n" + "word " * 200
        chunks = chunk_markdown(text, title="T")
        content, section = self._find(chunks, "Hello there.")
        assert section == "T > Install"
        assert "## Intro" in content
        page = tmp_path / "t.md"
        page.write_text(text)
        assert any("Hello there." in c["content"] for c in ingestor.chunk_file(page))

    def test_consecutive_short_sections_all_survive(self, ingestor, tmp_path):
        from core.markdown import chunk_markdown

        text = ("# T\n## Install\n" + LONG + "\n## Phone\nCall 0100.\n## Fax\nNone.\n## Post\nPO Box 1."
                "\n## Usage\n" + LONG + "\n## Email\nhi@example.com\n## Web\nexample.com")
        chunks = chunk_markdown(text, title="T")
        for needle in ("Call 0100.", "None.", "PO Box 1.", "hi@example.com", "## Web"):
            self._find(chunks, needle)
        page = tmp_path / "t.md"
        page.write_text(text)
        joined = "\n".join(c["content"] for c in ingestor.chunk_file(page))
        for needle in ("Call 0100.", "None.", "PO Box 1.", "hi@example.com", "## Web"):
            assert needle in joined

    def test_tiny_page_is_still_indexed(self, ingestor, tmp_path):
        page = tmp_path / "t.md"
        page.write_text("# Hours\nOpen 9 to 5.")
        chunks = ingestor.chunk_file(page)
        assert [c["metadata"]["section"] for c in chunks] == ["Hours"]

    def test_short_section_found_by_search(self, clients_dir, make_client, capsys, monkeypatch):
        import json
        import sys

        import scripts.ingest as ingest_cli
        import scripts.search as search_cli

        name = make_client("contact")
        docs = clients_dir / name / "documents"
        docs.mkdir()
        (docs / "t.md").write_text("# T\n## Install\n" + LONG + "\n## Contact\nEmail sales@example.com")
        monkeypatch.setattr(sys, "argv", ["ingest", "--clients-dir", str(clients_dir), "--client", name,
                                          "--json", "--no-code"])
        ingest_cli.main()
        capsys.readouterr()
        monkeypatch.setenv("TROVE_CLIENTS_DIR", str(clients_dir))
        monkeypatch.setattr(sys, "argv", ["search", "--client", name, "--json", "--full", "sales email contact"])
        search_cli.main()
        results = json.loads(capsys.readouterr().out)["results"]
        assert any("sales@example.com" in r["content"] for r in results), results


class TestMergedBreadcrumb:
    def test_merged_chunk_keeps_the_long_sections_heading(self):
        from core.markdown import chunk_markdown

        text = "# T\n## Install\n" + "Install it like this. " * 20 + "\n## Usage\nRun it now."
        chunks = chunk_markdown(text, title="T")
        assert len(chunks) == 1
        content, section = chunks[0]
        assert section == "T > Install"
        assert "## Usage" in content and "Run it now." in content

    def test_short_first_section_merged_forward_takes_the_long_sections_heading(self):
        from core.markdown import chunk_markdown

        text = "# T\n## Intro\nHello there.\n## Install\n" + "Install it like this. " * 20
        chunks = chunk_markdown(text, title="T")
        assert len(chunks) == 1
        content, section = chunks[0]
        assert section == "T > Install"
        assert content.startswith("# T\n\n## Intro")


@pytest.fixture(scope="module")
def ingestor():
    from core.ingest import DocumentIngestor

    return DocumentIngestor()


class TestMarkdownIngest:
    def test_chunks_carry_frontmatter_and_section(self, ingestor, tmp_path):
        page = tmp_path / "starter.md"
        page.write_text(PAGE)
        chunks = ingestor.chunk_file(page)
        assert chunks
        sections = [c["metadata"]["section"] for c in chunks]
        assert "Feeding a starter > Daily routine > Flour choice" in sections
        for chunk in chunks:
            meta = chunk["metadata"]
            assert meta["title"] == "Feeding a starter"
            assert meta["url"] == "https://example.com/guides/starter"
            assert meta["date"] == "2026-10-01"
            assert "page_type:" not in chunk["content"]
            assert "jar.jpg" not in chunk["content"]
            assert all(isinstance(v, (str, int, float, bool, type(None))) for v in meta.values())

    def test_image_stripping_can_be_turned_off(self, tmp_path):
        from core.ingest import DocumentIngestor

        page = tmp_path / "starter.md"
        page.write_text(PAGE)
        keep = DocumentIngestor(markdown_config={"strip_images": False})
        text = "\n".join(c["content"] for c in keep.chunk_file(page))
        assert "![A jar of bubbly starter](https://example.com/img/jar.jpg)" in text

    def test_bad_frontmatter_still_indexes_the_file(self, ingestor, tmp_path):
        page = tmp_path / "broken.md"
        page.write_text("---\ntitle: [oops\n---\n# Broken page\n\n" + "Real content about rivers and valleys. " * 5)
        chunks = ingestor.chunk_file(page)
        assert chunks
        assert any("rivers and valleys" in c["content"] for c in chunks)
        assert chunks[0]["metadata"]["title"] == "Broken page"  # falls back to the first H1

    def test_title_falls_back_to_first_heading(self, ingestor, tmp_path):
        page = tmp_path / "plain.md"
        page.write_text("# Plain page\n\n" + "Some body text that is long enough to be indexed on its own. " * 3)
        chunk = ingestor.chunk_file(page)[0]
        assert chunk["metadata"]["title"] == "Plain page"
        assert chunk["metadata"]["section"] == "Plain page"
        assert "url" not in chunk["metadata"]

    def test_non_markdown_files_are_unchanged(self, ingestor, tmp_path):
        page = tmp_path / "notes.txt"
        page.write_text("# Not a heading here\n\n" + "Plain text notes about rivers and valleys. " * 5)
        chunks = ingestor.chunk_file(page)
        assert chunks
        assert all("section" not in c["metadata"] for c in chunks)
        assert all(c["metadata"]["element_type"] != "MarkdownSection" for c in chunks)


class TestCitationInResults:
    CHUNK = {
        "id": "x",
        "content": "Discard half of the starter.",
        "metadata": {"source_file": "starter.md", "title": "Feeding a starter",
                     "url": "https://example.com/guides/starter", "section": "Feeding a starter > Daily routine"},
        "_rerank_score": 1.5,
    }

    def test_citation_helper(self):
        from core.search.searcher import citation

        assert citation(self.CHUNK) == {
            "title": "Feeding a starter",
            "url": "https://example.com/guides/starter",
            "section": "Feeding a starter > Daily routine",
        }
        assert citation({"metadata": {}}) == {"title": None, "url": None, "section": None}

    def test_mcp_formatting_shows_title_url_section(self):
        from mcp_server.shared import format_search_results

        text = format_search_results([self.CHUNK])
        assert "**Source:** starter.md" in text
        assert "**Title:** Feeding a starter" in text
        assert "**URL:** https://example.com/guides/starter" in text
        assert "**Section:** Feeding a starter > Daily routine" in text

    def test_context_prefix_includes_section(self):
        from core.ingest import build_context_prefix

        assert "Section: Feeding a starter > Daily routine." in build_context_prefix(self.CHUNK["metadata"])
