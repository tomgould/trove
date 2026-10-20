"""Handler functions for ingestion tools."""

import logging
import threading
from pathlib import Path

from mcp.types import TextContent

from core.client import load_client
from core.search import HybridSearcher

from mcp_server.shared import (
    CLIENTS_DIR,
    _cache,
    _job_manager,
    load_trove_client,
)

logger = logging.getLogger("trove-mcp")


def handle_ingest_file(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    file_path = arguments.get("file_path")

    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]
    if not file_path:
        return [TextContent(type="text", text="Error: file_path is required")]

    try:
        import time as _time
        from core.ingest import DocumentIngestor

        client = load_client(client_name, clients_dir=CLIENTS_DIR)
        fp = Path(file_path).resolve()

        # Validate file is under client's documents dir
        docs_dir = client.documents_path.resolve()
        if not fp.is_relative_to(docs_dir):
            return [TextContent(
                type="text",
                text=f"Error: File must be under client's documents directory: {docs_dir}",
            )]
        if not fp.exists():
            return [TextContent(type="text", text=f"Error: File not found: {fp}")]

        start = _time.time()
        emb_cfg = client.embedding_config
        ingestor = DocumentIngestor(
            embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
            passage_prefix=emb_cfg.get("passage_prefix", ""),
        )
        chunks, embeddings = ingestor.ingest_document(fp)

        # Index into searcher
        _, searcher = load_trove_client(client_name)
        searcher.index_chunks(chunks, embeddings)

        elapsed = _time.time() - start

        # Invalidate cache entry so it reloads
        _cache.put(client_name, client, searcher)

        return [TextContent(
            type="text",
            text=f"Ingested **{fp.name}**: {len(chunks)} chunks in {elapsed:.1f}s",
        )]
    except Exception as e:
        logger.exception("Ingest file error")
        return [TextContent(type="text", text=f"Ingest error: {e}")]


def handle_ingest_url(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    url = arguments.get("url")

    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]
    if not url:
        return [TextContent(type="text", text="Error: url is required")]

    # Validate URL scheme
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return [TextContent(type="text", text=f"Error: Only http/https URLs supported, got: {parsed.scheme}")]

    try:
        import tempfile
        import time as _time
        import requests as req
        from core.ingest import DocumentIngestor

        start = _time.time()
        client = load_client(client_name, clients_dir=CLIENTS_DIR)

        # Fetch URL
        resp = req.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        # Check size (50MB max)
        content_length = int(resp.headers.get("content-length", 0))
        if content_length > 50 * 1024 * 1024:
            return [TextContent(type="text", text=f"Error: Content too large ({content_length} bytes, max 50MB)")]

        content = resp.content
        if len(content) > 50 * 1024 * 1024:
            return [TextContent(type="text", text=f"Error: Content too large ({len(content)} bytes, max 50MB)")]

        content_type = resp.headers.get("content-type", "")

        # Save to temp file
        suffix = ".html" if "html" in content_type else ".pdf" if "pdf" in content_type else ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            emb_cfg = client.embedding_config
            ingestor = DocumentIngestor(
                embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
                passage_prefix=emb_cfg.get("passage_prefix", ""),
            )
            chunks, embeddings = ingestor.ingest_document(tmp_path)

            # Add URL as source
            for chunk in chunks:
                chunk["metadata"]["source_url"] = url

            _, searcher = load_trove_client(client_name)
            searcher.index_chunks(chunks, embeddings)
            _cache.put(client_name, client, searcher)

            elapsed = _time.time() - start
            return [TextContent(
                type="text",
                text=f"Ingested URL: {len(chunks)} chunks, content-type: {content_type}, time: {elapsed:.1f}s",
            )]
        finally:
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Ingest URL error")
        return [TextContent(type="text", text=f"Ingest error: {e}")]


def handle_reindex_client(arguments: dict) -> list[TextContent]:
    client_name = arguments.get("client_name")
    if not client_name:
        return [TextContent(type="text", text="Error: client_name is required")]

    try:
        job = _job_manager.create_job(client_name, "reindex")
    except RuntimeError as e:
        return [TextContent(type="text", text=f"Error: {e}")]

    def _run_reindex(job_id: str, cname: str):
        try:
            _job_manager.start_job(job_id)
            from core.ingest import DocumentIngestor, ingest_directory, ingest_client_code, build_file_manifest, save_file_manifest
            from core.ingest_progress import IngestionProgress

            client = load_client(cname, clients_dir=CLIENTS_DIR)
            emb_cfg = client.embedding_config
            ingestor = DocumentIngestor(
                embedding_model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
                passage_prefix=emb_cfg.get("passage_prefix", ""),
            )

            def _progress_cb(progress: IngestionProgress):
                _job_manager.update_progress(job_id, progress)

            doc_extensions = client.get_document_extensions()
            doc_exclude_patterns = client.documents_config.get("exclude_patterns", [])
            chunks, embeddings = ingest_directory(
                client.documents_path, ingestor,
                extensions=doc_extensions,
                progress_callback=_progress_cb,
                exclude_patterns=doc_exclude_patterns,
            )

            if chunks:
                searcher = HybridSearcher(
                    index_path=str(client.index_path),
                    collection_name=client.machine_name,
                    embedding_config=emb_cfg,
                    tokenizer_config=client.tokenization_config,
                )
                # Clear existing
                try:
                    searcher.chroma_client.delete_collection(client.machine_name)
                    searcher.collection = searcher.chroma_client.get_or_create_collection(
                        name=client.machine_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception:
                    pass
                searcher.index_chunks(chunks, embeddings)

                # Also index code if enabled
                code_chunks, code_embeddings = ingest_client_code(client, ingestor, rebuild=True)
                if code_chunks:
                    searcher.index_chunks(code_chunks, code_embeddings)

                # Save manifest
                manifest = build_file_manifest(client.documents_path, doc_extensions, exclude_patterns=doc_exclude_patterns)
                save_file_manifest(client, manifest)

                # Invalidate searcher cache
                _cache.put(cname, client, searcher)

                total = len(chunks) + len(code_chunks)
                _job_manager.complete_job(job_id, f"Reindexed: {total} chunks ({len(chunks)} docs, {len(code_chunks)} code)")
            else:
                _job_manager.complete_job(job_id, "No documents found to index")

        except Exception as e:
            logger.exception(f"Reindex job {job_id} failed")
            _job_manager.fail_job(job_id, str(e))

    t = threading.Thread(target=_run_reindex, args=(job.job_id, client_name), daemon=True)
    t.start()

    return [TextContent(
        type="text",
        text=f"Reindex started for **{client_name}**. Job ID: `{job.job_id}`\n\nUse `trove_ingest_status` to check progress.",
    )]


def handle_ingest_status(arguments: dict) -> list[TextContent]:
    job_id = arguments.get("job_id")
    client_name = arguments.get("client_name")

    job = None
    if job_id:
        job = _job_manager.get_job(job_id)
    elif client_name:
        jobs = _job_manager.get_client_jobs(client_name)
        job = jobs[0] if jobs else None

    if not job:
        return [TextContent(type="text", text="No ingestion job found.")]

    d = job.to_dict()
    lines = [f"**Job {d['job_id']}** ({d['job_type']} for {d['client_name']})"]
    lines.append(f"**Status:** {d['status']}")

    if d.get("progress"):
        p = d["progress"]
        lines.append(f"**Progress:** {p['percent']}% ({p['processed_files']}/{p['total_files']} files)")
        lines.append(f"**Chunks:** {p['total_chunks']}")
        lines.append(f"**ETA:** {p['eta']}")
        if p.get("failed_files"):
            lines.append(f"**Failed files:** {len(p['failed_files'])}")

    if d.get("result"):
        lines.append(f"**Result:** {d['result']}")
    if d.get("error"):
        lines.append(f"**Error:** {d['error']}")

    return [TextContent(type="text", text="\n".join(lines))]
