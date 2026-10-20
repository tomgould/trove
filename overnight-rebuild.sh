#!/usr/bin/env bash
# Nightly Trove update: reindex changes + update knowledge graphs
#
# Usage:
#   ./overnight-rebuild.sh            # incremental (changes only)
#   ./overnight-rebuild.sh --rebuild   # full rebuild from scratch
#
# Graph builds need a local LLM: LM Studio running with the model named in each
# client's client.yaml loaded. Graph builds are slow (roughly a minute per batch
# of 10 document chunks on a laptop GPU), so pick the clients you need.
#
# Edit REINDEX_CLIENTS and GRAPH_CLIENTS below to list your own clients, one per
# line (the folder names under clients/).
set -euo pipefail

cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
LOG="overnight-rebuild.log"
> "$LOG"

# Check for --rebuild flag
REBUILD=""
if [[ "${1:-}" == "--rebuild" ]]; then
    REBUILD="--rebuild"
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [[ -n "$REBUILD" ]]; then
    log "=== Starting full Trove rebuild ==="
else
    log "=== Starting incremental Trove update ==="
fi

# Log system resources
python -c "from core.resource_manager import ResourceManager; print(ResourceManager().summary())" 2>/dev/null | tee -a "$LOG" || true

# ============================================================
# Phase 0: SOURCE SYNC
# Sync configured sources (GitHub repos, web crawls) before reindex
# ============================================================
log "--- Phase 0: Source Sync ---"
if python scripts/sources.py sync --all 2>&1 | tee -a "$LOG"; then
    log "Done: source sync"
else
    log "WARN: Source sync failed, continuing with reindex..."
fi
echo

# ============================================================
# Phase 1: REINDEX
# Full rebuild times measured 2026-03-08 overnight run
# ============================================================
# Phase 1: REINDEX
REINDEX_CLIENTS=(
    example
    # myproject
)

# ============================================================
# Phase 2: KNOWLEDGE GRAPHS
# Needs graph.enabled in the client's client.yaml and LM Studio running.
# Use "python scripts/build_graph.py --client NAME --dry-run" to see how many
# chunks a build would process before committing to it.
# ============================================================
GRAPH_CLIENTS=(
    # myproject
)

# ============================================================

log "--- Phase 1: Reindexing ---"
for client in "${REINDEX_CLIENTS[@]}"; do
    log "Reindexing: $client"
    if python -u scripts/ingest.py --client "$client" $REBUILD 2>&1 | tee -a "$LOG"; then
        log "Done: $client"
    else
        log "WARN: Reindex failed for $client, continuing..."
    fi
    echo
done
echo

log "--- Phase 2: Knowledge Graph Builds ---"
# Health gate — verify LM Studio has a model loaded (not just responding)
if ! python -c "from core.health import check_lm_studio_model; exit(0 if check_lm_studio_model() else 1)" 2>/dev/null; then
    log "ERROR: LM Studio not serving a model. Skipping graph builds."
else
    for client in "${GRAPH_CLIENTS[@]}"; do
        log "Building graph: $client"
        if python -u scripts/build_graph.py --client "$client" $REBUILD 2>&1 | tee -a "$LOG"; then
            log "Done: $client"
        else
            log "WARN: Graph build failed for $client, continuing..."
        fi
        echo
    done
fi

log "=== Overnight run complete ==="
log "Check $LOG for full output"
