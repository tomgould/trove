---
description: Quick search Trove docs by client and query
user-invocable: true
args: client_name, query
---

# Trove Quick Search

<skill-content>
Search the Trove documentation for client **{client_name}** with query: {query}

## Instructions

1. If `{client_name}` is unknown or unclear, list available clients first:
   ```
   trove_list_clients()
   ```

2. Search Trove for the query:
   ```
   trove_search_client(client_name="{client_name}", query="{query}", top_k=10)
   ```

3. Present the relevant information from the search results

4. Include any code examples found

5. Cite which client and document the information comes from

## Tips

- If the initial search returns few results, try `mode="broad"` for wider semantic matching
- For exact term lookups, use `mode="precise"`
- For code-related queries, use `mode="code_first"`
- For entity/relationship queries, use `mode="graph_mix"` (requires knowledge graph)
- To explore entities directly, use `trove_search_entities(query="...", client_name="{client_name}")`
- To limit context size, add `max_tokens=2000`
- For a quick overview first, use `trove_summarize(query="{query}", client_name="{client_name}", top_k=5)`
- To ingest a new file, use `trove_ingest_file(client_name="{client_name}", file_path="...")`
- Use `trove_list_clients()` any time you're unsure which client to target
</skill-content>
