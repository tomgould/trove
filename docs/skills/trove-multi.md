---
description: Search across all Trove clients at once
user-invocable: true
args: query
---

# Trove Multi-Client Search

<skill-content>
Search across all Trove documentation clients for: {query}

## Instructions

1. Search all clients simultaneously:
   ```
   trove_search_multi(query="{query}", top_k=20)
   ```

   Or specify particular clients:
   ```
   trove_search_multi(query="{query}", clients=["client_a", "client_b"], top_k=20)
   ```

2. Results are merged using rank fusion and tagged with their source client

3. Present results grouped or ordered by relevance, noting which client each result comes from

4. If one client dominates the results, suggest the user search that client directly for deeper coverage

## When to Use

- When the user's question could span multiple projects
- When unsure which client has the relevant documentation
- For comparing how different projects handle the same concept
- For getting a broad view across all indexed knowledge
</skill-content>
