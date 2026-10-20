---
description: Get a quick overview of Trove search results with summaries
user-invocable: true
args: query
---

# Trove Overview

<skill-content>
Get a quick overview of documentation matching: {query}

## Instructions

1. If no client is loaded, list clients and pick the most relevant:
   ```
   trove_list_clients()
   ```

2. Get summarized results:
   ```
   trove_summarize(query="{query}", top_k=5)
   ```

3. Present the summaries to the user with source citations

4. If the user wants more detail on a specific result, follow up with a full search:
   ```
   trove_search(query="<refined query based on summary>", top_k=10, mode="precise")
   ```

## When to Use

- When the user wants a quick scan of what's available on a topic
- Before diving deep into a specific area
- When exploring unfamiliar documentation
- When context window budget is limited
</skill-content>
