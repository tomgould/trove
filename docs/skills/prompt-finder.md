---
description: Find the best AI prompt template for any task from 10k+ indexed prompts and live prompts.chat API
user-invocable: true
args: task_description
---

# Prompt Finder

<skill-content>
Find the best prompt template for: **{task_description}**

## Instructions

1. **Combined search** — search both local index and live API:
   ```
   trove_prompts_find(query="{task_description}", top_k=10)
   ```

2. **Review results** and identify the most relevant prompt(s). If the task involves a specific type, add the type filter:
   - For image generation prompts: `type="IMAGE"`
   - For video prompts: `type="VIDEO"`
   - For skill/workflow prompts: `type="SKILL"`

3. **Fetch the full prompt** for the best match:
   ```
   trove_prompts_get(prompt_id="<id from results>")
   ```

4. **Present the prompt** to the user:
   - Show the full prompt content
   - If it has `${Variable:Default}` placeholders, suggest appropriate values for the user's task
   - Fetch with substitutions if needed: `trove_prompts_get(prompt_id="<id>", variables={"Var": "value"})`

5. **If results are insufficient**, try targeted searches:
   - Browse trending: `trove_prompts_search_live(sort="popular", limit=10)`
   - Browse by category: `trove_prompts_search_live(category="web-development", limit=10)`
   - Browse newest: `trove_prompts_search_live(sort="newest", limit=10)`
   - Broader local search: `trove_search_client(client_name="prompts", query="{task_description}", mode="broad", top_k=15)`

## Tips

- The local index has 9,970+ prompt chunks with deep semantic matching
- The live API has the latest community contributions and vote counts
- Combined search deduplicates across both sources
- For code/dev prompts, try queries like "react reviewer", "python debugging", "code architect"
- For creative prompts, try "midjourney", "dall-e", "creative writing", "storyteller"
- Prompt IDs look like `cmj1zb0yc000ivl0rc9ka596e`
</skill-content>
