# Blog Writing Agent - Project Summary

## Overview

This project implements an AI agent that automatically plans, researches, and writes technical blog posts using LangGraph. The agent uses a multi-node workflow to create complete, well-structured blog posts with optional research and image generation.

## Input

**Primary Input:**
- **Topic**: A string describing what blog to write
  - Examples: 
    - `"Write a blog on Self Attention"`
    - `"State of Multimodal LLMs in 2026"`
    - `"Introduction to Transformer Architecture"`

**Optional Input:**
- **as_of**: Date string (ISO format: YYYY-MM-DD) for time-sensitive topics
  - Default: Current date
  - Used for filtering research results by recency

## Output

**Primary Output:**
1. **Markdown File**: A complete blog post saved as `{blog_title}.md` in the current directory
   - File name is generated from the blog title (sanitized)
   - Contains full blog content with proper Markdown formatting
   - Includes citations when research is performed

2. **Images Directory** (optional): If images are generated, they are saved in `images/` folder
   - Images are generated using Google's Gemini API
   - Maximum 3 images per blog
   - Images are automatically inserted into the markdown with proper formatting

**State Object** (returned from execution):
- `plan`: Complete blog plan with all sections/tasks
- `evidence`: List of web search results (if research was performed)
- `image_specs`: Specifications for generated images
- `final`: Final markdown content
- `mode`: Execution mode (closed_book/hybrid/open_book)
- `needs_research`: Whether research was performed
- `queries`: Search queries used (if research was performed)
- `sections`: List of all written sections

## Architecture & Workflow

The agent follows this execution flow:

```
START
  ↓
Router Node
  ├─→ Analyzes topic
  ├─→ Decides if research is needed
  └─→ Determines mode (closed_book/hybrid/open_book)
  ↓
Research Node (if needed)
  ├─→ Uses Tavily API to search web
  ├─→ Collects evidence with citations
  └─→ Filters by recency for time-sensitive topics
  ↓
Orchestrator Node
  ├─→ Creates detailed blog plan
  ├─→ Defines 5-9 sections with:
  │   - Goal (what reader should understand)
  │   - 3-6 concrete bullets to cover
  │   - Target word count (120-550 words)
  │   - Flags for research/citations/code needs
  └─→ Sets blog kind (explainer/tutorial/news_roundup/etc.)
  ↓
Fanout
  └─→ Distributes tasks to worker nodes in parallel
  ↓
Worker Nodes (parallel execution)
  ├─→ Each worker writes one section
  ├─→ Follows plan's bullets in order
  ├─→ Respects word count targets
  ├─→ Cites evidence when required
  └─→ Includes code snippets if needed
  ↓
Reducer Subgraph
  ├─→ Merge Content: Combines all sections
  ├─→ Decide Images: Determines if diagrams needed
  └─→ Generate & Place Images: Creates and inserts images
  ↓
END
  └─→ Saves final markdown file
```

## Execution Modes

1. **closed_book**: Evergreen topics that don't need recent information
   - No web research performed
   - Relies on LLM's training data
   - Example: "Write a blog on Self Attention"

2. **hybrid**: Topics needing some up-to-date examples but mostly evergreen
   - Performs web research for current examples/tools/models
   - Marks relevant sections with `requires_citations=True`
   - Example: "Latest Python Web Frameworks in 2025"

3. **open_book**: Volatile topics requiring fresh data
   - Performs extensive web research
   - Filters results by recency (default: last 7 days)
   - Sets blog_kind to "news_roundup"
   - Example: "AI News This Week"

## Files Created

1. **blog_writing_agent_complete.ipynb**: Main notebook with step-by-step execution
2. **run_blog_agent.py**: Standalone Python script to run the agent
3. **demo_run.py**: Demo script showing execution flow
4. **test_structure.py**: Validation script for structure testing

## Requirements

**Required:**
- Python 3.12+
- OpenAI API key (for GPT models)
- Dependencies: pydantic, langgraph, langchain-openai, langchain-core, langchain-community, python-dotenv

**Optional:**
- Tavily API key (for web research)
- Google API key (for image generation)

## Usage

### Option 1: Run Notebook
```bash
jupyter notebook blog_writing_agent_complete.ipynb
```
Then execute cells sequentially.

### Option 2: Run Python Script
```bash
python3 run_blog_agent.py
```

### Option 3: Use in Code
```python
from run_blog_agent import run

result = run("Write a blog on Self Attention")
print(result["final"])
```

## Example Output

For input: `"Write a blog on Self Attention"`

**Output:**
- File: `self_attention_in_transformer_architecture.md`
- Content: Complete blog with:
  - Introduction section
  - Core concept explanations
  - Implementation details
  - Code examples (if requires_code=True)
  - Common mistakes section
  - Conclusion
- Images: Optional diagrams showing attention mechanism flow

## Key Features

1. **Automatic Planning**: Creates detailed outline with specific goals and bullets
2. **Parallel Writing**: Multiple sections written simultaneously for efficiency
3. **Research Integration**: Automatically performs web research when needed
4. **Citation Management**: Properly cites sources for research-backed claims
5. **Image Generation**: Automatically generates diagrams when helpful
6. **Flexible Modes**: Adapts to evergreen vs. time-sensitive topics
7. **Word Count Control**: Respects target word counts per section

## Notes

- The agent is designed for technical blog posts
- It assumes the reader is a developer/technical audience
- Code snippets are included when `requires_code=True` in the plan
- Images are only generated if they materially improve understanding
- Research is performed automatically based on topic analysis
