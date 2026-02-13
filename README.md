# AI Blog Writing Agent with LangGraph

An intelligent AI agent that automatically plans, researches, and writes technical blog posts using LangGraph. This agent uses a multi-node workflow to create complete, well-structured blog posts with optional research and image generation.

## 🚀 Features

- **Automatic Planning**: Creates detailed blog outlines with specific goals and bullet points
- **Intelligent Research**: Automatically performs web research when needed using Tavily API
- **Parallel Writing**: Multiple sections written simultaneously for efficiency
- **Citation Management**: Properly cites sources for research-backed claims
- **Image Generation**: Automatically generates diagrams when helpful using Google Gemini
- **Flexible Modes**: Adapts to evergreen vs. time-sensitive topics
- **Word Count Control**: Respects target word counts per section

## 📋 Requirements

### Required
- Python 3.12+
- OpenAI API key (for GPT models)

### Optional
- Tavily API key (for web research)
- Google API key (for image generation)

### Dependencies
```bash
pip install pydantic langgraph langchain-openai langchain-core langchain-community python-dotenv
```

## 🏗️ Architecture

The agent follows this execution flow:

```
START
  ↓
Router Node → Analyzes topic, decides if research is needed
  ↓
Research Node (if needed) → Web search with Tavily API
  ↓
Orchestrator Node → Creates detailed blog plan (5-9 sections)
  ↓
Fanout → Distributes tasks to worker nodes in parallel
  ↓
Worker Nodes → Write sections simultaneously
  ↓
Reducer Subgraph → Merges content, decides/generates images
  ↓
END → Saves markdown file
```

## 📥 Input

- **Topic**: A string describing what blog to write
  - Example: `"Write a blog on Self Attention"`
  - Example: `"State of Multimodal LLMs in 2026"`
- **Optional**: `as_of` date (ISO format) for time-sensitive topics

## 📤 Output

1. **Markdown File**: Complete blog post saved as `{blog_title}.md`
2. **Images Directory**: Generated diagrams saved in `images/` folder (if applicable)
3. **State Object**: Contains plan, evidence, final content, and execution details

## 🎯 Execution Modes

1. **closed_book**: Evergreen topics (no research needed)
2. **hybrid**: Topics needing up-to-date examples
3. **open_book**: Volatile topics requiring fresh data (news roundups)

## 🚀 Quick Start

### 1. Setup Environment

Create a `.env` file:
```env
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here  # optional
GOOGLE_API_KEY=your-key-here   # optional
```

### 2. Run the Notebook

```bash
jupyter notebook blog_writing_agent_complete.ipynb
```

Execute cells sequentially to see the step-by-step process.

### 3. Or Run the Script

```bash
python3 run_blog_agent.py
```

### 4. Test the Structure

```bash
python3 test_structure.py
```

## 📁 Project Structure

```
.
├── blog_writing_agent_complete.ipynb  # Main notebook with step-by-step execution
├── run_blog_agent.py                  # Standalone Python script
├── demo_run.py                         # Demo showing execution flow
├── test_structure.py                   # Structure validation script
├── PROJECT_SUMMARY.md                  # Detailed project documentation
└── README.md                           # This file
```

## 💡 Usage Example

```python
from run_blog_agent import run

# Run the agent
result = run("Write a blog on Self Attention")

# Access the results
print(result["final"])  # Final markdown content
print(result["plan"])    # Blog plan with all sections
print(result["evidence"]) # Research evidence (if any)
```

## 🔧 How It Works

1. **Router** analyzes the topic and decides if web research is needed
2. **Research** (if needed) searches the web using Tavily API
3. **Orchestrator** creates a detailed plan with 5-9 sections
4. **Workers** write each section in parallel
5. **Reducer** merges sections, decides on images, generates images, and saves the final blog

## 📝 Example Output

For input: `"Write a blog on Self Attention"`

**Output:**
- File: `self_attention_in_transformer_architecture.md`
- Content: Complete blog with introduction, core concepts, implementation, code examples, common mistakes, and conclusion
- Images: Optional diagrams showing attention mechanism flow

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

## 🔗 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Tavily API](https://tavily.com/)
- [Google Gemini API](https://ai.google.dev/)

## 👤 Author

**Faisal Rehman**

- GitHub: [@faisalrehman1122](https://github.com/faisalrehman1122)

## 🙏 Acknowledgments

This project is based on the "This AI Agent Plans, Researches & Writes Blogs Automatically using LangGraph | Agentic AI Project" tutorial.

---

⭐ If you find this project helpful, please consider giving it a star!
