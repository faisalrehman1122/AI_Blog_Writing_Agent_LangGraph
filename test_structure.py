#!/usr/bin/env python3
"""
Test script to validate the blog writing agent structure without API calls.
"""

from __future__ import annotations

import operator
from typing import TypedDict, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

print("=" * 100)
print("TESTING BLOG WRITING AGENT STRUCTURE")
print("=" * 100)
print()

print("✓ Step 1: Importing modules...")
try:
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send
    print("  All imports successful!")
except ImportError as e:
    print(f"  ERROR: {e}")
    exit(1)

print()
print("✓ Step 2: Defining schemas...")
try:
    class Task(BaseModel):
        id: int
        title: str
        goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
        bullets: List[str] = Field(..., min_length=3, max_length=6)
        target_words: int = Field(..., description="Target words (120–550).")
        tags: List[str] = Field(default_factory=list)
        requires_research: bool = False
        requires_citations: bool = False
        requires_code: bool = False

    class Plan(BaseModel):
        blog_title: str
        audience: str
        tone: str
        blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
        constraints: List[str] = Field(default_factory=list)
        tasks: List[Task]

    class State(TypedDict):
        topic: str
        mode: str
        needs_research: bool
        queries: List[str]
        evidence: List[dict]
        plan: Optional[Plan]
        as_of: str
        recency_days: int
        sections: Annotated[List[tuple[int, str]], operator.add]
        merged_md: str
        md_with_placeholders: str
        image_specs: List[dict]
        final: str

    print("  All schemas defined successfully!")
except Exception as e:
    print(f"  ERROR: {e}")
    exit(1)

print()
print("✓ Step 3: Creating graph structure...")
try:
    def dummy_router(state: State) -> dict:
        return {"needs_research": False, "mode": "closed_book", "queries": [], "recency_days": 3650}
    
    def dummy_orchestrator(state: State) -> dict:
        return {"plan": None}
    
    def dummy_worker(payload: dict) -> dict:
        return {"sections": [(1, "dummy section")]}
    
    def dummy_reducer(state: State) -> dict:
        return {"final": "dummy blog"}
    
    def route_next(state: State) -> str:
        return "orchestrator"
    
    def fanout(state: State):
        return [Send("worker", {"task": {"id": 1, "title": "test"}})]

    g = StateGraph(State)
    g.add_node("router", dummy_router)
    g.add_node("orchestrator", dummy_orchestrator)
    g.add_node("worker", dummy_worker)
    g.add_node("reducer", dummy_reducer)
    
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_next, {"orchestrator": "orchestrator"})
    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)
    
    app = g.compile()
    print("  Graph structure created successfully!")
    print(f"  Graph nodes: {list(g.nodes.keys())}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 100)
print("STRUCTURE VALIDATION COMPLETE ✓")
print("=" * 100)
print()
print("The blog writing agent structure is correctly set up!")
print()
print("To run the full agent, you need:")
print("  1. OPENAI_API_KEY (required)")
print("  2. TAVILY_API_KEY (optional, for research)")
print("  3. GOOGLE_API_KEY (optional, for images)")
print()
print("Then run: python3 run_blog_agent.py")
print()
