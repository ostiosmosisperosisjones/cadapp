# cadapp Agent Instructions

## Local LLM Delegation
A local LLM is available via the `local-llm` MCP server. Use it to save tokens on routine tasks.

**Delegate to local-llm when:**
- Writing or editing a single self-contained file
- Generating boilerplate, docstrings, or tests for existing functions
- Simple refactors within one file
- Any grep/search task (use search_codebase)

**Handle yourself when:**
- Changes span multiple files
- The task requires understanding the full architecture
- The local LLM's output looks wrong or needs significant correction

## Search Strategy
- Always prefer `search_codebase` over Explore for finding things in the codebase
- Read only files directly relevant to the task
- Do not attempt to load the full codebase into context

## Project Notes
- This is a CAD application written in Python
