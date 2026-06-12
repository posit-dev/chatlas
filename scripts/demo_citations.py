"""
Demo of web search citations — run with:
  uv run python scripts/demo_citations.py
"""

from chatlas import ChatOpenAI, tool_web_search

chat = ChatOpenAI(model="gpt-4.1")
chat.register_tool(tool_web_search())
chat.chat("When was ggplot2 first released on CRAN, and who created it?", echo="all")

print("\n--- Turn contents ---\n")
turn = chat.get_last_turn()
assert turn is not None
for c in turn.contents:
    print(c)
