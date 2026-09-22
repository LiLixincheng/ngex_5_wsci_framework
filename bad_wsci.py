"""WSCI step 1: give the model every knowledge-base file."""

import os
from pathlib import Path

from ollama import chat

MODEL = os.getenv("WSCI_MODEL", "qwen3:4b")
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
""".strip()


def read_all_context() -> str:
    """Read every text file, deliberately including irrelevant information."""
    sections = []
    for file in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        sections.append(f"SOURCE: {file.name}\n{file.read_text(encoding='utf-8')}")
    return "\n\n".join(sections)


context = read_all_context()
response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT support assistant. Use the supplied "
                "knowledge base as your source of truth. Give concise, safe, "
                "step-by-step advice and do not invent unavailable facts.\n\n"
                f"KNOWLEDGE BASE:\n{context}"
            ),
        },
        {"role": "user", "content": question},
    ],
)

print("Context characters:", len(context))
print(response.message.content)
