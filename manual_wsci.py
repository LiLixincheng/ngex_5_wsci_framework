"""WSCI step 2: manually select only the relevant knowledge files."""

import os
from pathlib import Path

from ollama import chat

MODEL = os.getenv("WSCI_MODEL", "qwen3:4b")
BASE_DIR = Path(__file__).parent

question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
""".strip()

selected_files = [
    BASE_DIR / "knowledge" / "wifi_setup.txt",
    BASE_DIR / "knowledge" / "password_changes.txt",
    BASE_DIR / "knowledge" / "service_status.txt",
]

context = "\n\n".join(
    f"SOURCE: {file.name}\n{file.read_text(encoding='utf-8')}"
    for file in selected_files
)

response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT support assistant. Answer only from "
                "the selected context. Give concise, step-by-step advice.\n\n"
                f"SELECTED CONTEXT:\n{context}"
            ),
        },
        {"role": "user", "content": question},
    ],
)

print("Selected files:", ", ".join(file.name for file in selected_files))
print("Context characters:", len(context))
print(response.message.content)
