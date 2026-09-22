"""WSCI steps 3-6: select, compress, write, and isolate context."""

import json
import os
from pathlib import Path
from typing import Any

from ollama import chat

MODEL = os.getenv("WSCI_MODEL", "qwen3:4b")
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STATE_FILE = BASE_DIR / "state.json"

question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
""".strip()

ROUTES = {
    "wifi_setup.txt": ("wi-fi", "wifi", "wireless", "eduroam", "connect"),
    "password_changes.txt": ("password", "credential", "login", "sign in"),
    "service_status.txt": ("status", "outage", "operational", "still works"),
    "email_setup.txt": ("email", "mail", "webmail", "inbox"),
    "vpn.txt": ("vpn", "remote access"),
    "printing.txt": ("print", "printer"),
    "classroom_projectors.txt": ("projector", "display", "hdmi", "usb-c"),
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "likely_cause": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "warning": {"type": "string"},
        "escalation": {"type": "string"},
    },
    "required": ["summary", "likely_cause", "steps", "warning", "escalation"],
}


def select_context(user_question: str) -> list[Path]:
    """SELECT: return knowledge files whose route keywords occur in the query."""
    normalized = user_question.casefold()
    selected = [
        KNOWLEDGE_DIR / filename
        for filename, keywords in ROUTES.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return selected or sorted(KNOWLEDGE_DIR.glob("*.txt"))


def read_context(files: list[Path]) -> str:
    return "\n\n".join(
        f"SOURCE: {file.name}\n{file.read_text(encoding='utf-8')}" for file in files
    )


def compress_context(context: str, user_question: str) -> str:
    """COMPRESS: ask Qwen to keep only information relevant to the query."""
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Compress the supplied IT-support context. Keep only facts "
                    "that directly help answer the question. Preserve concrete "
                    "status information, warnings, and ordered troubleshooting "
                    "steps. Do not answer the question and do not add facts."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION:\n{user_question}\n\nCONTEXT:\n{context}",
            },
        ],
        options={"temperature": 0},
    )
    return response.message.content.strip()


def isolated_context(state: dict[str, Any], task: str) -> dict[str, Any]:
    """ISOLATE: expose only the state needed for the current task."""
    if task == "diagnostic":
        return {
            "diagnostic_context": state["diagnostic_context"],
            "resolution_context": state.get("resolution_context", {}),
        }
    if task == "report":
        return {"report_context": state["report_context"]}
    raise ValueError(f"Unknown task: {task}")


def save_state(
    compressed_context: str,
    answer: dict[str, Any],
    selected_files: list[Path],
    raw_context: str,
) -> None:
    """WRITE: persist structured, separately addressable state artifacts."""
    state = {
        "diagnostic_context": {
            "problem": question,
            "device": "Windows laptop",
            "service": "eduroam campus Wi-Fi",
            "wifi_status": "operational",
            "relevant_knowledge": compressed_context,
        },
        "resolution_context": answer,
        "report_context": {
            "selected_file_count": len(selected_files),
            "selected_files": [file.name for file in selected_files],
            "raw_context_characters": len(raw_context),
            "compressed_context_characters": len(compressed_context),
        },
    }
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


selected_files = select_context(question)
context = read_context(selected_files)
compressed_context = compress_context(context, question)

initial_state = {
    "diagnostic_context": {
        "problem": question,
        "device": "Windows laptop",
        "service": "eduroam campus Wi-Fi",
        "wifi_status": "operational",
        "relevant_knowledge": compressed_context,
    },
    "resolution_context": {},
    "report_context": {
        "selected_file_count": len(selected_files),
        "selected_files": [file.name for file in selected_files],
        "raw_context_characters": len(context),
        "compressed_context_characters": len(compressed_context),
    },
}

diagnostic_state = isolated_context(initial_state, "diagnostic")
response = chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT support assistant. Use only the "
                "isolated diagnostic state. Return valid JSON matching the "
                "requested schema. Give concise, safe, ordered steps and do not "
                "invent facts."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(diagnostic_state, ensure_ascii=False, indent=2),
        },
    ],
    format=ANSWER_SCHEMA,
    options={"temperature": 0},
)

answer = json.loads(response.message.content)
save_state(compressed_context, answer, selected_files, context)

saved_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
print("Selected files:", ", ".join(file.name for file in selected_files))
print("Raw context characters:", len(context))
print("Compressed context characters:", len(compressed_context))
print(json.dumps(isolated_context(saved_state, "diagnostic"), indent=2, ensure_ascii=False))
print(f"State written to: {STATE_FILE.name}")
