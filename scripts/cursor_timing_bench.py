#!/usr/bin/env python3
"""Controlled cursor-agent timing benchmarks (Hermes CursorAgentClient).

Runs three scenarios and prints cursor-timing / cursor-timing-tool lines.
Requires CURSOR_API_KEY in ~/.hermes/.env (or environment).

Usage:
  cd ~/.hermes/hermes-agent && python3 scripts/cursor_timing_bench.py
  HERMES_CURSOR_TIMING_TOOLS=1 python3 scripts/cursor_timing_bench.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ENV = Path.home() / ".hermes" / ".env"


def _load_dotenv() -> None:
    if not _ENV.is_file():
        return
    for line in _ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _run_case(
    client,
    *,
    label: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
) -> dict:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
    except Exception as exc:
        wall_ms = int((time.monotonic() - t0) * 1000)
        print(f"FAIL wall_ms={wall_ms} error={exc!r}", flush=True)
        return {"label": label, "ok": False, "error": str(exc), "wall_ms": wall_ms}

    wall_ms = int((time.monotonic() - t0) * 1000)
    choice = resp.choices[0]
    msg = choice.message
    internal = getattr(msg, "cursor_internal_tools", None) or []
    usage = getattr(resp, "usage", None)
    finish = getattr(choice, "finish_reason", "?")
    content_len = len(getattr(msg, "content", "") or "")
    tool_calls = getattr(msg, "tool_calls", None) or []
    print(
        f"OK wall_ms={wall_ms} finish={finish} "
        f"cursor_internal_tools={len(internal)} hermes_tool_calls={len(tool_calls)} "
        f"content_chars={content_len} "
        f"prompt_tokens={getattr(usage, 'prompt_tokens', '?')}",
        flush=True,
    )
    return {
        "label": label,
        "ok": True,
        "wall_ms": wall_ms,
        "finish": finish,
        "cursor_tools": len(internal),
        "hermes_tool_calls": len(tool_calls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cursor timing benchmark harness")
    parser.add_argument(
        "--model",
        default=os.getenv("HERMES_CURSOR_BENCH_MODEL", "composer-2.5"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set HERMES_CURSOR_TIMING_VERBOSE=1 for milestone lines",
    )
    parser.add_argument(
        "--tools-detail",
        action="store_true",
        help="Set HERMES_CURSOR_TIMING_TOOLS=1 for per-tool lines",
    )
    parser.add_argument(
        "--repeat-warm",
        action="store_true",
        help="Run the light-read scenario twice (cold then warm workspace)",
    )
    args = parser.parse_args()

    _load_dotenv()
    os.environ.setdefault("HERMES_CURSOR_TIMING", "1")
    if args.verbose:
        os.environ["HERMES_CURSOR_TIMING_VERBOSE"] = "1"
    if args.tools_detail:
        os.environ["HERMES_CURSOR_TIMING_TOOLS"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from agent.cursor_agent_client import CursorAgentClient

    api_key = os.getenv("CURSOR_API_KEY", "").strip()
    if not api_key:
        print("ERROR: CURSOR_API_KEY not set (check ~/.hermes/.env)", file=sys.stderr)
        return 1

    # Aux calls: tiny ephemeral workspace (fast). Tool cases: real repo so
    # read/grep find files (empty temp dirs cause ~25s glob/grep timeouts).
    client_aux = CursorAgentClient(api_key=api_key)
    client_repo = CursorAgentClient(api_key=api_key, workspace=str(_REPO))
    results: list[dict] = []

    # 1) Auxiliary-style: no tools, tiny prompt (floor latency)
    results.append(
        _run_case(
            client_aux,
            label="aux_no_tools",
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": "Reply with exactly one word: pong. No tools.",
                },
                {"role": "user", "content": "ping"},
            ],
            tools=None,
        )
    )

    # 2) Light agentic: one built-in read via cursor workspace file
    bench_file = _REPO / "scripts" / ".cursor_bench_target.txt"
    bench_file.write_text("bench-marker-line-42\n", encoding="utf-8")
    rel = "scripts/.cursor_bench_target.txt"
    results.append(
        _run_case(
            client_repo,
            label="light_read_one_file",
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Use your built-in read_file on the workspace file "
                        f"'{rel}' and report the exact line content. One tool max."
                    ),
                },
                {"role": "user", "content": f"Read {rel} and quote the line."},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "noop_hermes_tool",
                        "description": "unused placeholder",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )

    # 3) Medium: grep + read in repo (few tools, not 30)
    results.append(
        _run_case(
            client_repo,
            label="medium_grep_read",
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "In this workspace (hermes-agent repo root), grep for "
                        "'_CursorCallTiming' in agent/cursor_agent_client.py only, "
                        "then read ~40 lines around the class. Built-in tools only; "
                        "do not search outside the workspace. Stop after summary."
                    ),
                },
                {
                    "role": "user",
                    "content": "Find _CursorCallTiming and summarize it briefly.",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "noop_hermes_tool",
                        "description": "unused",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )

    if args.repeat_warm:
        results.append(
            _run_case(
                client_repo,
                label="warm_repeat_light_read",
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Read workspace file '{rel}' again and quote the line."
                        ),
                    },
                    {"role": "user", "content": "Same read as before."},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "noop_hermes_tool",
                            "description": "unused",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )

    client_aux.close()
    client_repo.close()

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        status = "ok" if r.get("ok") else "FAIL"
        extra = ""
        if r.get("ok"):
            extra = (
                f" wall_ms={r['wall_ms']} cursor_tools={r.get('cursor_tools', '?')} "
                f"finish={r.get('finish', '?')}"
            )
        else:
            extra = f" {r.get('error', '')}"
        print(f"  {r['label']}: {status}{extra}", flush=True)

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
