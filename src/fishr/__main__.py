from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "chat":
        _run_chat(args[1:])
    else:
        print(f"unknown command: {args[0]}")
        print("usage: fishr chat [--host HOST] [--port PORT]")
        sys.exit(1)


def _run_chat(extra: list[str]) -> None:
    import argparse

    from fishr.Loop import asyncio

    parser = argparse.ArgumentParser(
        prog="fishr chat", description="fishr chat - local LLM playground"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    namespace = parser.parse_args(extra)

    from fishr.chat import Run

    try:
        asyncio.run(Run(host=namespace.host, port=namespace.port))
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
