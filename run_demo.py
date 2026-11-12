"""One-click demo run (VS Code: open this file and press the Run ▶ button).

Runs the agent with the demo request on 5 works and prints how long it took.
Same as: python main.py --profile profiles/moscardo.json --query "libros escritos por Leila Guerriero" --author "Leila Guerriero" --max-works 5
"""

import time

from editorial_agent.cli import main

if __name__ == "__main__":
    start = time.perf_counter()
    code = main([
        "--profile", "profiles/moscardo.json",
        "--query", "libros escritos por Leila Guerriero",
        "--author", "Leila Guerriero",
        "--max-works", "5",
    ])
    print(f"Duración: {time.perf_counter() - start:.0f} s")
    raise SystemExit(code)
