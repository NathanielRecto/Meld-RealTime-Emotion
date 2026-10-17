"""Start the local web app, then open http://127.0.0.1:8000 in a browser.

    python scripts/serve.py

Loading and warming up the models takes about 20 s before the page connects.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("meldrt.server:app", host="127.0.0.1", port=8000)
