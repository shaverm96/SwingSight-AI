from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(ROOT)

from swingsight.config import load_dotenv

load_dotenv(ROOT / ".env")

from webapp import create_app


def main() -> None:
    app = create_app()
    host = os.environ.get("SWINGSIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("SWINGSIGHT_PORT", "8000"))
    debug = os.environ.get("SWINGSIGHT_DEBUG", "true").lower() not in {"0", "false", "no"}
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
