from __future__ import annotations

from webapp import create_app


app = create_app()


if __name__ == "__main__":
    # Local-only development server. Keep this app off public networks.
    app.run(host="127.0.0.1", port=8000, debug=True)
