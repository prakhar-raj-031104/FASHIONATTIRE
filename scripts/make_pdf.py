#!/usr/bin/env python3
"""Render docs/WRITEUP.md into the single submission PDF.

    python scripts/make_pdf.py [--repo-url https://github.com/you/repo]

Markdown -> styled HTML -> PDF via headless Chrome (no pandoc/LaTeX needed).
The --repo-url is substituted into the write-up's GitHub-link placeholder.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MD = REPO / "docs" / "WRITEUP.md"
HTML = REPO / "docs" / "writeup.html"
PDF = REPO / "docs" / "Glance_ML_Assignment_Writeup.pdf"

CSS = """
/* Tuned so the full write-up — every deliverable retained — fits 5 A4 pages. */
@page { size: A4; margin: 13mm 12mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       font-size: 10.4pt; line-height: 1.42; color: #1b1f24; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 2px; color: #0b1f3a; letter-spacing: -0.3px; }
h2 { font-size: 13.2pt; margin: 14px 0 5px; padding-bottom: 3px;
     border-bottom: 1.5px solid #d7dde5; color: #0b1f3a; page-break-after: avoid; }
h3 { font-size: 11.3pt; margin: 10px 0 4px; color: #14355f; page-break-after: avoid; }
h4 { font-size: 10.6pt; margin: 8px 0 3px; color: #14355f; }
p, li { margin: 4px 0; }
ul, ol { padding-left: 19px; margin: 4px 0; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: #f2f4f7; padding: 1px 4px; border-radius: 3px; color: #8a2f6b; }
pre { background: #0f1620; color: #e6edf3; padding: 9px 12px; border-radius: 6px;
      overflow-x: auto; font-size: 8.7pt; line-height: 1.4; margin: 7px 0;
      page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 9.1pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d7dde5; padding: 3.5px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 700; }
blockquote { border-left: 2px solid #2f81f7; margin: 5px 0; padding: 2px 9px;
             background: #f5f9ff; color: #33404f; }
hr { border: none; border-top: 1px solid #d7dde5; margin: 7px 0; }
strong { color: #0b1f3a; }
a { color: #1a63c4; text-decoration: none; }
"""


def build_html(repo_url: str | None) -> str:
    import markdown

    text = MD.read_text(encoding="utf-8")
    if repo_url:
        text = text.replace("<GITHUB_REPO_URL>", repo_url)
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Glance ML Assignment — Multimodal Fashion &amp; Context Retrieval</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default=None, help="GitHub repo URL to embed")
    args = ap.parse_args()

    if not MD.exists():
        print(f"missing {MD}", file=sys.stderr)
        return 1

    HTML.write_text(build_html(args.repo_url), encoding="utf-8")
    print(f"wrote {HTML}")

    chrome = next(
        (c for c in ("google-chrome", "chromium", "chromium-browser", "chrome")
         if shutil.which(c)), None
    )
    if not chrome:
        print("No Chrome/Chromium found — open the HTML and use Print → Save as PDF.")
        return 0

    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if PDF.exists():
        print(f"wrote {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")
        return 0
    print("PDF generation failed:\n", res.stderr[-1500:], file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
