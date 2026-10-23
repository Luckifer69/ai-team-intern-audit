#!/usr/bin/env python3
"""Render a memo .md to an A4 PDF (11 pt, 2 cm margins) and print its page count.

Used to check the "<= 1 page" limit. Needs the `markdown` package and LibreOffice (soffice).
GitHub-style lists (no blank line before them, 2-space nesting) are adapted for python-markdown.

Usage: python -I tools/render_memo.py partA/memo.md partC/memo.md [--out-dir rendered/]
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

CSS = ("@page{size:A4;margin:2cm} body{font-family:'DejaVu Sans',sans-serif;font-size:11pt} "
       "ul,ol{margin:2px 0;padding-left:1.4em} li{margin:0} p{margin:5px 0} h2{font-size:14pt;margin:0 0 4px} "
       "table{border-collapse:collapse;font-size:9.5pt} td,th{border:1px solid #999;padding:2px;white-space:nowrap}")


def to_html(md_text):
    is_item = lambda l: re.match(r"\s*(- |\d+\. )", l) is not None
    out = []
    for l in md_text.split("\n"):
        if re.match(r"^ +(- |\d+\. )", l):            # nested item: python-markdown needs 4 spaces
            l = "    " + l.lstrip()
        elif re.match(r"^ {2,3}\S", l) and out and (is_item(out[-1]) or out[-1].startswith("    ")):
            l = ("    " if out[-1].startswith("    ") and is_item(out[-1]) else "  ") + l.lstrip()
        if is_item(l) and out and out[-1].strip() and not is_item(out[-1]) and not out[-1].startswith(" "):
            out.append("")                               # list right after a paragraph
        out.append(l)
    body = markdown.markdown("\n".join(out), extensions=["tables"])
    # LibreOffice splits table columns evenly unless told otherwise; size the label column so
    # numeric cells don't wrap (GitHub-style renderers do this automatically).
    body = re.sub(r"<tr>\s*<(t[hd])>", r'<tr>\n<\1 width="40%">', body)
    return f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body>{body}</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("memos", nargs="+")
    ap.add_argument("--out-dir", default=None, help="keep the PDFs here (default: temp dir)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(args.out_dir or tmp)
        out.mkdir(parents=True, exist_ok=True)
        for m in map(Path, args.memos):
            html = out / f"{m.parent.name}_{m.stem}.html"
            html.write_text(to_html(m.read_text(encoding="utf-8")), encoding="utf-8")
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf:writer_web_pdf_Export",
                            "--outdir", str(out), str(html)], check=True, capture_output=True, timeout=180)
            pdf = html.with_suffix(".pdf")
            info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
            pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
            print(f"{m}: {pages} page(s) {'OK' if pages == 1 else 'OVER LIMIT'}  ({len(m.read_text().split())} words)")
            if not args.out_dir:
                html.unlink()


if __name__ == "__main__":
    main()
