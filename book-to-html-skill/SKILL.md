---
name: book-to-html
description: Convert multi-part markdown books into a single HTML file with sidebar chapter navigation, table of contents, progress bar, search, and responsive design. Use when the user has a markdown book (with chapters, parts, appendices) and wants a beautiful navigable HTML version. Supports Chinese and English books, code blocks, tables, math notation, and multi-file merging.
---

# Book to HTML

Convert markdown books into a polished HTML with chapter navigation.

## Quick Start

```bash
python3 <skill_dir>/scripts/build_book_html.py <input.md> -o <output.html>
```

## Usage Patterns

### Single markdown file → HTML

```bash
python3 <skill_dir>/scripts/build_book_html.py book.md -o book.html
```

### Multiple part files → merged HTML

```bash
python3 <skill_dir>/scripts/build_book_html.py part1.md part2.md part3.md -o book.html
```

### With custom title

```bash
python3 <skill_dir>/scripts/build_book_html.py book.md -o book.html --title "My Book Title"
```

## Features

- **Sidebar navigation** — auto-extracts chapters (第X章), sections (X.Y), parts (第X部分), appendices (A.1)
- **Table of contents** — full TOC at the top of the page
- **Search** — filter nav items by keyword
- **Progress bar** — shows reading position
- **Back to top** — floating button
- **Responsive** — mobile-friendly with collapsible sidebar
- **Print-ready** — optimized print CSS for PDF export
- **Syntax highlighting** — code blocks with dark theme
- **Chinese typography** — Noto Serif/Sans SC fonts

## Heading Structure Detection

The script auto-detects common book heading patterns:

| Pattern | Role | Example |
|---------|------|---------|
| `# 第X部分：...` | Part divider | `# 第一部分：基础与原理` |
| `# 第X章 ...` or `## 第X章 ...` | Chapter | `# 第一章 引言` |
| `## X.Y ...` or `### X.Y ...` | Section | `## 1.1 概述` |
| `### X.Y.Z ...` | Subsection | `### 2.2.1 前向过程` |
| `## A. ...` | Appendix heading | `## A. 数学推导` |
| `### A.1 ...` | Appendix section | `### A.1 推导过程` |

Also detects: `后记`, `附录`, `前言`, `本章小结`

## Customization

Edit the script's CSS variables at the top to change colors, fonts, or layout:

```python
CSS_VARS = """
:root {
    --bg: #fafaf9;
    --bg-sidebar: #1e1e2e;
    --accent: #6c5ce7;
    /* ... */
}
"""
```

## Notes

- Code blocks (``` fenced ```) are skipped when parsing headings — `# comments` inside code won't pollute the nav
- The script handles both h1 and h2 level chapters (common when book parts use different heading levels)
- For books with 500+ headings, search becomes especially useful
