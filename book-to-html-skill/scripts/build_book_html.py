#!/usr/bin/env python3
"""
book-to-html: Convert markdown books into HTML with chapter navigation.

Usage:
    python3 build_book_html.py input.md -o output.html
    python3 build_book_html.py part1.md part2.md part3.md -o book.html
    python3 build_book_html.py input.md -o output.html --title "My Book"
"""

import re
import sys
import html as html_module
import argparse
from pathlib import Path


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = r"""
:root {
    --bg: #fafaf9;
    --bg-sidebar: #1e1e2e;
    --bg-sidebar-hover: #2a2a3e;
    --bg-sidebar-active: #3a3a5e;
    --text: #1a1a2e;
    --text-sidebar: #cdd6f4;
    --text-sidebar-muted: #888;
    --accent: #6c5ce7;
    --accent-light: #a29bfe;
    --accent-bg: #f0edff;
    --border: #e8e8e8;
    --code-bg: #f5f5f5;
    --code-text: #d63384;
    --sidebar-width: 320px;
    --heading-font: 'Noto Serif SC', 'Source Han Serif SC', 'Songti SC', serif;
    --body-font: 'Noto Sans SC', 'Source Han Sans SC', 'PingFang SC', -apple-system, sans-serif;
    --mono-font: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 80px; }
body {
    font-family: var(--body-font); font-size: 16px; line-height: 1.8;
    color: var(--text); background: var(--bg);
}
#sidebar {
    position: fixed; top: 0; left: 0; width: var(--sidebar-width);
    height: 100vh; background: var(--bg-sidebar); color: var(--text-sidebar);
    overflow-y: auto; z-index: 100; display: flex; flex-direction: column;
    transition: transform 0.3s ease;
}
#sidebar::-webkit-scrollbar { width: 6px; }
#sidebar::-webkit-scrollbar-track { background: transparent; }
#sidebar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
.sidebar-header { padding: 28px 24px 20px; border-bottom: 1px solid rgba(255,255,255,0.08); flex-shrink: 0; }
.sidebar-header h2 { font-family: var(--heading-font); font-size: 18px; font-weight: 700; color: #fff; letter-spacing: 0.5px; line-height: 1.4; }
.sidebar-header .subtitle { font-size: 12px; color: var(--text-sidebar-muted); margin-top: 6px; }
.nav-search { padding: 12px 16px; flex-shrink: 0; }
.nav-search input {
    width: 100%; padding: 8px 14px; border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; background: rgba(255,255,255,0.06); color: #fff;
    font-size: 13px; outline: none; transition: border-color 0.2s;
}
.nav-search input:focus { border-color: var(--accent-light); }
.nav-search input::placeholder { color: rgba(255,255,255,0.3); }
.nav-links { flex: 1; overflow-y: auto; padding: 8px 0 24px; }
.nav-part {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--accent-light); padding: 20px 24px 6px; margin-top: 4px;
}
.nav-chapter {
    display: block; padding: 8px 24px; color: var(--text-sidebar);
    text-decoration: none; font-size: 14px; font-weight: 500; line-height: 1.5;
    border-left: 3px solid transparent; transition: all 0.15s;
}
.nav-chapter:hover { background: var(--bg-sidebar-hover); color: #fff; border-left-color: var(--accent); }
.nav-chapter.active { background: var(--bg-sidebar-active); color: #fff; border-left-color: var(--accent); }
.nav-section {
    display: block; padding: 5px 24px 5px 36px; color: var(--text-sidebar-muted);
    text-decoration: none; font-size: 13px; line-height: 1.5;
    border-left: 3px solid transparent; transition: all 0.15s;
}
.nav-section:hover { color: var(--text-sidebar); background: var(--bg-sidebar-hover); border-left-color: rgba(108,92,231,0.4); }
.nav-section.active { color: var(--accent-light); background: var(--bg-sidebar-active); border-left-color: var(--accent); }
.nav-section.hidden, .nav-chapter.hidden { display: none; }
#main { margin-left: var(--sidebar-width); min-height: 100vh; }
#topbar {
    position: sticky; top: 0; background: rgba(250,250,249,0.92);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border); padding: 12px 48px; z-index: 50;
    display: flex; align-items: center; justify-content: space-between;
}
#topbar .breadcrumb { font-size: 13px; color: #888; }
#topbar .breadcrumb a { color: var(--accent); text-decoration: none; }
#topbar .breadcrumb a:hover { text-decoration: underline; }
.topbar-actions { display: flex; gap: 12px; align-items: center; }
.topbar-actions button {
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px;
    background: #fff; color: var(--text); font-size: 13px; cursor: pointer; transition: all 0.15s;
}
.topbar-actions button:hover { border-color: var(--accent); color: var(--accent); }
#content { max-width: 860px; margin: 0 auto; padding: 40px 48px 120px; }
h1 {
    font-family: var(--heading-font); font-size: 32px; font-weight: 800;
    color: var(--text); margin: 60px 0 24px; padding-bottom: 12px;
    border-bottom: 3px solid var(--accent); line-height: 1.3;
}
h1:first-child { margin-top: 0; font-size: 38px; text-align: center; border-bottom: none; }
h2 {
    font-family: var(--heading-font); font-size: 24px; font-weight: 700;
    color: var(--text); margin: 48px 0 18px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border); line-height: 1.4;
}
h3 { font-family: var(--heading-font); font-size: 20px; font-weight: 600; color: var(--text); margin: 36px 0 14px; line-height: 1.4; }
h4 { font-family: var(--heading-font); font-size: 17px; font-weight: 600; color: #444; margin: 28px 0 12px; }
p { margin: 0 0 16px; text-align: justify; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
ul, ol { margin: 0 0 20px; padding-left: 28px; }
li { margin-bottom: 6px; }
li > ul, li > ol { margin-top: 6px; margin-bottom: 4px; }
code.inline {
    font-family: var(--mono-font); font-size: 0.88em; background: var(--code-bg);
    color: var(--code-text); padding: 2px 6px; border-radius: 4px; border: 1px solid #eee;
}
pre { margin: 0 0 24px; border-radius: 10px; overflow: hidden; border: 1px solid #e0e0e0; }
pre code {
    font-family: var(--mono-font); font-size: 13.5px; line-height: 1.6;
    display: block; padding: 20px 24px; background: #282c34; color: #abb2bf;
    overflow-x: auto; tab-size: 4;
}
.table-wrapper { margin: 0 0 24px; overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
thead { background: #f8f8f8; }
th { padding: 12px 16px; text-align: left; font-weight: 600; color: #444; border-bottom: 2px solid var(--border); white-space: nowrap; }
td { padding: 10px 16px; border-bottom: 1px solid #f0f0f0; }
tr:last-child td { border-bottom: none; }
tr:hover { background: #fafafa; }
blockquote {
    margin: 0 0 24px; padding: 16px 24px; border-left: 4px solid var(--accent);
    background: var(--accent-bg); border-radius: 0 8px 8px 0; color: #444;
}
blockquote p { margin-bottom: 8px; }
blockquote p:last-child { margin-bottom: 0; }
hr { border: none; height: 1px; background: var(--border); margin: 48px 0; }
.toc { background: #f8f7ff; border: 1px solid #e8e5ff; border-radius: 12px; padding: 28px 32px; margin: 32px 0 48px; }
.toc h2 { font-size: 20px; margin: 0 0 20px; padding-bottom: 12px; border-bottom: 2px solid var(--accent); color: var(--accent); }
.toc-content { max-height: 600px; overflow-y: auto; }
.toc-h1 a { font-weight: 700; font-size: 15px; color: var(--text); display: block; padding: 6px 0; }
.toc-h1 a:hover { color: var(--accent); text-decoration: none; }
.toc-h2 a { font-weight: 500; font-size: 14px; color: #444; display: block; padding: 4px 0; }
.toc-h2 a:hover { color: var(--accent); text-decoration: none; }
.toc-h3 a { font-size: 13px; color: #777; display: block; padding: 3px 0; }
.toc-h3 a:hover { color: var(--accent); text-decoration: none; }
#back-to-top {
    position: fixed; bottom: 32px; right: 32px; width: 44px; height: 44px;
    border-radius: 50%; background: var(--accent); color: #fff; border: none;
    cursor: pointer; font-size: 20px; display: none; align-items: center;
    justify-content: center; box-shadow: 0 4px 12px rgba(108,92,231,0.3);
    transition: all 0.2s; z-index: 99;
}
#back-to-top:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(108,92,231,0.4); }
#back-to-top.visible { display: flex; }
#progress { position: fixed; top: 0; left: var(--sidebar-width); right: 0; height: 3px; z-index: 60; }
#progress-bar { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-light)); width: 0%; transition: width 0.1s; }
#menu-toggle { display: none; position: fixed; top: 12px; left: 12px; z-index: 200; width: 40px; height: 40px; border-radius: 8px; background: var(--bg-sidebar); color: #fff; border: none; cursor: pointer; font-size: 20px; align-items: center; justify-content: center; }
#sidebar-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 90; }
@media (max-width: 1024px) { :root { --sidebar-width: 280px; } #content { padding: 32px 24px 100px; } }
@media (max-width: 768px) {
    #menu-toggle { display: flex; }
    #sidebar { transform: translateX(-100%); width: 300px; }
    #sidebar.open { transform: translateX(0); }
    #sidebar-overlay.open { display: block; }
    #main { margin-left: 0; }
    #topbar { padding: 12px 16px 12px 56px; }
    #progress { left: 0; }
    #content { padding: 24px 16px 80px; }
    h1 { font-size: 26px; margin: 40px 0 20px; }
    h1:first-child { font-size: 28px; }
    h2 { font-size: 20px; margin: 36px 0 14px; }
    h3 { font-size: 17px; margin: 28px 0 10px; }
    pre code { font-size: 12.5px; padding: 16px; }
}
@media print {
    #sidebar, #topbar, #back-to-top, #progress, #menu-toggle { display: none !important; }
    #main { margin-left: 0; }
    #content { max-width: 100%; padding: 0; }
    h1 { page-break-before: always; }
    h1:first-child { page-break-before: avoid; }
    pre { page-break-inside: avoid; }
    table { page-break-inside: avoid; }
}
::selection { background: rgba(108,92,231,0.2); }
"""

# ── JS ───────────────────────────────────────────────────────────────────────

JS = r"""
const sections = [];
const navLinks = document.querySelectorAll('.nav-chapter, .nav-section');
const headings = document.querySelectorAll('h1[id], h2[id], h3[id]');
headings.forEach(h => { sections.push({ id: h.id, el: h, offset: h.offsetTop }); });

function updateActiveNav() {
    const scrollY = window.scrollY + 120;
    let current = null;
    for (const section of sections) { if (section.el.offsetTop <= scrollY) current = section; }
    navLinks.forEach(link => link.classList.remove('active'));
    if (current) {
        const activeLink = document.querySelector(`a[href="#${current.id}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
            const navContainer = document.querySelector('.nav-links');
            const linkTop = activeLink.offsetTop;
            const navHeight = navContainer.clientHeight;
            if (linkTop < navContainer.scrollTop || linkTop > navContainer.scrollTop + navHeight - 40)
                navContainer.scrollTo({ top: linkTop - navHeight / 3, behavior: 'smooth' });
        }
        const breadcrumb = document.getElementById('current-section');
        if (breadcrumb) { const text = current.el.textContent; breadcrumb.textContent = text.length > 40 ? text.substring(0, 40) + '…' : text; }
    }
}
function updateProgress() {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    document.getElementById('progress-bar').style.width = progress + '%';
}
function updateBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (window.scrollY > 400) btn.classList.add('visible'); else btn.classList.remove('visible');
}
document.getElementById('back-to-top').addEventListener('click', () => { window.scrollTo({ top: 0, behavior: 'smooth' }); });
let ticking = false;
window.addEventListener('scroll', () => {
    if (!ticking) { requestAnimationFrame(() => { updateActiveNav(); updateProgress(); updateBackToTop(); ticking = false; }); ticking = true; }
});
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('sidebar-overlay');
menuToggle.addEventListener('click', () => { sidebar.classList.toggle('open'); overlay.classList.toggle('open'); });
overlay.addEventListener('click', () => { sidebar.classList.remove('open'); overlay.classList.remove('open'); });
navLinks.forEach(link => { link.addEventListener('click', () => { if (window.innerWidth <= 768) { sidebar.classList.remove('open'); overlay.classList.remove('open'); } }); });
const searchInput = document.getElementById('nav-search');
searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    navLinks.forEach(link => { if (!query) { link.classList.remove('hidden'); } else { const text = link.textContent.toLowerCase(); link.classList.toggle('hidden', !text.includes(query)); } });
});
function toggleToc() {
    const toc = document.getElementById('toc');
    if (toc) { toc.style.display = toc.style.display === 'none' ? 'block' : 'none'; if (toc.style.display !== 'none') toc.scrollIntoView({ behavior: 'smooth' }); }
}
updateActiveNav(); updateProgress();
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); history.pushState(null, null, this.getAttribute('href')); }
    });
});
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text):
    text = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', text)
    text = re.sub(r'[\s]+', '-', text.strip())
    return text.lower()


def is_chapter(text):
    return bool(re.match(r'^第.+章', text))

def is_part(text):
    return bool(re.match(r'^第.+部分', text))

def is_appendix_section(text):
    return bool(re.match(r'^[A-Z]\.\d+', text))

def is_appendix_heading(text):
    return bool(re.match(r'^[A-Z]\.\s', text))


def inline_fmt(text):
    text = html_module.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`([^`]+)`', r'<code class="inline">\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def process_table(tlines):
    if len(tlines) < 2:
        return ''
    headers = [c.strip() for c in tlines[0].split('|') if c.strip()]
    rows = []
    for line in tlines[2:]:
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if cells:
            rows.append(cells)
    result = ['<div class="table-wrapper"><table>']
    result.append('<thead><tr>')
    for h in headers:
        result.append(f'<th>{inline_fmt(h)}</th>')
    result.append('</tr></thead><tbody>')
    for row in rows:
        result.append('<tr>')
        for cell in row:
            result.append(f'<td>{inline_fmt(cell)}</td>')
        result.append('</tr>')
    result.append('</tbody></table></div>')
    return '\n'.join(result)


# ── Parser ───────────────────────────────────────────────────────────────────

def parse_headings(md_text):
    headings = []
    lines = md_text.split('\n')
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r'^(#{1,4})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            slug = slugify(text) + f'-{i}'
            headings.append((level, text, slug, i))
    return headings


def build_nav(headings, title):
    nav = []
    nav.append('<nav id="sidebar">')
    nav.append('<div class="sidebar-header">')
    nav.append(f'<h2>{html_module.escape(title)}</h2>')
    nav.append('</div>')
    nav.append('<div class="nav-search">')
    nav.append('<input type="text" id="nav-search" placeholder="搜索章节..." autocomplete="off">')
    nav.append('</div>')
    nav.append('<div class="nav-links">')

    for level, text, slug, _ in headings:
        if level == 1 and is_part(text):
            nav.append(f'<div class="nav-part">{text}</div>')
        elif level == 1 and text in ('后记', '附录', 'Afterword', 'Appendix'):
            nav.append(f'<div class="nav-part">{text}</div>')
        elif is_chapter(text):
            nav.append(f'<a href="#{slug}" class="nav-chapter" data-level="2">{text}</a>')
        elif is_appendix_heading(text):
            nav.append(f'<a href="#{slug}" class="nav-chapter" data-level="2">{text}</a>')
        elif re.match(r'^\d+\.\d+\s', text) or is_appendix_section(text):
            nav.append(f'<a href="#{slug}" class="nav-section" data-level="3">{text}</a>')
        elif re.match(r'^\d+\.\d+\.\d+', text):
            nav.append(f'<a href="#{slug}" class="nav-section" data-level="3">{text}</a>')
        elif level == 2 and not text.startswith(('ELF:', '前言', 'Preface')):
            nav.append(f'<a href="#{slug}" class="nav-section" data-level="3">{text}</a>')

    nav.append('</div></nav>')
    return '\n'.join(nav)


def build_toc(headings):
    toc = ['<div class="toc" id="toc">', '<h2>目录</h2>', '<div class="toc-content">']
    for level, text, slug, _ in headings:
        if level <= 3:
            indent = (level - 1) * 20
            cls = f'toc-h{level}'
            toc.append(f'<div class="{cls}" style="margin-left:{indent}px"><a href="#{slug}">{text}</a></div>')
    toc.append('</div></div>')
    return '\n'.join(toc)


def md_to_html(md_text, headings):
    lines = md_text.split('\n')
    html_parts = []
    in_code = False
    code_lang = ''
    code_lines = []
    in_table = False
    table_lines = []
    in_list = False
    list_type = None
    list_items = []
    in_blockquote = False
    heading_map = {h[3]: (h[0], h[1], h[2]) for h in headings}

    def flush_list():
        nonlocal in_list, list_items, list_type
        if in_list and list_items:
            tag = list_type or 'ul'
            html_parts.append(f'<{tag}>')
            for item in list_items:
                html_parts.append(f'<li>{inline_fmt(item)}</li>')
            html_parts.append(f'</{tag}>')
            list_items = []
            in_list = False
            list_type = None

    def flush_table():
        nonlocal in_table, table_lines
        if in_table and table_lines:
            html_parts.append(process_table(table_lines))
            table_lines = []
            in_table = False

    def flush_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            html_parts.append('</blockquote>')
            in_blockquote = False

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith('```'):
            if in_code:
                code_content = html_module.escape('\n'.join(code_lines))
                html_parts.append(f'<pre><code class="language-{code_lang}">{code_content}</code></pre>')
                code_lines = []
                in_code = False
                code_lang = ''
            else:
                flush_list(); flush_table(); flush_blockquote()
                lang_match = re.match(r'^```(\w*)', line.strip())
                code_lang = lang_match.group(1) if lang_match else ''
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if '|' in line and re.match(r'^\s*\|.*\|\s*$', line):
            flush_list(); flush_blockquote()
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line.strip())
            i += 1
            continue
        else:
            flush_table()

        m = re.match(r'^(#{1,4})\s+(.+)$', line)
        if m:
            flush_list(); flush_blockquote()
            level = len(m.group(1))
            text = m.group(2).strip()
            if i in heading_map:
                _, clean_text, slug = heading_map[i]
                html_parts.append(f'<h{level} id="{slug}">{inline_fmt(clean_text)}</h{level}>')
            else:
                html_parts.append(f'<h{level}>{inline_fmt(text)}</h{level}>')
            i += 1
            continue

        if re.match(r'^---+\s*$', line.strip()):
            flush_list(); flush_blockquote()
            html_parts.append('<hr>')
            i += 1
            continue

        if line.strip().startswith('>'):
            if not in_blockquote:
                flush_list()
                in_blockquote = True
                html_parts.append('<blockquote>')
            content = re.sub(r'^>\s*', '', line.strip())
            html_parts.append(f'<p>{inline_fmt(content)}</p>')
            i += 1
            continue
        else:
            flush_blockquote()

        list_match = re.match(r'^(\s*)[-*]\s+(.+)$', line)
        if list_match:
            if not in_list or list_type != 'ul':
                flush_list(); in_list = True; list_type = 'ul'; list_items = []
            list_items.append(list_match.group(2))
            i += 1
            continue

        ol_match = re.match(r'^(\s*)\d+\.\s+(.+)$', line)
        if ol_match:
            if not in_list or list_type != 'ol':
                flush_list(); in_list = True; list_type = 'ol'; list_items = []
            list_items.append(ol_match.group(2))
            i += 1
            continue

        if not line.strip():
            flush_list()
            i += 1
            continue

        flush_list()
        para_lines = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith('#') and not lines[i + 1].strip().startswith('```') and not lines[i + 1].strip().startswith('>') and not re.match(r'^\s*[-*]\s+', lines[i + 1]) and not re.match(r'^\s*\d+\.\s+', lines[i + 1]) and not re.match(r'^---+\s*$', lines[i + 1].strip()) and '|' not in lines[i + 1]:
            i += 1
            para_lines.append(lines[i])
        para_text = ' '.join(para_lines)
        html_parts.append(f'<p>{inline_fmt(para_text)}</p>')
        i += 1

    flush_list(); flush_table(); flush_blockquote()
    if in_code:
        code_content = html_module.escape('\n'.join(code_lines))
        html_parts.append(f'<pre><code class="language-{code_lang}">{code_content}</code></pre>')

    return '\n'.join(html_parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_html(md_text, title=None):
    headings = parse_headings(md_text)

    # Auto-detect title from first h1 if not provided
    if not title:
        for level, text, _, _ in headings:
            if level == 1:
                title = text
                break
    if not title:
        title = 'Book'

    nav_html = build_nav(headings, title)
    toc_html = build_toc(headings)
    content_html = md_to_html(md_text, headings)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)}</title>
<style>{CSS}</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700;800&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body>
<button id="menu-toggle" aria-label="Toggle menu">☰</button>
<div id="sidebar-overlay"></div>
{nav_html}
<div id="main">
<div id="topbar">
    <div class="breadcrumb" id="breadcrumb"><a href="#toc">目录</a> / <span id="current-section">前言</span></div>
    <div class="topbar-actions">
        <button onclick="toggleToc()" title="目录">📑 目录</button>
        <button onclick="window.print()" title="打印">🖨️ 打印</button>
    </div>
</div>
<div id="progress"><div id="progress-bar"></div></div>
<div id="content">
{toc_html}
{content_html}
</div>
</div>
<button id="back-to-top" title="回到顶部">↑</button>
<script>{JS}</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description='Convert markdown book to HTML with chapter navigation')
    parser.add_argument('inputs', nargs='+', help='Input markdown file(s). Multiple files are concatenated in order.')
    parser.add_argument('-o', '--output', required=True, help='Output HTML file path')
    parser.add_argument('--title', default=None, help='Book title (auto-detected from first h1 if omitted)')
    args = parser.parse_args()

    # Read and merge inputs
    parts = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f'Error: file not found: {path}', file=sys.stderr)
            sys.exit(1)
        parts.append(p.read_text(encoding='utf-8'))
    md_text = '\n\n'.join(parts)

    # Generate
    html_text = generate_html(md_text, title=args.title)

    # Write
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding='utf-8')

    headings = parse_headings(md_text)
    print(f'Done! {len(headings)} headings → {out} ({len(html_text.encode("utf-8")) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
