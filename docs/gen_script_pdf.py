#!/usr/bin/env python3
"""Generate presenter-friendly PDF from lecture script markdown."""

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    KeepTogether, Table, TableStyle
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ─── Fonts ───────────────────────────────────────────────────────────────────
# FY = jf-openhuninn 圓體（粉圓風格，一般內文用）
# HM = STHeiti Medium（完整繁體字覆蓋，粗體強調用）
import os as _os
pdfmetrics.registerFont(TTFont(
    'FY', _os.path.expanduser('~/Library/Fonts/fanyuan.ttf')))
pdfmetrics.registerFont(TTFont(
    'HM', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('H', normal='FY', bold='HM')

# ─── Palette ─────────────────────────────────────────────────────────────────
NAVY   = HexColor('#1B3A5C')
SKY    = HexColor('#2D6A9F')
TEXT   = HexColor('#18192B')
MUTED  = HexColor('#5E6578')
CODE   = HexColor('#8B1A2B')
DIVIDER= HexColor('#C4D4E8')
SLIDE_LABEL_COL = HexColor('#8BB8D8')

# ─── Layout ──────────────────────────────────────────────────────────────────
W, H   = A4
LM     = 2.2 * cm
RM     = 2.2 * cm
TM     = 1.5 * cm
BM     = 2.2 * cm
CW     = W - LM - RM           # usable content width

# ─── Style definitions ───────────────────────────────────────────────────────
# 字體方案：FY（jf-openhuninn 圓體）一般文字；HM（香萃零度黑）粗體強調
BODY = ParagraphStyle(
    'body', fontName='FY', fontSize=16, leading=32,
    textColor=TEXT, spaceAfter=10, spaceBefore=0, alignment=TA_LEFT)

BULLET = ParagraphStyle(
    'bullet', fontName='FY', fontSize=15.5, leading=30,
    textColor=TEXT, leftIndent=22, spaceAfter=8,
    bulletText='·', bulletFontName='HM', bulletFontSize=20,
    bulletColor=SKY, bulletIndent=0)

NITEM = ParagraphStyle(
    'nitem', fontName='FY', fontSize=15.5, leading=30,
    textColor=TEXT, leftIndent=28, spaceAfter=8)

SUBLABEL = ParagraphStyle(
    'sublabel', fontName='HM', fontSize=16, leading=26,
    textColor=SKY, spaceBefore=16, spaceAfter=6)

HDR_SLIDE_LABEL = ParagraphStyle(
    'hdrslide', fontName='FY', fontSize=10.5, leading=16,
    textColor=SLIDE_LABEL_COL)

HDR_TITLE = ParagraphStyle(
    'hdrtitle', fontName='HM', fontSize=19, leading=27,
    textColor=white)

NOTE = ParagraphStyle(
    'note', fontName='FY', fontSize=12, leading=20,
    textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=4)

COVER_MAIN = ParagraphStyle(
    'cover_main', fontName='HM', fontSize=23, leading=38,
    textColor=NAVY, alignment=TA_CENTER)

COVER_SUB = ParagraphStyle(
    'cover_sub', fontName='FY', fontSize=13, leading=22,
    textColor=MUTED, alignment=TA_CENTER, spaceBefore=6)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def esc(t: str) -> str:
    """Escape XML-special chars so ReportLab won't choke."""
    return (t.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;'))


def md2rl(line: str) -> str:
    """Convert inline **bold** and `code` to ReportLab XML markup."""
    parts = []
    last = 0
    for m in re.finditer(r'\*\*(.+?)\*\*|`([^`]+)`', line, re.DOTALL):
        parts.append(esc(line[last:m.start()]))
        if m.group(1) is not None:           # **bold** → HM + 2pt larger = weight+size double contrast
            parts.append(f'<font name="HM" size="18">{esc(m.group(1))}</font>')
        else:                                 # `code`
            parts.append(f'<font name="HM" color="{CODE.hexval()}">'
                         f'{esc(m.group(2))}</font>')
        last = m.end()
    parts.append(esc(line[last:]))
    return ''.join(parts)


def slide_header(num: int, title: str) -> Table:
    """Return a navy full-width banner for a slide section."""
    lp = Paragraph(f'SLIDE {num}', HDR_SLIDE_LABEL)
    tp = Paragraph(esc(title),     HDR_TITLE)
    t = Table([[lp, tp]], colWidths=[CW * 0.16, CW * 0.84])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), NAVY),
        ('TOPPADDING',   (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 11),
        ('LEFTPADDING',  (0, 0), (0,  -1), 14),
        ('LEFTPADDING',  (1, 0), (1,  -1), 6),
        ('RIGHTPADDING', (1, 0), (1,  -1), 14),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def process_lines(raw_lines: list[str]) -> list:
    """Convert a list of markdown lines into ReportLab flowables."""
    out = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue

        # ── horizontal rule
        if line == '---':
            out.append(Spacer(1, 0.2 * cm))
            out.append(HRFlowable(
                width='100%', thickness=0.4, color=DIVIDER,
                spaceBefore=0, spaceAfter=0.2 * cm))
            continue

        # ── standalone bold line → sub-section label
        m = re.match(r'^\*\*([^*]+)\*\*$', line)
        if m:
            out.append(Paragraph(esc(m.group(1)), SUBLABEL))
            continue

        # ── bullet  - item
        if line.startswith('- '):
            out.append(Paragraph(md2rl(line[2:].strip()), BULLET))
            continue

        # ── numbered item  1. text
        m = re.match(r'^(\d+)\.\s+(.+)$', line)
        if m:
            prefix = (f'<font name="HM" color="{SKY.hexval()}">'
                      f'{m.group(1)}.</font> ')
            out.append(Paragraph(prefix + md2rl(m.group(2)), NITEM))
            continue

        # ── italic note line  *...*  (closing italic, not bold)
        m = re.match(r'^\*([^*].+?[^*])\*$', line)
        if m:
            out.append(Paragraph(esc(m.group(1)), NOTE))
            continue

        # ── regular paragraph
        rl = md2rl(line)
        if rl.strip():
            out.append(Paragraph(rl, BODY))

    return out


# ─── Footer ──────────────────────────────────────────────────────────────────
def draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('FY', 9.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(LM, 1.2 * cm,
                      '以圖神經網路預測短影音平台表現  ·  GNN 期末專案  ·  財金所 羅頤')
    canvas.drawRightString(W - RM, 1.2 * cm, f'第 {doc.page} 頁')
    # thin rule above footer
    canvas.setStrokeColor(DIVIDER)
    canvas.setLineWidth(0.5)
    canvas.line(LM, 1.65 * cm, W - RM, 1.65 * cm)
    canvas.restoreState()


# ─── Parser ──────────────────────────────────────────────────────────────────
def parse_md(filepath: str) -> list[dict]:
    text = Path(filepath).read_text('utf-8')
    sections, current = [], None
    for line in text.split('\n'):
        m = re.match(r'^## Slide (\d+)｜(.+)$', line)
        if m:
            if current:
                sections.append(current)
            current = {'num': int(m.group(1)), 'title': m.group(2), 'lines': []}
        elif current is not None:
            current['lines'].append(line)
    if current:
        sections.append(current)
    return sections


# ─── Builder ─────────────────────────────────────────────────────────────────
def build_pdf(md_path: str, out_path: str):
    sections = parse_md(md_path)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM,  bottomMargin=BM,
        title='講稿：以圖神經網路預測短影音平台表現',
        author='財金所 羅頤',
    )

    story = []

    # ── Cover block ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph('講 稿', ParagraphStyle(
        'ctag', fontName='HM', fontSize=12, textColor=SKY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(esc('以圖神經網路預測短影音平台表現'), COVER_MAIN))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        esc('GNN 期末專案  ·  財金所 羅頤  ·  ShortVideo (WWW 2025)  ·  HeteroGNN + GraphSAGE'),
        COVER_SUB))
    story.append(Spacer(1, 1.2 * cm))
    story.append(HRFlowable(width='100%', thickness=2, color=NAVY,
                             spaceBefore=0, spaceAfter=0))
    story.append(Spacer(1, 0.9 * cm))

    # ── Slide sections ───────────────────────────────────────────────────────
    for sec in sections:
        header   = slide_header(sec['num'], sec['title'])
        spacer_s = Spacer(1, 0.35 * cm)
        flowables = process_lines(sec['lines'])

        # Keep the header + first two body paragraphs on the same page
        anchor = [header, spacer_s] + flowables[:2]
        story.append(KeepTogether(anchor))
        story += flowables[2:]
        story.append(Spacer(1, 0.85 * cm))

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    print(f'✓  PDF saved → {out_path}')


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    build_pdf(
        '/Users/luoyi/Desktop/shortvideo_mmgcn_poc/docs/presentation_script.md',
        '/Users/luoyi/Desktop/shortvideo_mmgcn_poc/docs/presentation_script.pdf',
    )
