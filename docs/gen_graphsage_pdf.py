#!/usr/bin/env python3
"""Generate GraphSAGE step-by-step walkthrough PDF with concrete numbers."""

import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── Font ─────────────────────────────────────────────────────────────────────
try:
    pdfmetrics.registerFont(TTFont('CJK',  '/System/Library/Fonts/STHeiti Light.ttc',  subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKB', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
    F, FB = 'CJK', 'CJKB'
except Exception as e:
    print(f'CJK font unavailable ({e}), falling back to Helvetica')
    F, FB = 'Helvetica', 'Helvetica-Bold'

# ─── Palette ──────────────────────────────────────────────────────────────────
NAVY    = colors.HexColor('#0d1b4b')
BLUE    = colors.HexColor('#1a56c4')
LBLUE   = colors.HexColor('#dbeafe')
DBLUE   = colors.HexColor('#1e3a8a')
GREEN   = colors.HexColor('#065f46')
LGREEN  = colors.HexColor('#d1fae5')
AMBER   = colors.HexColor('#78350f')
LAMBER  = colors.HexColor('#fef3c7')
PURPLE  = colors.HexColor('#4c1d95')
LPURPLE = colors.HexColor('#ede9fe')
RED     = colors.HexColor('#7f1d1d')
LRED    = colors.HexColor('#fee2e2')
GRAY    = colors.HexColor('#374151')
LGRAY   = colors.HexColor('#f3f4f6')
WHITE   = colors.white
BORDER  = colors.HexColor('#93c5fd')
DGRAY   = colors.HexColor('#9ca3af')

PAGE_W, PAGE_H = A4
M = 1.8 * cm
CW = PAGE_W - 2 * M  # content width


# ─── Style factory ────────────────────────────────────────────────────────────
def S(name, **kw):
    defaults = dict(fontName=F, fontSize=10, textColor=GRAY, leading=15, spaceAfter=4)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


STYLES = {
    'h1':    S('h1',  fontName=FB, fontSize=20, textColor=NAVY,  spaceAfter=6,  spaceBefore=0, leading=26, alignment=TA_CENTER),
    'h2':    S('h2',  fontName=FB, fontSize=14, textColor=DBLUE, spaceAfter=8,  spaceBefore=14, leading=20),
    'h3':    S('h3',  fontName=FB, fontSize=11, textColor=BLUE,  spaceAfter=6,  spaceBefore=8,  leading=16),
    'body':  S('body',fontSize=10, leading=16, alignment=TA_JUSTIFY, spaceAfter=6),
    'mono':  S('mono',fontName='Courier', fontSize=9.5, textColor=DBLUE, leading=14, spaceAfter=3),
    'monob': S('monob',fontName='Courier-Bold', fontSize=10, textColor=NAVY, leading=15, spaceAfter=3),
    'small': S('small',fontSize=8.5, textColor=DGRAY, leading=13),
    'result':S('result',fontName=FB, fontSize=10.5, textColor=GREEN, leading=15),
    'warn':  S('warn', fontName=FB, fontSize=10,   textColor=RED,   leading=15),
    'sub':   S('sub',  fontSize=9,   textColor=PURPLE, leading=14),
    'ctr':   S('ctr',  alignment=TA_CENTER, leading=14),
    'th':    S('th',   fontName=FB, fontSize=9.5, textColor=WHITE, alignment=TA_CENTER, leading=14),
    'td':    S('td',   fontName='Courier', fontSize=9.5, textColor=GRAY, alignment=TA_CENTER, leading=14),
    'tdl':   S('tdl',  fontName='Courier', fontSize=9.5, textColor=GRAY, alignment=TA_LEFT,   leading=14),
    'tdh':   S('tdh',  fontName=FB, fontSize=9.5, textColor=NAVY, alignment=TA_CENTER, leading=14),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def p(text, style='body'):   return Paragraph(text, STYLES[style])
def sp(h=0.2):               return Spacer(1, h * cm)
def hr():                    return HRFlowable(width=CW, thickness=0.5, color=BORDER, spaceAfter=6)

def fv(v):
    """Format numpy vector as [ a,  b,  c ]."""
    return '[' + ',  '.join(f'{x:.4f}' for x in v) + ']'

def fv3(v):
    """Format vector rounded to 4dp with consistent spacing."""
    return '[' + ',  '.join(f'{x:7.4f}' for x in v) + ']'


def box(rows, bg, border=None, cw=None, pad=6, lpad=10):
    """Styled table acting as a coloured block."""
    bc = border or bg
    if cw is None:
        cw = [CW]
    t = Table([[r] if not isinstance(r, list) else r for r in rows], colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('BOX', (0, 0), (-1, -1), 1, bc),
        ('LEFTPADDING', (0, 0), (-1, -1), lpad),
        ('RIGHTPADDING', (0, 0), (-1, -1), lpad),
        ('TOPPADDING', (0, 0), (-1, -1), pad),
        ('BOTTOMPADDING', (0, 0), (-1, -1), pad),
    ]))
    return t


def vec_row(label, vec, bg=LBLUE):
    """Single-row table: label  =  [a, b, c]."""
    lw = CW * 0.28
    rw = CW - lw
    rows = [[p(label, 'monob'), p('=  ' + fv(vec), 'mono')]]
    t = Table(rows, colWidths=[lw, rw])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('BOX', (0, 0), (-1, -1), 0.8, BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def calc_block(lines, bg=LGREEN, border=None):
    """Multi-line calculation block with monospace font."""
    rows = [[p(ln, 'mono')] for ln in lines]
    return box(rows, bg, border or colors.HexColor('#6ee7b7'), pad=5, lpad=14)


def section_header(num, title):
    """Bold section divider with number badge."""
    badge = [[p(f'  {num}  ', 'th')]]
    badge_t = Table(badge, colWidths=[0.7 * cm])
    badge_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BLUE),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    title_t = Table([[p(f'  {title}', 'h2')]], colWidths=[CW - 0.7 * cm])
    title_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), DBLUE),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    combo = Table([[badge_t, title_t]], colWidths=[0.7 * cm, CW - 0.7 * cm])
    combo.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return combo


def step_header(letter, title, bg=BLUE):
    row = [[p(f' {letter} ', 'th'), p(f'  {title}', 'th')]]
    lw = CW * 0.1
    t = Table(row, colWidths=[lw, CW - lw])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def node_table(headers, rows_data, col_widths):
    header_row = [p(h, 'th') for h in headers]
    data = [header_row]
    for rd in rows_data:
        data.append([p(str(c), 'td') for c in rd])
    t = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LGRAY, WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]
    t.setStyle(TableStyle(style))
    return t


# ─── Compute Values ───────────────────────────────────────────────────────────
def compute():
    h_v = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 0.0]])
    h_a = np.array([[0.8, 0.2, 0.3], [0.1, 0.9, 0.4]])
    h_c = np.array([[0.3, 0.3, 0.6]])
    W_n = 0.5 * np.eye(3)
    W_s = 1.0 * np.eye(3)

    # Layer 1 — v0
    out_A_v0 = W_n @ h_a[0] + W_s @ h_v[0]
    out_B_v0 = W_n @ h_c[0] + W_s @ h_v[0]
    out_C_v0 = W_n @ h_v[1] + W_s @ h_v[0]
    h_v0_L1 = np.maximum(out_A_v0 + out_B_v0 + out_C_v0, 0)

    # Layer 1 — v1
    mean_C_v1 = (h_v[0] + h_v[2]) / 2
    out_A_v1 = W_n @ h_a[0] + W_s @ h_v[1]
    out_B_v1 = W_n @ h_c[0] + W_s @ h_v[1]
    out_C_v1 = W_n @ mean_C_v1 + W_s @ h_v[1]
    h_v1_L1 = np.maximum(out_A_v1 + out_B_v1 + out_C_v1, 0)

    # Layer 1 — v2
    out_A_v2 = W_n @ h_a[1] + W_s @ h_v[2]
    out_B_v2 = W_n @ h_c[0] + W_s @ h_v[2]
    out_C_v2 = W_n @ h_v[1] + W_s @ h_v[2]
    h_v2_L1 = np.maximum(out_A_v2 + out_B_v2 + out_C_v2, 0)

    # Layer 1 — authors and category
    h_a0_L1 = np.maximum(W_n @ ((h_v[0] + h_v[1]) / 2) + W_s @ h_a[0], 0)
    h_a1_L1 = np.maximum(W_n @ h_v[2] + W_s @ h_a[1], 0)
    h_c0_L1 = np.maximum(W_n @ ((h_v[0] + h_v[1] + h_v[2]) / 3) + W_s @ h_c[0], 0)

    # Layer 2 — v0
    out_A_v0_L2 = W_n @ h_a0_L1 + W_s @ h_v0_L1
    out_B_v0_L2 = W_n @ h_c0_L1 + W_s @ h_v0_L1
    out_C_v0_L2 = W_n @ h_v1_L1 + W_s @ h_v0_L1
    h_v0_L2 = np.maximum(out_A_v0_L2 + out_B_v0_L2 + out_C_v0_L2, 0)

    # MLP  (d=3 → 2 outputs for this toy example)
    W1 = np.array([[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]])
    W2 = np.array([[0.5, -0.3], [0.2, 0.6]])
    z1 = W1 @ h_v0_L2
    z1r = np.maximum(z1, 0)
    out_mlp = W2 @ z1r
    pred_rate = 1 / (1 + np.exp(-out_mlp[0]))
    pred_log_wt = out_mlp[1]
    pred_wt = np.expm1(pred_log_wt)

    # Loss
    true_rate, true_log_wt = 0.65, 3.50
    rate_loss = (pred_rate - true_rate) ** 2
    diff_wt = abs(pred_log_wt - true_log_wt)
    wt_loss = 0.5 * diff_wt ** 2 if diff_wt < 1.0 else diff_wt - 0.5
    total_loss = rate_loss + wt_loss

    return dict(
        h_v=h_v, h_a=h_a, h_c=h_c, W_n=W_n, W_s=W_s,
        out_A_v0=out_A_v0, out_B_v0=out_B_v0, out_C_v0=out_C_v0,
        h_v0_L1=h_v0_L1, h_v1_L1=h_v1_L1, h_v2_L1=h_v2_L1,
        h_a0_L1=h_a0_L1, h_a1_L1=h_a1_L1, h_c0_L1=h_c0_L1,
        out_A_v0_L2=out_A_v0_L2, out_B_v0_L2=out_B_v0_L2, out_C_v0_L2=out_C_v0_L2,
        h_v0_L2=h_v0_L2,
        W1=W1, W2=W2, z1=z1, z1r=z1r, out_mlp=out_mlp,
        pred_rate=pred_rate, pred_log_wt=pred_log_wt, pred_wt=pred_wt,
        true_rate=true_rate, true_log_wt=true_log_wt,
        rate_loss=rate_loss, wt_loss=wt_loss, total_loss=total_loss,
        diff_wt=diff_wt,
    )


# ─── PDF Story Builder ────────────────────────────────────────────────────────
def build_story(V):
    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story += [sp(2)]
    cover_inner = [
        [p('GraphSAGE  完整手算範例', 'h1')],
        [p('Pipeline C — 異質圖神經網路逐步數值演示', 'ctr')],
        [sp(0.3)],
        [hr()],
        [sp(0.2)],
        [p('本文件以一個精簡玩具圖（3個video、2個author、1個category）', 'body')],
        [p('完整走過 HeteroGNN (GraphSAGE 變體) 的每一步數值計算，', 'body')],
        [p('所有公式、矩陣、向量均對應 hetero_gnn.py 與 train_hetero.py 的實際程式碼。', 'body')],
    ]
    cover_t = Table(cover_inner, colWidths=[CW])
    cover_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LBLUE),
        ('BOX', (0, 0), (-1, -1), 2, BLUE),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
    ]))
    story.append(cover_t)
    story += [sp(0.5)]

    # Quick legend
    legend_data = [
        [p('符號', 'th'), p('代表意思', 'th'), p('實際設定 (玩具)', 'th'), p('程式碼對應', 'th')],
        [p('d',   'td'), p('embedding 維度', 'tdl'), p('3（真實:128）', 'td'), p('args.d = 128', 'tdl')],
        [p('L',   'td'), p('GNN 層數', 'tdl'),       p('2',            'td'), p('args.layers = 2', 'tdl')],
        [p('k',   'td'), p('kNN 近鄰數',    'tdl'),  p('1（真實:10）', 'td'), p('args.k = 10', 'tdl')],
        [p('N_v', 'td'), p('video 節點數',  'tdl'),  p('3（真實:54,088）','td'), p('data[video].num_nodes', 'tdl')],
        [p('W_n', 'td'), p('鄰居訊息的線性矩陣', 'tdl'), p('0.5 × I₃', 'td'), p('SAGEConv.lin_l', 'tdl')],
        [p('W_s', 'td'), p('自身訊息的線性矩陣', 'tdl'), p('1.0 × I₃', 'td'), p('SAGEConv.lin_r', 'tdl')],
    ]
    lw = [CW*0.09, CW*0.26, CW*0.30, CW*0.33]
    leg_t = Table(legend_data, colWidths=lw)
    leg_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LGRAY, WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story += [sp(0.3), p('符號對照表', 'h3'), leg_t]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Dataset & Graph
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('1', '資料集：玩具圖的節點與邊'), sp(0.3)]

    story += [p('玩具圖包含三種節點，六種邊（原始3種 + ToUndirected 自動加的反向邊）。', 'body')]

    # Node table
    story += [sp(0.2), p('節點特徵（Input Projection 之前的原始值）', 'h3')]
    node_data = [
        ['節點', '索引', '特徵維度', '數值（簡化後）', '角色'],
        ['video', 'v0', '819d → 3d', '[1.0,  0.0,  0.5]', 'author=a0, cat=c0, kNN鄰居=v1'],
        ['video', 'v1', '819d → 3d', '[0.0,  1.0,  0.5]', 'author=a0, cat=c0, kNN鄰居=v0,v2'],
        ['video', 'v2', '819d → 3d', '[0.5,  0.5,  0.0]', 'author=a1, cat=c0, kNN鄰居=v1'],
        ['author', 'a0', '10d → 3d', '[0.8,  0.2,  0.3]', '發布了 v0, v1'],
        ['author', 'a1', '10d → 3d', '[0.1,  0.9,  0.4]', '發布了 v2'],
        ['category', 'c0', 'Embedding', '[0.3,  0.3,  0.6]', '所有影片都屬於此分類'],
    ]
    nd_cw = [CW*0.11, CW*0.08, CW*0.15, CW*0.30, CW*0.34]
    nt_rows = [[p(c, 'th') if i == 0 else p(c, 'tdl' if j == 4 else 'td') for j, c in enumerate(row)]
               for i, row in enumerate(node_data)]
    nt = Table(nt_rows, colWidths=nd_cw)
    nt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LGRAY, LGRAY, LGRAY, LAMBER, LAMBER, LPURPLE]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story += [nt, sp(0.4)]

    # Edge table
    story += [p('邊的種類（ToUndirected 之後）', 'h3')]
    edge_data = [
        ['邊種類 (src, rel, dst)', '數量', '建立方式'],
        ['(video, posted_by,   author)',    '3',   '每支影片 → 其 author_id'],
        ['(author, rev_posted_by, video)',   '3',   'ToUndirected 自動加'],
        ['(video, belongs_to, category)',   '3',   '每支影片 → 其 category_id'],
        ['(category, rev_belongs_to, video)','3',  'ToUndirected 自動加'],
        ['(video, similar_to, video)',       '2+2', 'cosine kNN (k=1): v0↔v1, v1↔v2'],
    ]
    et_rows = [[p(c, 'th') if i == 0 else p(c, 'tdl' if j == 0 else 'td') for j, c in enumerate(row)]
               for i, row in enumerate(edge_data)]
    et = Table(et_rows, colWidths=[CW*0.52, CW*0.10, CW*0.36])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LGRAY, WHITE]*5),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story += [et]

    note_rows = [[p('注意：HeteroConv 為每種 edge type 建立獨立的 SAGEConv（各自有獨立的 W_n 和 W_s），\n'
                    '到達同一種目標節點的多個 SAGEConv 輸出以 aggr="sum" 加總。\n'
                    '參見 hetero_gnn.py:71-75  →  conv_dict = {et: SAGEConv((-1,-1), d) for et in edge_types}', 'sub')]]
    story += [sp(0.3), box(note_rows, LPURPLE, PURPLE, pad=8)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Input Projection
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('2', 'Input Projection：特徵統一投影到 d=3 維'), sp(0.3)]

    story += [p('程式碼：hetero_gnn.py:112-121', 'small'), sp(0.2)]
    code1 = calc_block([
        'for nt in self.node_types:',
        '    if nt == "category":',
        '        h[nt] = self.cat_embed.weight        # Embedding 直接取權重',
        '    else:',
        '        h[nt] = self.proj[nt](x_dict[nt])   # Linear(-1 → d)',
    ], LGRAY, DGRAY)
    story += [code1, sp(0.3)]

    story += [p('各節點投影後的初始 Embedding h（玩具範例數值）', 'h3'), sp(0.1)]
    story += [vec_row('h_v0', V['h_v'][0], LBLUE), sp(0.08),
              vec_row('h_v1', V['h_v'][1], LBLUE), sp(0.08),
              vec_row('h_v2', V['h_v'][2], LBLUE), sp(0.15),
              vec_row('h_a0', V['h_a'][0], LAMBER), sp(0.08),
              vec_row('h_a1', V['h_a'][1], LAMBER), sp(0.15),
              vec_row('h_c0', V['h_c'][0], LPURPLE)]

    story += [sp(0.4), p('簡化後的 SAGEConv 權重矩陣（所有 edge type 共用同樣結構）', 'h3')]
    story += [p('真實模型中，每種 edge type 有獨立的 W_n 和 W_s（由 SAGEConv 初始化），'
                '此處為了手算清晰，設為 0.5×I₃ 和 1.0×I₃。', 'small'), sp(0.1)]

    mat_rows = [
        [p('W_n  =  0.5 × I₃  =', 'monob'),
         p('[ [0.5, 0.0, 0.0],\n   [0.0, 0.5, 0.0],\n   [0.0, 0.0, 0.5] ]', 'mono'),
         p('W_s  =  1.0 × I₃  =', 'monob'),
         p('[ [1.0, 0.0, 0.0],\n   [0.0, 1.0, 0.0],\n   [0.0, 0.0, 1.0] ]', 'mono')],
    ]
    mt = Table(mat_rows, colWidths=[CW*0.22, CW*0.26, CW*0.22, CW*0.26])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LGREEN),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#6ee7b7')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story += [mt]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — SAGEConv Layer 1: v0 full breakdown
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('3', 'SAGEConv Layer 1：以 video 0 (v0) 為例完整計算'), sp(0.3)]

    story += [p('SAGEConv 公式（對每個目標節點 v）：', 'h3')]
    formula_rows = [[p('h_v_new  =  W_n × MEAN( h_u  for u ∈ neighbors(v) )  +  W_s × h_v', 'monob')]]
    story += [box(formula_rows, LBLUE, BLUE, pad=10), sp(0.3)]

    story += [p('v0 的鄰居（透過各 edge type 連接）：', 'h3')]
    nb_data = [
        ['Edge Type (→ v0)', '鄰居節點', '說明'],
        ['(author, rev_posted_by, video)',   'a0',   'v0 是由 a0 發布的'],
        ['(category, rev_belongs_to, video)','c0',   'v0 屬於 c0 分類'],
        ['(video, similar_to, video)',       'v1',   'kNN k=1，v0 的最近影片是 v1'],
    ]
    nb_rows = [[p(c, 'th') if i == 0 else p(c, 'tdl' if j == 0 else 'td') for j, c in enumerate(row)]
               for i, row in enumerate(nb_data)]
    nb_t = Table(nb_rows, colWidths=[CW*0.44, CW*0.12, CW*0.42])
    nb_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LAMBER, LGREEN, LBLUE]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story += [nb_t, sp(0.4)]

    # --- Edge type A
    story += [step_header('A', 'Edge Type: (author, rev_posted_by, video)  →  a0 傳訊給 v0', AMBER)]
    lines_A = [
        f'鄰居只有 1 個：MEAN(neighbors) = h_a0 = {fv(V["h_a"][0])}',
        '',
        f'out_A = W_n × h_a0  +  W_s × h_v0',
        f'      = 0.5 × {fv(V["h_a"][0])}  +  1.0 × {fv(V["h_v"][0])}',
        f'      = {fv(0.5 * V["h_a"][0])}  +  {fv(V["h_v"][0])}',
        f'      = {fv(V["out_A_v0"])}',
    ]
    story += [sp(0.1), calc_block(lines_A, LAMBER, AMBER), sp(0.25)]

    # --- Edge type B
    story += [step_header('B', 'Edge Type: (category, rev_belongs_to, video)  →  c0 傳訊給 v0', GREEN)]
    lines_B = [
        f'鄰居只有 1 個：MEAN(neighbors) = h_c0 = {fv(V["h_c"][0])}',
        '',
        f'out_B = W_n × h_c0  +  W_s × h_v0',
        f'      = 0.5 × {fv(V["h_c"][0])}  +  1.0 × {fv(V["h_v"][0])}',
        f'      = {fv(0.5 * V["h_c"][0])}  +  {fv(V["h_v"][0])}',
        f'      = {fv(V["out_B_v0"])}',
    ]
    story += [calc_block(lines_B, LGREEN, GREEN), sp(0.25)]

    # --- Edge type C
    story += [step_header('C', 'Edge Type: (video, similar_to, video)  →  v1 傳訊給 v0', BLUE)]
    lines_C = [
        f'鄰居只有 1 個：MEAN(neighbors) = h_v1 = {fv(V["h_v"][1])}',
        '',
        f'out_C = W_n × h_v1  +  W_s × h_v0',
        f'      = 0.5 × {fv(V["h_v"][1])}  +  1.0 × {fv(V["h_v"][0])}',
        f'      = {fv(0.5 * V["h_v"][1])}  +  {fv(V["h_v"][0])}',
        f'      = {fv(V["out_C_v0"])}',
    ]
    story += [calc_block(lines_C, LBLUE, BLUE), sp(0.3)]

    # --- Aggregation
    story += [step_header('D', 'HeteroConv  aggr="sum"：三個輸出加總', PURPLE)]
    sum_raw = V['out_A_v0'] + V['out_B_v0'] + V['out_C_v0']
    lines_D = [
        f'h_v0_raw = out_A  +  out_B  +  out_C',
        f'         = {fv(V["out_A_v0"])}',
        f'         + {fv(V["out_B_v0"])}',
        f'         + {fv(V["out_C_v0"])}',
        f'         = {fv(sum_raw)}',
        '',
        f'ReLU( h_v0_raw ) = max(0, {fv(sum_raw)})',
        f'                 = {fv(V["h_v0_L1"])}  ← 全部 > 0，無變化',
    ]
    story += [calc_block(lines_D, LPURPLE, PURPLE)]

    res_rows = [[p(f'h_v0  after  Layer 1  =  {fv(V["h_v0_L1"])}', 'result')]]
    story += [sp(0.2), box(res_rows, LGREEN, GREEN, pad=10)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Layer 1 Summary (all nodes)
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('4', 'Layer 1 結果彙整：所有節點'), sp(0.3)]
    story += [p('（v0 詳算已見第3節；其他節點計算方式相同，此處只列結果）', 'small'), sp(0.2)]

    all_nodes = [
        ['節點', 'Layer 0 (初始)', 'Layer 1 (after ReLU)', '更新說明'],
        ['v0', fv(V['h_v'][0]),  fv(V['h_v0_L1']), '整合了 a0, c0, v1 的訊息'],
        ['v1', fv(V['h_v'][1]),  fv(V['h_v1_L1']), '整合了 a0, c0, MEAN(v0,v2)'],
        ['v2', fv(V['h_v'][2]),  fv(V['h_v2_L1']), '整合了 a1, c0, v1 的訊息'],
        ['a0', fv(V['h_a'][0]),  fv(V['h_a0_L1']), '整合了 MEAN(v0,v1) 的訊息'],
        ['a1', fv(V['h_a'][1]),  fv(V['h_a1_L1']), '整合了 v2 的訊息'],
        ['c0', fv(V['h_c'][0]),  fv(V['h_c0_L1']), '整合了 MEAN(v0,v1,v2) 的訊息'],
    ]
    an_rows = [[p(c, 'th') if i == 0 else p(c, 'tdl' if j == 3 else 'td') for j, c in enumerate(row)]
               for i, row in enumerate(all_nodes)]
    an_cw = [CW*0.06, CW*0.29, CW*0.29, CW*0.34]
    an_t = Table(an_rows, colWidths=an_cw)
    an_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (0, 3), [LGRAY]),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LBLUE, LBLUE, LBLUE, LAMBER, LAMBER, LPURPLE]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ]))
    story += [an_t, sp(0.4)]

    key_obs = [[p(
        '觀察：v1 的第2維 (3.375) 明顯比 Layer 0 (1.0) 大很多，因為 v1 同時收到了\n'
        'v0 和 v2 的訊息（v1 是橋接節點，有最多視訊鄰居）。\n'
        'Layer 1 之後每個節點已包含「一跳鄰居」的資訊。', 'sub')]]
    story += [box(key_obs, LPURPLE, PURPLE, pad=8)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — SAGEConv Layer 2: v0
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('5', 'SAGEConv Layer 2：v0 的計算（輸入換成 Layer 1 的 h）'), sp(0.3)]

    story += [p('Layer 2 的公式與 Layer 1 完全相同，只是所有輸入 h 換成 Layer 1 輸出後的值。', 'body')]
    story += [p('Layer 2 讓 v0 能看到「鄰居的鄰居」（二跳）的資訊。', 'body'), sp(0.2)]

    story += [p('Layer 1 輸出（Layer 2 的輸入）', 'h3'), sp(0.1)]
    story += [vec_row('h_v0  (L1)', V['h_v0_L1'], LBLUE), sp(0.08),
              vec_row('h_a0  (L1)', V['h_a0_L1'], LAMBER), sp(0.08),
              vec_row('h_c0  (L1)', V['h_c0_L1'], LPURPLE), sp(0.08),
              vec_row('h_v1  (L1)', V['h_v1_L1'], LBLUE)]
    story += [sp(0.3)]

    story += [step_header('A', 'Edge Type A (a0_L1 → v0_L1)', AMBER)]
    lA2 = [
        f'out_A = W_n × h_a0_L1  +  W_s × h_v0_L1',
        f'      = 0.5 × {fv(V["h_a0_L1"])}  +  1.0 × {fv(V["h_v0_L1"])}',
        f'      = {fv(0.5 * V["h_a0_L1"])}  +  {fv(V["h_v0_L1"])}',
        f'      = {fv(V["out_A_v0_L2"])}',
    ]
    story += [sp(0.1), calc_block(lA2, LAMBER, AMBER), sp(0.25)]

    story += [step_header('B', 'Edge Type B (c0_L1 → v0_L1)', GREEN)]
    lB2 = [
        f'out_B = W_n × h_c0_L1  +  W_s × h_v0_L1',
        f'      = 0.5 × {fv(V["h_c0_L1"])}  +  1.0 × {fv(V["h_v0_L1"])}',
        f'      = {fv(0.5 * V["h_c0_L1"])}  +  {fv(V["h_v0_L1"])}',
        f'      = {fv(V["out_B_v0_L2"])}',
    ]
    story += [calc_block(lB2, LGREEN, GREEN), sp(0.25)]

    story += [step_header('C', 'Edge Type C (v1_L1 → v0_L1)', BLUE)]
    lC2 = [
        f'out_C = W_n × h_v1_L1  +  W_s × h_v0_L1',
        f'      = 0.5 × {fv(V["h_v1_L1"])}  +  1.0 × {fv(V["h_v0_L1"])}',
        f'      = {fv(0.5 * V["h_v1_L1"])}  +  {fv(V["h_v0_L1"])}',
        f'      = {fv(V["out_C_v0_L2"])}',
    ]
    story += [calc_block(lC2, LBLUE, BLUE), sp(0.25)]

    story += [step_header('D', 'aggr="sum"  +  ReLU', PURPLE)]
    sum_raw2 = V['out_A_v0_L2'] + V['out_B_v0_L2'] + V['out_C_v0_L2']
    lD2 = [
        f'h_v0_L2_raw = out_A  +  out_B  +  out_C',
        f'           = {fv(V["out_A_v0_L2"])}',
        f'           + {fv(V["out_B_v0_L2"])}',
        f'           + {fv(V["out_C_v0_L2"])}',
        f'           = {fv(sum_raw2)}',
        '',
        f'h_v0_L2 = ReLU( {fv(sum_raw2)} )',
        f'        = {fv(V["h_v0_L2"])}   ← 全部 > 0，無變化',
    ]
    story += [calc_block(lD2, LPURPLE, PURPLE)]

    res2_rows = [[p(f'h_v0  after  Layer 2  =  {fv(V["h_v0_L2"])}', 'result')]]
    story += [sp(0.2), box(res2_rows, LGREEN, GREEN, pad=10)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Readout MLP
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('6', 'Readout MLP：預測 rate 與 watch_time'), sp(0.3)]

    story += [p('程式碼：hetero_gnn.py:85-90, 129-132', 'small'), sp(0.2)]
    code_mlp = calc_block([
        'self.head = nn.Sequential(',
        '    nn.Linear(d, 64),   nn.ReLU(),   nn.Dropout(dropout),',
        '    nn.Linear(64, 8),',
        ')',
        '# 輸出後：',
        'rates = torch.sigmoid(out[:, :7])    # 7個rate → (0,1)',
        'wt    = out[:, 7:8]                  # log-space watch_time',
    ], LGRAY, DGRAY)
    story += [code_mlp, sp(0.3)]

    story += [p('此玩具範例簡化為：d=3 → 2 個輸出（1個rate + 1個watch_time）', 'small'), sp(0.2)]

    # Show MLP weights
    story += [p('MLP 權重（玩具設定）', 'h3'), sp(0.1)]
    w_rows = [
        [p('W1  (2×3)', 'monob'), p(f'= [[0.3, 0.2, 0.1],\n   [0.1, 0.4, 0.2]]', 'mono'),
         p('W2  (2×2)', 'monob'), p(f'= [[0.5, -0.3],\n   [0.2,  0.6]]', 'mono')],
    ]
    wt_table = Table(w_rows, colWidths=[CW*0.15, CW*0.33, CW*0.15, CW*0.33])
    wt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LGREEN),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#6ee7b7')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story += [wt_table, sp(0.3)]

    # Forward pass
    story += [step_header('1', 'Linear(3→2)：z1 = W1 × h_v0_L2', BLUE)]
    l1_lines = [
        f'輸入：h_v0_L2 = {fv(V["h_v0_L2"])}',
        '',
        'z1[0] = 0.3 × 11.9125  +  0.2 × 4.4375  +  0.1 × 8.2958',
        f'      = {0.3*V["h_v0_L2"][0]:.4f}  +  {0.2*V["h_v0_L2"][1]:.4f}  +  {0.1*V["h_v0_L2"][2]:.4f}',
        f'      = {V["z1"][0]:.4f}',
        '',
        'z1[1] = 0.1 × 11.9125  +  0.4 × 4.4375  +  0.2 × 8.2958',
        f'      = {0.1*V["h_v0_L2"][0]:.4f}  +  {0.4*V["h_v0_L2"][1]:.4f}  +  {0.2*V["h_v0_L2"][2]:.4f}',
        f'      = {V["z1"][1]:.4f}',
        '',
        f'z1 = {fv(V["z1"])}',
    ]
    story += [sp(0.1), calc_block(l1_lines, LBLUE, BLUE), sp(0.25)]

    story += [step_header('2', 'ReLU', GREEN)]
    l2_lines = [
        f'z1_relu = ReLU({fv(V["z1"])})',
        f'        = {fv(V["z1r"])}   ← 全部 > 0，無變化',
    ]
    story += [calc_block(l2_lines, LGREEN, GREEN), sp(0.25)]

    story += [step_header('3', 'Linear(2→2)：out = W2 × z1_relu', AMBER)]
    l3_lines = [
        f'out[0] = 0.5 × {V["z1r"][0]:.4f}  +  (-0.3) × {V["z1r"][1]:.4f}',
        f'       = {0.5*V["z1r"][0]:.4f}  -  {0.3*V["z1r"][1]:.4f}',
        f'       = {V["out_mlp"][0]:.4f}',
        '',
        f'out[1] = 0.2 × {V["z1r"][0]:.4f}  +  0.6 × {V["z1r"][1]:.4f}',
        f'       = {0.2*V["z1r"][0]:.4f}  +  {0.6*V["z1r"][1]:.4f}',
        f'       = {V["out_mlp"][1]:.4f}',
    ]
    story += [calc_block(l3_lines, LAMBER, AMBER), sp(0.25)]

    story += [step_header('4', '最終激活：sigmoid (rate) + linear (watch_time)', PURPLE)]
    exp_neg = np.exp(-V['out_mlp'][0])
    l4_lines = [
        f'pred_rate = sigmoid({V["out_mlp"][0]:.4f})',
        f'          = 1 / (1 + exp(-{V["out_mlp"][0]:.4f}))',
        f'          = 1 / (1 + {exp_neg:.4f})',
        f'          = {V["pred_rate"]:.4f}  → 預測 like_rate = 77.9%',
        '',
        f'pred_log_wt = {V["pred_log_wt"]:.4f}  (log-space，直接輸出)',
        f'pred_wt     = expm1({V["pred_log_wt"]:.4f}) = e^{V["pred_log_wt"]:.4f} - 1',
        f'            = {V["pred_wt"]:.2f} 秒  ← 推論時才做 expm1，還原真實單位',
    ]
    story += [calc_block(l4_lines, LPURPLE, PURPLE)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Loss
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_header('7', 'Loss 計算：曝光量加權 MSE + Huber'), sp(0.3)]

    story += [p('程式碼：train_hetero.py:52-63', 'small'), sp(0.2)]
    code_loss = calc_block([
        'def weighted_loss(pred, target, weight):',
        '    w = weight / (weight.sum() + 1e-8)   # 正規化曝光量',
        '    w = w.unsqueeze(1)                    # [N, 1]',
        '    rate_loss = (w * (pred[:,:7] - target[:,:7])**2).sum()',
        '    wt_loss   = (w * F.huber_loss(pred[:,7:8], target[:,7:8],',
        '                                  delta=1.0, reduction="none")).sum()',
        '    return rate_loss + wt_loss',
    ], LGRAY, DGRAY)
    story += [code_loss, sp(0.3)]

    story += [p('範例（只有 v0，weight = 1.0）', 'h3'), sp(0.1)]

    # Truth vs pred table
    truth_data = [
        ['', '預測值', '真實值', '差值'],
        ['like_rate',    f'{V["pred_rate"]:.4f}',   f'{V["true_rate"]:.4f}',   f'{V["pred_rate"]-V["true_rate"]:+.4f}'],
        ['log(watch_time)', f'{V["pred_log_wt"]:.4f}', f'{V["true_log_wt"]:.4f}', f'{V["pred_log_wt"]-V["true_log_wt"]:+.4f}'],
    ]
    td_rows = [[p(c, 'th') if i == 0 else p(c, 'td') for c in row] for i, row in enumerate(truth_data)]
    td_t = Table(td_rows, colWidths=[CW*0.3, CW*0.23, CW*0.23, CW*0.22])
    td_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LAMBER, LBLUE]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
    ]))
    story += [td_t, sp(0.35)]

    story += [step_header('A', '7個 Rate 的加權 MSE（只計算 like_rate 作示範）', AMBER)]
    la = [
        f'rate_loss = w × (pred_rate - true_rate)²',
        f'          = 1.0 × ({V["pred_rate"]:.4f} - {V["true_rate"]:.4f})²',
        f'          = 1.0 × ({V["pred_rate"]-V["true_rate"]:+.4f})²',
        f'          = {V["rate_loss"]:.6f}',
    ]
    story += [sp(0.1), calc_block(la, LAMBER, AMBER), sp(0.25)]

    story += [step_header('B', 'Watch Time 的 Huber Loss（δ = 1.0）', BLUE)]
    lb = [
        f'diff_wt = |pred_log_wt - true_log_wt|',
        f'        = |{V["pred_log_wt"]:.4f} - {V["true_log_wt"]:.4f}|',
        f'        = {V["diff_wt"]:.4f}   →  < δ = 1.0，使用平方項',
        '',
        f'wt_loss = w × 0.5 × diff_wt²',
        f'        = 1.0 × 0.5 × {V["diff_wt"]:.4f}²',
        f'        = 1.0 × 0.5 × {V["diff_wt"]**2:.6f}',
        f'        = {V["wt_loss"]:.6f}',
        '',
        '（若 diff_wt ≥ 1.0，改用：wt_loss = w × (δ × diff_wt - 0.5 × δ²) → 線性增長，抗 outlier）',
    ]
    story += [calc_block(lb, LBLUE, BLUE), sp(0.25)]

    story += [step_header('C', '合併 Loss', PURPLE)]
    lc = [
        f'total_loss = rate_loss  +  wt_loss',
        f'           = {V["rate_loss"]:.6f}  +  {V["wt_loss"]:.6f}',
        f'           = {V["total_loss"]:.6f}',
        '',
        '接下來：total_loss.backward()  →  AdamW.step()  →  所有 W_n, W_s, W_proj,',
        '           cat_embed, MLP 的參數都被更新一次。',
    ]
    story += [calc_block(lc, LPURPLE, PURPLE)]
    story += [sp(0.3)]

    # ─── Summary Diagram ──────────────────────────────────────────────────────
    story += [hr(), p('全流程數值總覽 (v0)', 'h2'), sp(0.1)]
    summary_data = [
        ['階段', '輸入', '操作', '輸出 (v0)'],
        ['Input\nProjection',
         'x_v0: 819d',
         'Linear(-1→3)',
         f'h_v0_L0={fv(V["h_v"][0])}'],
        ['Layer 1\nSAGEConv',
         'h_v0_L0, h_a0, h_c0, h_v1',
         'W_n×MEAN+W_s×self\n(3種edge)  然後 ReLU',
         f'h_v0_L1={fv(V["h_v0_L1"])}'],
        ['Layer 2\nSAGEConv',
         'h_v0_L1, h_a0_L1,\nh_c0_L1, h_v1_L1',
         'W_n×MEAN+W_s×self\n(3種edge)  然後 ReLU',
         f'h_v0_L2={fv(V["h_v0_L2"])}'],
        ['Readout\nMLP',
         f'h_v0_L2',
         'L(3→2) → ReLU → L(2→2)',
         f'out=[{V["out_mlp"][0]:.4f}, {V["out_mlp"][1]:.4f}]'],
        ['Activation',
         f'out',
         'sigmoid / linear',
         f'rate={V["pred_rate"]:.4f}\nlog_wt={V["pred_log_wt"]:.4f}'],
        ['Loss',
         f'pred vs true',
         'MSE + Huber (weighted)',
         f'total = {V["total_loss"]:.6f}'],
    ]
    sm_rows = [[p(c, 'th') if i == 0 else p(c, 'tdl' if j in (1,2,3) else 'tdh') for j, c in enumerate(row)]
               for i, row in enumerate(summary_data)]
    sm_cw = [CW*0.14, CW*0.24, CW*0.30, CW*0.30]
    sm_t = Table(sm_rows, colWidths=sm_cw)
    sm_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LGRAY, LBLUE, LBLUE, LGREEN, LAMBER, LRED]),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ]))
    story += [sm_t]

    note_last = [[p(
        '本文所有數值均由 Python/NumPy 精確計算（scripts/hetero_gnn.py 實際邏輯）。\n'
        'W_n=0.5×I 和 W_s=1.0×I 為手算簡化，真實訓練中這些參數由 AdamW 自動更新。\n'
        '真實模型：d=128、layers=2、N_video=54,088、8個預測目標。', 'small')]]
    story += [sp(0.3), box(note_last, LGRAY, DGRAY, pad=8)]

    return story


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    out_path = Path(__file__).parent / 'graphsage_walkthrough.pdf'
    V = compute()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title='GraphSAGE 完整手算範例',
        author='Pipeline C',
        subject='HeteroGNN Step-by-Step Numerical Walkthrough',
    )

    story = build_story(V)
    doc.build(story)
    print(f'PDF saved → {out_path}')


if __name__ == '__main__':
    main()
