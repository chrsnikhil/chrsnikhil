"""Render subject.png as an animated ASCII SVG for the profile README.

    python art/build_svg.py              # writes art/skull.svg
    python art/build_svg.py --seed 42    # re-roll where and when the effects fire

GitHub strips scripts from READMEs, so everything moves with SMIL: static ASCII layers
revealed and covered by animated masks, clips and overlays. The timers run on periods that
don't divide into each other, so the sequence takes a very long time to repeat.
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
COLS = 120
CW, LH, FS, BASE = 6, 11, 10, 8          # cell width, line height, font size, baseline (SVG units)
PAD = 24
RAMP = ' .:-=+*o#%@'
EDGE = '-\\|/'                            # edge direction 0, 45, 90, 135 degrees (y down)
SCR = '01<>/\\|{}[]#$%&*+=?'
BG = '#06080b'
ACC = '#00d9ff'


# ---------- image -> per-cell fields ----------

def grid_fields(cols):
    rows = round(cols * CW / LH)
    rgba = np.asarray(Image.open(HERE / 'subject.png').convert('RGBA')).astype(np.float32) / 255
    a = rgba[..., 3]
    pm = [rgba[..., c] * a for c in range(3)]

    def area(ch):                                     # box-filter down to one value per cell
        return np.asarray(Image.fromarray(ch, 'F').resize((cols, rows), Image.BOX))

    R, G, B, A = (area(c) for c in (*pm, a))
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    v = np.maximum(np.power(np.clip(L / 0.6, 0, 1), 0.72), A * 0.16)

    def shift(p, dy, dx):
        return p[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]

    p = np.pad(v, 1, mode='edge')
    bl = sum(shift(p, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
    p = np.pad(bl, 1, mode='edge')
    gx = ((shift(p, -1, 1) + 2 * shift(p, 0, 1) + shift(p, 1, 1))
          - (shift(p, -1, -1) + 2 * shift(p, 0, -1) + shift(p, 1, -1))) * 0.25
    gy = ((shift(p, 1, -1) + 2 * shift(p, 1, 0) + shift(p, 1, 1))
          - (shift(p, -1, -1) + 2 * shift(p, -1, 0) + shift(p, -1, 1))) * 0.25
    gy *= CW / LH                                     # cells are taller than wide
    em = np.hypot(gx, gy)

    # non-max suppression: keep an edge only where it peaks across its own width
    q = (((np.degrees(np.arctan2(gy, gx)) + 180 + 22.5) % 180) // 45).astype(int)
    pe = np.pad(em, 1)
    edge = np.zeros_like(em, dtype=bool)
    for k, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        m = (q == k) & (em >= 0.11) & (em >= shift(pe, dy, dx)) & (em >= shift(pe, -dy, -dx))
        edge |= m
    ang = (((np.degrees(np.arctan2(gx, -gy)) + 180 + 22.5) % 180) // 45).astype(int)

    return dict(rows=rows, cols=cols, R=R, G=G, B=B, A=A, v=v, gx=gx, gy=gy, em=em, edge=edge, ang=ang)


def hex4(r, g, b):
    q = lambda x: format(int(round(min(1.0, max(0.0, x)) * 15)), 'x')
    return f'#{q(r)}{q(g)}{q(b)}'


# ---------- layers (lists of (col, row, char, fill)) ----------

def base_layer(f):
    """Shaded bone, lit by a fixed key light from the upper left."""
    lx, ly, lz = np.array([-0.45, -0.55, 0.70]) / np.linalg.norm([-0.45, -0.55, 0.70])
    Nx, Ny = -f['gx'] * 3.2, -f['gy'] * 3.2
    ndl = (Nx * lx + Ny * ly + lz) / np.sqrt(Nx * Nx + Ny * Ny + 1)
    sh = 0.72 + 0.5 * np.maximum(ndl, 0)
    vv = np.clip(f['v'] * sh, 0, 1)
    out = []
    for j in range(f['rows']):
        for i in range(f['cols']):
            a = f['A'][j, i]
            if a < 0.04:
                continue
            if f['edge'][j, i] and vv[j, i] > 0.12:
                ch = EDGE[f['ang'][j, i]]
            else:
                ch = RAMP[min(len(RAMP) - 1, int(vv[j, i] * len(RAMP)))]
            if ch == ' ':
                continue
            ia = 1 / (a * 0.58)
            c = [(min(1, f[k][j, i] * ia) ** 0.75 * 0.95 + off) * sh[j, i]
                 for k, off in (('R', 0.08), ('G', 0.08), ('B', 0.1))]
            out.append((i, j, ch, hex4(*c)))
    return out


def xray_layer(f):
    """Bone outlines in cyan; the beanie and the lenses go see-through."""
    out = []
    for j in range(f['rows']):
        for i in range(f['cols']):
            v, em = f['v'][j, i], f['em'][j, i]
            if (f['edge'][j, i] or em > 0.09) and v > 0.3:
                s = min(1.0, 0.55 + em * 5)
                out.append((i, j, EDGE[f['ang'][j, i]], hex4(0.15 * s, 0.85 * s, s)))
            elif v > 0.32:
                x = (v - 0.32) / 0.68
                out.append((i, j, RAMP[1 + min(4, int(x * 5))], hex4(0, 0.18 + x * 0.35, 0.22 + x * 0.4)))
    return out


def subject_cells(f, thresh=0.04, grow=1):
    m = f['A'] > thresh
    for _ in range(grow):
        p = np.pad(m, 1)
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return [(i, j) for j, i in zip(*np.nonzero(m))]


# ---------- SVG helpers ----------

def esc(c):
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;'}.get(c, c)


def rows_text(cells):
    """One <text> per (row, colour) with a per-glyph x list: compact, and layout never
    depends on which monospace font the viewer's machine falls back to."""
    groups = {}
    for i, j, ch, fill in cells:
        groups.setdefault((j, fill), []).append((i, ch))
    out = []
    for (j, fill), items in sorted(groups.items()):
        items.sort()
        xs = ' '.join(str(i * CW) for i, _ in items)
        out.append(f'<text x="{xs}" y="{j * LH + BASE}" fill="{fill}">{"".join(esc(c) for _, c in items)}</text>')
    return '\n'.join(out)


def scatter_text(cells, fill, attrs=''):
    """Glyphs at arbitrary cells in one <text>, positioned with x and y lists."""
    xs = ' '.join(str(i * CW) for i, j, _ in cells)
    ys = ' '.join(str(j * LH + BASE) for i, j, _ in cells)
    return f'<text x="{xs}" y="{ys}" fill="{fill}"{attrs}>{"".join(esc(c) for _, _, c in cells)}</text>'


def loop_path(pts):
    """Closed Catmull-Rom spline through pts, as cubic Beziers."""
    n = len(pts)
    d = f'M{pts[0][0]:.0f},{pts[0][1]:.0f}'
    for k in range(n):
        p0, p1, p2, p3 = pts[(k - 1) % n], pts[k], pts[(k + 1) % n], pts[(k + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f' C{c1[0]:.0f},{c1[1]:.0f} {c2[0]:.0f},{c2[1]:.0f} {p2[0]:.0f},{p2[1]:.0f}'
    return d + 'Z'


def fmt(xs):
    return ';'.join(f'{x:.4g}' for x in xs)


# ---------- build ----------

def build(seed):
    rnd = random.Random(seed)
    f = grid_fields(COLS)
    rows, cols = f['rows'], f['cols']
    W, H = cols * CW, rows * LH
    face = [(i, j) for j, i in zip(*np.nonzero(f['A'] > 0.5))]
    bone = [(i, j) for j, i in zip(*np.nonzero((f['A'] > 0.5) & (f['v'] > 0.45)))]
    pt = lambda c: (c[0] * CW + CW / 2, c[1] * LH + LH / 2)

    base = base_layer(f)
    xray = xray_layer(f)
    halo = subject_cells(f, grow=1)
    scr_a = [(i, j, rnd.choice(SCR)) for i, j in halo]
    scr_b = [(i, j, rnd.choice(SCR)) for i, j in halo]

    # glint: '/' over the dark sunglass lenses
    glint = [(i, j, '/') for j in range(rows) for i in range(cols)
             if 0.26 < (i + .5) / cols < 0.74 and 0.43 < (j + .5) / rows < 0.555
             and f['A'][j, i] > 0.3 and f['v'][j, i] < 0.6]

    # background specks in 8 twinkle groups
    empty = [(i, j) for j in range(rows) for i in range(cols) if f['A'][j, i] < 0.02]
    specks = rnd.sample(empty, int(len(empty) * 0.03))
    dust = []
    for g in range(8):
        cells = [(i, j, '+' if rnd.random() < 0.08 else '.') for i, j in specks[g::8]]
        dur, beg = rnd.uniform(4, 12), rnd.uniform(0, 6)
        dust.append(scatter_text(cells, '#2a3442', ' opacity="0.2"') .replace(
            '</text>', f'<animate attributeName="opacity" values="0.15;0.9;0.15" dur="{dur:.2f}s" '
                       f'begin="{1.5 + beg:.2f}s" repeatCount="indefinite"/></text>'))

    # glitch sparks: 12 groups of cells that blink cyan for a few frames
    sparks = []
    for g in range(12):
        cells = [(i, j, rnd.choice(SCR)) for i, j in rnd.sample(face, 6)]
        dur, beg, on = rnd.uniform(2.5, 9.5), rnd.uniform(2, 9), rnd.uniform(0.08, 0.2)
        covers = ''.join(f'M{i * CW},{j * LH}h{CW}v{LH}h-{CW}z' for i, j, _ in cells)
        sparks.append(
            f'<g opacity="0"><animate attributeName="opacity" values="1;0" keyTimes="0;{on / dur:.4f}" '
            f'calcMode="discrete" dur="{dur:.2f}s" begin="{beg:.2f}s" repeatCount="indefinite"/>'
            f'<path d="{covers}" fill="{BG}"/>{scatter_text(cells, ACC)}</g>')

    # shockwave emitters: each fires once per period, somewhere new each time
    rings, ring_begins = [], []
    for k, period in enumerate((6.7, 9.1, 12.3)):
        spots = [pt(rnd.choice(bone)) for _ in range(7)]
        beg = rnd.uniform(2.2, 2.2 + period)
        ring_begins.append((period, round(beg, 2)))
        grow = 1.6 / period
        rmax = W * rnd.uniform(0.35, 0.5)
        rings.append(
            f'<circle id="ring{k}" r="0" fill="none" stroke-width="10">'
            f'<animate attributeName="cx" values="{fmt(p[0] for p in spots)}" calcMode="discrete" dur="{7 * period}s" begin="{beg:.2f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{fmt(p[1] for p in spots)}" calcMode="discrete" dur="{7 * period}s" begin="{beg:.2f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="r" values="0;{rmax:.0f};{rmax:.0f}" keyTimes="0;{grow:.4f};1" dur="{period}s" begin="{beg:.2f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="stroke-width" values="8;26;26" keyTimes="0;{grow:.4f};1" dur="{period}s" begin="{beg:.2f}s" repeatCount="indefinite"/>'
            f'<animate attributeName="stroke-opacity" values="1;0;0" keyTimes="0;{grow:.4f};1" dur="{period}s" begin="{beg:.2f}s" repeatCount="indefinite"/>'
            f'</circle>')

    # x-ray lens: wanders the face on a 31 s loop, opens twice per loop
    LD = 31.0
    lens_pts = [pt(rnd.choice(bone)) for _ in range(6)]
    lens_path = loop_path(lens_pts)
    neg_path = loop_path([(-x, -y) for x, y in lens_pts])
    R, RIM = W * 0.14, LH * 1.3
    times, rad = [0.0], [0.0]
    for lo, hi in ((3, 10), (17, 24)):
        s = rnd.uniform(lo, hi)
        times += [s, s + 0.5, s + 3.8, s + 4.3]
        rad += [0, R, R, 0]
    times.append(LD)
    rad.append(0)
    kt = fmt(t / LD for t in times)
    lens_anim = lambda attr, vals: (f'<animate attributeName="{attr}" values="{fmt(vals)}" keyTimes="{kt}" '
                                    f'dur="{LD}s" repeatCount="indefinite"/>')
    motion = f'<animateMotion path="{lens_path}" dur="{LD}s" repeatCount="indefinite"/>'
    neg_motion = f'<animateMotion path="{neg_path}" dur="{LD}s" repeatCount="indefinite"/>'
    open_ = [1 if r > 0 else 0 for r in rad]

    # spotlight: wanders on its own 23 s loop
    spot_path = loop_path([(W * rnd.uniform(0.2, 0.8), H * rnd.uniform(0.2, 0.8)) for _ in range(7)])

    # glint band sweeps across the lenses every 7.3 s
    gx0, gx1 = W * 0.2, W * 0.8
    gy0, gy1 = H * 0.42, H * 0.57
    band = f'M0,{gy1:.0f} L{CW * 4},{gy1:.0f} L{CW * 4 + (gy1 - gy0) * 0.6:.0f},{gy0:.0f} L{(gy1 - gy0) * 0.6:.0f},{gy0:.0f}Z'

    diag = math.hypot(W, H)
    cx0, cy0 = W / 2, H * 0.55

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W + 2 * PAD} {H + 2 * PAD}" width="{W + 2 * PAD}" height="{H + 2 * PAD}" role="img" aria-label="ASCII skull">
<style>text{{font-family:Consolas,"Cascadia Mono",Menlo,"DejaVu Sans Mono","Liberation Mono",monospace;font-size:{FS}px;font-weight:700}}</style>
<defs>
<clipPath id="card"><rect width="{W + 2 * PAD}" height="{H + 2 * PAD}" rx="14"/></clipPath>
<clipPath id="intro"><circle cx="{cx0:.0f}" cy="{cy0:.0f}" r="0"><animate attributeName="r" values="0;{diag:.0f}" dur="1.8s" begin="0.2s" fill="freeze"/></circle></clipPath>
<g id="rings">{''.join(rings)}</g>
<circle id="lensRim" r="0" fill="none" stroke-width="0">{motion}{lens_anim('r', [r + RIM / 2 if r else 0 for r in rad])}{lens_anim('stroke-width', [RIM * o for o in open_])}</circle>
<clipPath id="lensClip"><circle r="0">{motion}{lens_anim('r', rad)}</circle></clipPath>
<mask id="scrMask" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">
<use href="#rings" xlink:href="#rings" stroke="#fff"/>
<use href="#lensRim" xlink:href="#lensRim" stroke="#fff"/>
<circle cx="{cx0:.0f}" cy="{cy0:.0f}" r="0" fill="none" stroke="#fff" stroke-width="30"><animate attributeName="r" values="0;{diag:.0f}" dur="1.8s" begin="0.2s" fill="freeze"/><animate attributeName="stroke-opacity" values="1;1;0" keyTimes="0;0.8;1" dur="2s" begin="0.2s" fill="freeze"/></circle>
</mask>
<mask id="glintMask" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">
<path d="{band}" fill="#fff"><animateTransform attributeName="transform" type="translate" values="{gx0 - 60:.0f},0;{gx1:.0f},0;{gx1:.0f},0" keyTimes="0;0.12;1" dur="7.3s" begin="3.1s" repeatCount="indefinite"/></path>
</mask>
<radialGradient id="shade"><stop offset="0" stop-color="{BG}" stop-opacity="0"/><stop offset="0.22" stop-color="{BG}" stop-opacity="0"/><stop offset="0.65" stop-color="{BG}" stop-opacity="0.26"/><stop offset="1" stop-color="{BG}" stop-opacity="0.38"/></radialGradient>
<radialGradient id="glow"><stop offset="0" stop-color="#ffd9a0" stop-opacity="0.09"/><stop offset="1" stop-color="#ffd9a0" stop-opacity="0"/></radialGradient>
</defs>
<g clip-path="url(#card)">
<rect width="{W + 2 * PAD}" height="{H + 2 * PAD}" fill="{BG}"/>
<g transform="translate({PAD},{PAD})">
<g transform="translate({W / 2:.0f},{H / 2:.0f})"><g><animateTransform attributeName="transform" type="skewX" values="0;0.8;0;-0.8;0" keyTimes="0;0.25;0.5;0.75;1" calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1;.45 0 .55 1;.45 0 .55 1" dur="14s" repeatCount="indefinite"/><g transform="translate({-W / 2:.0f},{-H / 2:.0f})">
<g opacity="0"><animate attributeName="opacity" values="0;1" dur="1.5s" begin="0.6s" fill="freeze"/>
{chr(10).join(dust)}
</g>
<g clip-path="url(#intro)">
{rows_text(base)}
</g>
{''.join(sparks)}
<g mask="url(#glintMask)">{scatter_text(glint, '#f4f8ff')}</g>
<use href="#rings" xlink:href="#rings" stroke="{BG}"/>
<circle r="0" fill="{BG}">{motion}{lens_anim('r', [r + RIM if r else 0 for r in rad])}</circle>
<g clip-path="url(#lensClip)"><g>{motion}<g transform="scale(1.3)"><g>{neg_motion}
{rows_text(xray)}
</g></g></g></g>
<g mask="url(#scrMask)">
<g>{scatter_text(scr_a, ACC)}<animate attributeName="opacity" values="1;0" calcMode="discrete" dur="0.24s" repeatCount="indefinite"/></g>
<g opacity="0">{scatter_text(scr_b, ACC)}<animate attributeName="opacity" values="0;1" calcMode="discrete" dur="0.24s" repeatCount="indefinite"/></g>
</g>
<rect y="{-PAD - 20}" width="{W}" height="{LH * 1.2:.0f}" fill="{ACC}" opacity="0.06"><animate attributeName="y" values="{-PAD - 20};{H + PAD + 20};{H + PAD + 20}" keyTimes="0;0.35;1" dur="9.7s" begin="2.4s" repeatCount="indefinite"/></rect>
<rect x="{-W}" y="{-H}" width="{2 * W}" height="{2 * H}" fill="url(#shade)"><animateMotion path="{spot_path}" dur="23s" repeatCount="indefinite"/></rect>
<rect x="{-W * 0.25:.0f}" y="{-W * 0.25:.0f}" width="{W * 0.5:.0f}" height="{W * 0.5:.0f}" fill="url(#glow)" style="mix-blend-mode:screen"><animateMotion path="{spot_path}" dur="23s" repeatCount="indefinite"/></rect>
</g></g></g>
</g>
</g>
</svg>
'''
    schedule = dict(lens_open=[round(times[k], 2) for k in (1, 5)], ring_begin=ring_begins)
    return svg, dict(cells=rows * cols, base=len(base), xray=len(xray), scramble=len(halo), **schedule)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=str(HERE / 'skull.svg'))
    args = ap.parse_args()
    svg, stats = build(args.seed)
    Path(args.out).write_text(svg, encoding='utf-8')
    print(f'{args.out}: {len(svg.encode()) / 1024:.0f} KB', stats)
