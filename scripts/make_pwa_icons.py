#!/usr/bin/env python3
"""Generate the NeuraMesh PWA icons (pure stdlib — no Pillow).

Draws the mesh mark — five nodes joined by edges on a dark field — at
512 px and box-downsamples to the sizes iOS/Android want. Output goes to
web/public/ so Vite copies the files into the built app verbatim.

Usage: python3 scripts/make_pwa_icons.py
"""

import os
import struct
import zlib

SIZE = 512
BG = (11, 18, 32)          # --color background: dark navy
EDGE = (34, 88, 130)       # dimmed edge lines
NODE = (56, 189, 248)      # cyan nodes (matches the UI accent)
CORE = (14, 165, 233)      # center node

# Node layout (relative coords in a 0..1 square).
NODES = [
    (0.50, 0.50, 0.085),   # coordinator, slightly larger
    (0.24, 0.28, 0.060),
    (0.76, 0.26, 0.060),
    (0.22, 0.74, 0.060),
    (0.74, 0.76, 0.060),
]
EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]
EDGE_WIDTH = 0.022


def dist_to_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denominator = abx * abx + aby * aby or 1e-9
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denominator))
    dx, dy = px - (ax + t * abx), py - (ay + t * aby)
    return (dx * dx + dy * dy) ** 0.5


def blend(base, top, alpha):
    return tuple(int(b + (t - b) * alpha) for b, t in zip(base, top))


def render(size):
    pixels = []
    inv = 1.0 / size
    for y in range(size):
        row = bytearray()
        for x in range(size):
            u, v = (x + 0.5) * inv, (y + 0.5) * inv
            color = BG

            # Edges (soft 1.5 px feather).
            for a, b in EDGES:
                ax, ay, _ = NODES[a]
                bx, by, _ = NODES[b]
                d = dist_to_segment(u, v, ax, ay, bx, by)
                if d < EDGE_WIDTH:
                    feather = min(1.0, (EDGE_WIDTH - d) / (1.5 * inv))
                    color = blend(color, EDGE, feather)

            # Nodes on top.
            for index, (nx, ny, radius) in enumerate(NODES):
                d = ((u - nx) ** 2 + (v - ny) ** 2) ** 0.5
                if d < radius:
                    feather = min(1.0, (radius - d) / (1.5 * inv))
                    tone = CORE if index == 0 else NODE
                    color = blend(color, tone, feather)

            row += bytes(color)
        pixels.append(bytes(row))
    return pixels


def downsample(pixels, src, dst):
    """Box filter: src must be a multiple-ish of dst; averages blocks."""
    out = []
    for y in range(dst):
        row = bytearray()
        y0, y1 = y * src // dst, max(y * src // dst + 1, (y + 1) * src // dst)
        for x in range(dst):
            x0, x1 = x * src // dst, max(x * src // dst + 1, (x + 1) * src // dst)
            r = g = b = n = 0
            for sy in range(y0, y1):
                src_row = pixels[sy]
                for sx in range(x0, x1):
                    r += src_row[sx * 3]
                    g += src_row[sx * 3 + 1]
                    b += src_row[sx * 3 + 2]
                    n += 1
            row += bytes((r // n, g // n, b // n))
        out.append(bytes(row))
    return out


def write_png(path, pixels, size):
    def chunk(kind, data):
        payload = kind + data
        return (struct.pack(">I", len(data)) + payload
                + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + row for row in pixels)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))
    print(f"wrote {path} ({size}x{size})")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "web", "public")
    os.makedirs(out_dir, exist_ok=True)
    base = render(SIZE)
    write_png(os.path.join(out_dir, "icon-512.png"), base, SIZE)
    for size, name in [(192, "icon-192.png"), (180, "apple-touch-icon.png")]:
        write_png(os.path.join(out_dir, name), downsample(base, SIZE, size), size)


if __name__ == "__main__":
    main()
