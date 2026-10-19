"""Generate the Untorn vector monogram and Windows multi-resolution icon.

Both assets share one set of coordinates. Windows frames use a tighter optical
margin so the monogram remains legible at taskbar and title-bar sizes.
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
IVORY = "#eee9df"
RED = "#ef5148"
BODY = (
    (112, 96),
    (168, 96),
    (168, 304),
    (208, 344),
    (304, 344),
    (344, 304),
    (344, 96),
    (400, 96),
    (400, 326),
    (326, 400),
    (186, 400),
    (112, 326),
)
ACCENT = ((344, 96), (400, 96), (400, 176), (344, 176))
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def vector_path(points):
    return "M" + "L".join(f"{x} {y}" for x, y in points) + "Z"


def raster(size, *, compact):
    scale = 4
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    extent = 384 if compact else 512
    for points, colour in ((BODY, IVORY), (ACCENT, RED)):
        mapped = [
            (
                (x - 256 + extent / 2) * size * scale / extent,
                (y - 248 + extent / 2) * size * scale / extent,
            )
            for x, y in points
        ]
        draw.polygon(mapped, fill=colour)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main():
    (ROOT / "build").mkdir(exist_ok=True)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" '
        'role="img" aria-labelledby="title">\n'
        '  <title id="title">Untorn Engine</title>\n'
        f'  <path fill="{IVORY}" d="{vector_path(BODY)}"/>\n'
        f'  <path fill="{RED}" d="{vector_path(ACCENT)}"/>\n'
        "</svg>\n"
    )
    (ROOT / "docs/mark.svg").write_text(svg, encoding="utf-8")
    raster(512, compact=False).save(ROOT / "build/mark-preview.png")
    frames = [raster(size, compact=True) for size in SIZES]
    frames[-1].save(
        ROOT / "source/untorn.ico",
        format="ICO",
        sizes=[(n, n) for n in SIZES],
        append_images=frames[:-1],
    )


if __name__ == "__main__":
    main()
