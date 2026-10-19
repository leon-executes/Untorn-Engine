"""Build the original landscape-and-enlargement icon from vector geometry."""

from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    scale = 4
    im = Image.new("RGBA", (512 * scale, 512 * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def rect(box, radius, fill, outline=None, width=1):
        d.rounded_rectangle(
            tuple(v * scale for v in box),
            radius=radius * scale,
            fill=fill,
            outline=outline,
            width=width * scale,
        )

    def poly(points, fill):
        d.polygon([(x * scale, y * scale) for x, y in points], fill=fill)

    rect((12, 12, 500, 500), 100, "#10151c", "#303945", 3)
    rect((96, 151, 383, 396), 30, "#202c35")
    poly([(120, 365), (201, 254), (259, 322), (304, 280), (359, 365)], "#b4d9d5")
    d.ellipse((282 * scale, 198 * scale, 326 * scale, 242 * scale), fill="#eee9df")
    rect((96, 151, 383, 396), 30, None, "#eee9df", 15)
    d.line(
        [(321 * scale, 194 * scale), (410 * scale, 105 * scale)], fill="#10151c", width=42 * scale
    )
    d.line(
        [(321 * scale, 194 * scale), (410 * scale, 105 * scale)], fill="#ff5148", width=24 * scale
    )
    d.line(
        [(340 * scale, 105 * scale), (410 * scale, 105 * scale), (410 * scale, 175 * scale)],
        fill="#ff5148",
        width=24 * scale,
    )
    im = im.resize((512, 512), Image.Resampling.LANCZOS)
    (ROOT / "build").mkdir(exist_ok=True)
    im.save(ROOT / "build/mark-preview.png")
    im.save(
        ROOT / "source/untorn.ico",
        sizes=[(n, n) for n in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)],
    )
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><title>Untorn Engine — image enlargement</title><rect x="12" y="12" width="488" height="488" rx="100" fill="#10151c" stroke="#303945" stroke-width="3"/><rect x="96" y="151" width="287" height="245" rx="30" fill="#202c35"/><path d="M120 365L201 254L259 322L304 280L359 365Z" fill="#b4d9d5"/><circle cx="304" cy="220" r="22" fill="#eee9df"/><rect x="96" y="151" width="287" height="245" rx="30" fill="none" stroke="#eee9df" stroke-width="15"/><path d="M321 194L410 105" stroke="#10151c" stroke-width="42"/><path d="M321 194L410 105M340 105H410V175" fill="none" stroke="#ff5148" stroke-width="24" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
    (ROOT / "docs/mark.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
