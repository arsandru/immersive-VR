import base64
import re
from pathlib import Path

import cairosvg


BASE_DIR = Path(__file__).resolve().parent
LEFT_SVG = BASE_DIR / "semantic_projection_primary_faceted.svg"
RIGHT_SVG = BASE_DIR / "semantic_projection_primary_final.svg"
OUTPUT_SVG = BASE_DIR / "semantic_projection_publication_figure.svg"
OUTPUT_PDF = BASE_DIR / "semantic_projection_publication_figure.pdf"


def parse_svg_size(svg_text: str) -> tuple[float, float]:
    viewbox_match = re.search(r'viewBox=["\']([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)["\']', svg_text)
    if viewbox_match:
        return float(viewbox_match.group(3)), float(viewbox_match.group(4))

    width_match = re.search(r'width=["\']([\d.]+)(?:pt|px)?["\']', svg_text)
    height_match = re.search(r'height=["\']([\d.]+)(?:pt|px)?["\']', svg_text)
    if width_match and height_match:
        return float(width_match.group(1)), float(height_match.group(1))

    raise ValueError("Could not determine SVG size from viewBox or width/height.")


def svg_to_data_uri(path: Path) -> tuple[str, float, float]:
    svg_text = path.read_text(encoding="utf-8")
    width, height = parse_svg_size(svg_text)
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}", width, height


def build_publication_figure() -> None:
    left_uri, left_w, left_h = svg_to_data_uri(LEFT_SVG)
    right_uri, right_w, right_h = svg_to_data_uri(RIGHT_SVG)

    panel_height = 420
    left_panel_w = panel_height * (left_w / left_h)
    right_panel_w = panel_height * (right_w / right_h)
    panel_width = max(left_panel_w, right_panel_w)

    margin = 40
    gutter = 24
    label_x = margin
    first_label_y = 26
    first_panel_y = 36
    second_label_y = first_panel_y + panel_height + gutter
    second_panel_y = second_label_y + 10
    figure_w = margin + panel_width + margin
    figure_h = second_panel_y + panel_height + margin

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{figure_w:.1f}pt" height="{figure_h:.1f}pt" viewBox="0 0 {figure_w:.1f} {figure_h:.1f}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{label_x}" y="{first_label_y}" font-family="Helvetica, Arial, sans-serif" font-size="12">a</text>
  <image x="{margin + (panel_width - left_panel_w) / 2:.1f}" y="{first_panel_y}" width="{left_panel_w:.1f}" height="{panel_height}" preserveAspectRatio="xMidYMid meet" xlink:href="{left_uri}"/>
  <text x="{label_x}" y="{second_label_y}" font-family="Helvetica, Arial, sans-serif" font-size="12">b</text>
  <image x="{margin + (panel_width - right_panel_w) / 2:.1f}" y="{second_panel_y}" width="{right_panel_w:.1f}" height="{panel_height}" preserveAspectRatio="xMidYMid meet" xlink:href="{right_uri}"/>
</svg>
"""
    OUTPUT_SVG.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(OUTPUT_PDF))
    print(f"Saved publication figure to {OUTPUT_SVG}")
    print(f"Saved publication figure to {OUTPUT_PDF}")


if __name__ == "__main__":
    build_publication_figure()
