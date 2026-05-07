from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from edgevision.schemas import InspectionReport, QualityStatus


STATUS_COLORS = {
    QualityStatus.OK: (38, 166, 91),
    QualityStatus.NG: (220, 53, 69),
    QualityStatus.REVIEW: (255, 193, 7),
}


def annotate_report(image: Image.Image, report: InspectionReport) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for item in report.items:
        bbox = item.detection.bbox.to_int_list()
        color = STATUS_COLORS.get(item.quality.status, (255, 255, 255))
        draw.rectangle(bbox, outline=color, width=3)
        label = f"{item.item_id}: {item.identity.label} {item.quality.status.value}"
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        bg = [bbox[0], max(0, bbox[1] - text_height - 4), bbox[0] + (text_bbox[2] - text_bbox[0]) + 4, bbox[1]]
        draw.rectangle(bg, fill=color)
        draw.text((bg[0] + 2, bg[1] + 2), label, fill=(0, 0, 0), font=font)

    return annotated

