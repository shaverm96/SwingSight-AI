from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 44
CONTENT_WIDTH = PAGE_WIDTH - (MARGIN * 2)

FOREST = HexColor("#073820")
FOREST_DARK = HexColor("#042916")
GREEN = HexColor("#21924A")
BRIGHT_GREEN = HexColor("#78D65B")
INK = HexColor("#17231B")
MUTED = HexColor("#5F6E63")
PALE_GREEN = HexColor("#EEF8EA")
PALE_CARD = HexColor("#F7FBF5")
DIVIDER = HexColor("#DCE7DC")
AMBER = HexColor("#B77A17")
SOFT_RED = HexColor("#B95042")
WHITE = HexColor("#FFFFFF")

SUSPICIOUS_ZERO_METRICS = {
    "tempo_estimate",
    "balance_score",
    "shoulder_turn_proxy",
    "hip_turn_proxy",
    "head_movement_cm",
    "head_stability",
}

SCORE_METRICS = (
    (
        "kinematic_sequence_score",
        "Kinematic Sequence",
        "How efficiently movement transfers from the lower body through the upper body and arms.",
    ),
    (
        "lateral_weight_shift_score",
        "Lateral Weight Shift",
        "A video-based estimate of how your pressure and body move toward the lead side.",
    ),
    (
        "spine_maintenance_score",
        "Spine Maintenance",
        "How consistently you maintain posture and spinal tilt through the swing.",
    ),
    (
        "x_factor_score",
        "X-Factor",
        "An estimate of the separation between the upper body and hips during the swing.",
    ),
)

MEASUREMENT_METRICS = (
    ("spine_angle_deg", "Spine Angle", "degrees"),
    ("spine_angle_variation_deg", "Spine Angle Variation", "degrees"),
    ("shoulder_turn_deg", "Shoulder Rotation", "degrees"),
    ("head_movement_px", "Head Movement", "pixels"),
    ("hip_movement_px", "Hip Movement", "pixels"),
    ("knee_movement_px", "Knee Movement", "pixels"),
    ("foot_stability_px", "Foot Stability", "pixels"),
    ("hand_path_px", "Hand Path", "pixels"),
    ("lateral_shift_ratio", "Lateral Shift", "ratio"),
)


def _ensure_reports_dir(reports_dir: str) -> Path:
    output = Path(reports_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def safe_number(value: Any, decimals: int = 1) -> float | None:
    """Return a finite number rounded for safe presentation, or ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, decimals)


def is_valid_metric(value: Any) -> bool:
    """Whether a value can be shown without exposing null or invalid data."""
    return safe_number(value) is not None


def format_score(value: Any) -> str:
    score = safe_number(value, decimals=0)
    if score is None:
        return "Not available"
    return f"{int(max(0, min(100, score)))} / 100"


def format_degrees(value: Any) -> str:
    number = safe_number(value, decimals=1)
    return "Not available" if number is None else f"{abs(number):.1f}\N{DEGREE SIGN}"


def format_pixels(value: Any) -> str:
    number = safe_number(value, decimals=0)
    return "Not available" if number is None else f"{int(round(abs(number))):,} px"


def format_ratio(value: Any) -> str:
    number = safe_number(value, decimals=2)
    return "Not available" if number is None else f"{number:.2f}"


def humanize_metric_name(name: str) -> str:
    labels = {
        key: label for key, label, _ in SCORE_METRICS
    } | {
        key: label for key, label, _ in MEASUREMENT_METRICS
    }
    if name in labels:
        return labels[name]
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip().title()


def build_report_filename(club_name: Any) -> str:
    """Create a player-friendly, filesystem-safe PDF filename from the club."""
    club = sanitize_text(club_name)
    if not club or club.lower() in {"unknown", "not detected", "none", "n/a", "not available"}:
        return "SwingSight_Swing_Report.pdf"
    club = re.sub(r"[^A-Za-z0-9]+", "-", club).strip("-")
    return f"SwingSight_{club}_Swing_Report.pdf" if club else "SwingSight_Swing_Report.pdf"


def sanitize_text(value: Any) -> str:
    """Normalize untrusted coaching text for predictable PDF layout."""
    if isinstance(value, dict):
        value = value.get("description") or value.get("text") or value.get("title") or ""
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u2022": "-",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = " ".join(text.split())
    return text.encode("latin-1", "replace").decode("latin-1")


def _usable_metric(name: str, value: Any) -> float | None:
    number = safe_number(value, decimals=4)
    if number is None:
        return None
    if name in SUSPICIOUS_ZERO_METRICS and number == 0:
        return None
    return number


def _club_name(result: Dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    return sanitize_text(summary.get("club") or result.get("club") or result.get("detected_club") or "Not detected")


def _score_value(result: Dict[str, Any]) -> int | None:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    score = safe_number(summary.get("swing_score", result.get("swing_score")), decimals=0)
    return int(max(0, min(100, score))) if score is not None else None


def _score_status(score: int | None) -> tuple[str, HexColor]:
    if score is None:
        return "Review ready", MUTED
    if score >= 80:
        return "Strong", GREEN
    if score >= 65:
        return "On Track", HexColor("#4A8C64")
    if score >= 50:
        return "Developing", AMBER
    return "Needs Focus", SOFT_RED


def _short_report_id(result: Dict[str, Any]) -> str:
    raw = sanitize_text(result.get("analysis_id"))
    compact = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return compact[-8:] if compact else ""


def _analysis_date(result: Dict[str, Any]) -> str:
    for value in (result.get("analysis_date"), result.get("created_at"), result.get("completed_at")):
        text = sanitize_text(value)
        if not text:
            continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%B %d, %Y")
        except ValueError:
            return text
    return datetime.now().strftime("%B %d, %Y")


def _wrap_text(text: Any, font_name: str, font_size: float, max_width: float, max_lines: int | None = None) -> list[str]:
    words = sanitize_text(text).split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        final = lines[-1]
        while final and pdfmetrics.stringWidth(f"{final}...", font_name, font_size) > max_width:
            final = final.rsplit(" ", 1)[0] if " " in final else final[:-1]
        lines[-1] = f"{final.rstrip()}..." if final else "..."
    return lines


def _draw_wrapped_text(
    pdf: canvas.Canvas,
    text: Any,
    x: float,
    baseline_y: float,
    max_width: float,
    *,
    font_name: str = "Helvetica",
    font_size: float = 9,
    leading: float = 12,
    color: HexColor = INK,
    max_lines: int | None = None,
) -> float:
    lines = _wrap_text(text, font_name, font_size, max_width, max_lines)
    pdf.setFillColor(color)
    pdf.setFont(font_name, font_size)
    y = baseline_y
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _draw_footer(pdf: canvas.Canvas, page_number: int, report_id: str) -> None:
    pdf.setStrokeColor(DIVIDER)
    pdf.setLineWidth(0.6)
    pdf.line(MARGIN, 50, PAGE_WIDTH - MARGIN, 50)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.8)
    pdf.drawString(MARGIN, 37, "SwingSight AI - Clear feedback for better swings")
    if report_id:
        pdf.drawCentredString(PAGE_WIDTH / 2, 37, f"Report ID: {report_id}")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 37, f"Page {page_number} of 2")
    pdf.setFont("Helvetica", 5.7)
    pdf.drawString(
        MARGIN,
        24,
        "AI-generated coaching feedback is intended for practice guidance and is not a substitute for instruction from a qualified golf professional.",
    )


def _draw_page_header(pdf: canvas.Canvas, page_label: str) -> None:
    logo_y = PAGE_HEIGHT - 67
    pdf.setFillColor(FOREST)
    pdf.roundRect(MARGIN, logo_y, 25, 25, 7, stroke=0, fill=1)
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawCentredString(MARGIN + 12.5, logo_y + 7.4, "S")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(MARGIN + 34, logo_y + 7.2, "SwingSight")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 7.2)
    pdf.drawRightString(PAGE_WIDTH - MARGIN, logo_y + 8, page_label.upper())
    pdf.setStrokeColor(DIVIDER)
    pdf.setLineWidth(0.6)
    pdf.line(MARGIN, PAGE_HEIGHT - 79, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 79)


def _paint_page_background(pdf: canvas.Canvas) -> None:
    pdf.setFillColor(WHITE)
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)


def _draw_summary_card(pdf: canvas.Canvas, result: Dict[str, Any], x: float, y: float, width: float, height: float) -> None:
    score = _score_value(result)
    status, status_color = _score_status(score)
    club = _club_name(result)
    focus = _primary_focus(result)

    pdf.setFillColor(PALE_GREEN)
    pdf.roundRect(x, y, width, height, 15, stroke=0, fill=1)

    center_x = x + 60
    center_y = y + (height / 2) + 3
    pdf.setStrokeColor(HexColor("#CDE4CC"))
    pdf.setLineWidth(5)
    pdf.circle(center_x, center_y, 31, stroke=1, fill=0)
    pdf.setStrokeColor(status_color)
    pdf.setLineWidth(5)
    if score is not None:
        pdf.arc(center_x - 31, center_y - 31, center_x + 31, center_y + 31, 90, -(score / 100) * 360)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(center_x, center_y - 3, str(score) if score is not None else "--")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 6.5)
    pdf.drawCentredString(center_x, center_y - 15, "SWING SCORE")
    pdf.setFillColor(status_color)
    pdf.setFont("Helvetica-Bold", 7.4)
    pdf.drawCentredString(center_x, center_y - 43, status.upper())

    separator_x = x + 128
    pdf.setStrokeColor(HexColor("#D1E4D0"))
    pdf.setLineWidth(0.6)
    pdf.line(separator_x, y + 16, separator_x, y + height - 16)

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawString(separator_x + 20, y + height - 30, "CLUB")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(separator_x + 20, y + height - 48, club)

    focus_x = separator_x + 156
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawString(focus_x, y + height - 30, "PRIMARY FOCUS")
    _draw_wrapped_text(
        pdf,
        focus,
        focus_x,
        y + height - 47,
        width - (focus_x - x) - 18,
        font_name="Helvetica-Bold",
        font_size=9.2,
        leading=11,
        color=FOREST,
        max_lines=3,
    )


def _details(result: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("gemini_analysis", "detailed_analysis", "coaching_details"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = sanitize_text(value)
        if text:
            return text
    return ""


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [sanitize_text(value)] if sanitize_text(value) else []
    if not isinstance(value, list):
        return []
    return [text for text in (sanitize_text(item) for item in value) if text]


def _primary_focus(result: Dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    details = _details(result)
    return _first_text(
        result.get("next_focus"),
        summary.get("next_focus"),
        details.get("next_focus"),
        "Build a repeatable, athletic motion.",
    )


def _coach_summary(result: Dict[str, Any]) -> str:
    details = _details(result)
    return _first_text(
        details.get("summary"),
        details.get("coach_summary"),
        result.get("coach_summary"),
        result.get("score_rationale"),
        "This swing gives us a clear starting point. Use one focused rehearsal at a time to build a more repeatable motion.",
    )


def _is_genuine_strength(text: str) -> bool:
    lowered = text.lower()
    generic_fragments = (
        "uploaded successfully",
        "process the video",
        "solid base here",
        "review will highlight",
        "clear starting point",
    )
    return bool(text) and not any(fragment in lowered for fragment in generic_fragments)


def _strengths(result: Dict[str, Any]) -> list[str]:
    details = _details(result)
    candidates = _as_text_list(details.get("strengths")) + _as_text_list(result.get("strengths"))
    unique: list[str] = []
    for item in candidates:
        if _is_genuine_strength(item) and item not in unique:
            unique.append(item)
        if len(unique) == 3:
            break
    return unique or ["This swing gives us a clear starting point for developing more consistent movement."]


def _strength_title(text: str) -> str:
    lowered = text.lower()
    if "head" in lowered:
        return "Steady Head Movement"
    if "hip" in lowered:
        return "Productive Hip Turn"
    if "shoulder" in lowered:
        return "Strong Shoulder Turn"
    if "tempo" in lowered or "rhythm" in lowered:
        return "Smooth Rhythm"
    if "balance" in lowered or "finish" in lowered:
        return "Balanced Motion"
    if "path" in lowered:
        return "Useful Club Path"
    return "Building Your Foundation"


def _focus_guidance(focus: str) -> tuple[str, str]:
    lowered = focus.lower()
    if "knee" in lowered:
        return (
            "A slightly more athletic knee position can improve balance, posture, and your ability to rotate throughout the swing.",
            "Feel balanced through the middle of your feet with your knees relaxed rather than locked.",
        )
    if "head" in lowered:
        return (
            "A quieter head can make it easier to return the club to the ball with consistent contact and balance.",
            "Turn around a steady center rather than following the ball with your head.",
        )
    if "transition" in lowered or "tempo" in lowered or "rhythm" in lowered:
        return (
            "A smoother change of direction gives your body time to sequence from the ground up and improves balance through impact.",
            "Complete the backswing before starting down.",
        )
    if "hip" in lowered or "rotate" in lowered or "belt buckle" in lowered:
        return (
            "More rotation through impact can help your body continue through the shot and give the arms more room to swing.",
            "Let the belt buckle begin turning toward the target.",
        )
    if "posture" in lowered or "spine" in lowered:
        return (
            "Maintaining posture helps you preserve a repeatable strike zone and gives your body a stable platform to rotate around.",
            "Stay athletic and keep your chest over the ball through the first move down.",
        )
    return (
        "One clear movement focus makes practice more useful and helps the change carry from rehearsal into your normal swing.",
        "Make a few slow rehearsals, then keep the same feel when you hit the next ball.",
    )


def _improvements(result: Dict[str, Any]) -> list[str]:
    details = _details(result)
    candidates = _as_text_list(details.get("improvements")) + _as_text_list(result.get("improvements"))
    unique: list[str] = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
        if len(unique) == 3:
            break
    return unique


def _priority_from_recommendation(recommendation: str) -> tuple[str, str, str, str]:
    lowered = recommendation.lower()
    if "tempo" in lowered or "rhythm" in lowered or "backswing" in lowered or "transition" in lowered:
        return (
            "Smooth the Transition",
            "The backswing and downswing rhythm may be uneven.",
            "A smoother change of direction can help the body sequence more naturally into impact.",
            "Complete the backswing before starting down.",
        )
    if "head" in lowered:
        return (
            "Keep the Head Quieter",
            "Head movement may make consistent contact harder to repeat.",
            "A steadier center supports balance and a more reliable strike.",
            "Turn around a steady center.",
        )
    if "hip" in lowered or "rotate" in lowered or "belt buckle" in lowered:
        return (
            "Rotate Through Impact",
            "More hip rotation may help the body continue through the shot.",
            "Rotation gives the arms room to swing and supports a balanced finish.",
            "Let the belt buckle begin turning toward the target.",
        )
    if "knee" in lowered or "setup" in lowered:
        return (
            "Build an Athletic Setup",
            "Your setup can use a little more relaxed knee flex and balance.",
            "An athletic address makes it easier to rotate and stay in posture.",
            "Feel balanced through the middle of your feet.",
        )
    if "posture" in lowered or "spine" in lowered:
        return (
            "Maintain Your Posture",
            "Your posture may change as the swing moves toward impact.",
            "A steadier posture helps preserve the strike zone and swing plane.",
            "Stay athletic and keep your chest over the ball.",
        )
    return (
        "Your Next Adjustment",
        recommendation,
        "A single clear adjustment helps turn feedback into more repeatable practice.",
        "Make a few slow rehearsals before your next ball.",
    )


def _priorities(result: Dict[str, Any]) -> list[tuple[str, str, str, str]]:
    priorities = [_priority_from_recommendation(item) for item in _improvements(result)]
    if priorities:
        return priorities[:3]
    return [
        (
            "Build Your Foundation",
            "This swing gives us a clear starting point for developing more consistent movement.",
            "A stable, repeatable motion is the foundation for every improvement that follows.",
            "Start with a smooth rehearsal and finish in balance.",
        )
    ]


def _draw_coach_take(pdf: canvas.Canvas, result: Dict[str, Any], baseline_y: float) -> None:
    pdf.setFillColor(FOREST)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN, baseline_y, "Coach's Take")
    _draw_wrapped_text(
        pdf,
        _coach_summary(result),
        MARGIN,
        baseline_y - 17,
        CONTENT_WIDTH,
        font_size=9.4,
        leading=12,
        color=MUTED,
        max_lines=2,
    )


def _draw_focus_card(pdf: canvas.Canvas, result: Dict[str, Any], x: float, y: float, width: float, height: float) -> None:
    focus = _primary_focus(result)
    explanation, cue = _focus_guidance(focus)
    pdf.setFillColor(FOREST)
    pdf.roundRect(x, y, width, height, 14, stroke=0, fill=1)
    pdf.setFillColor(BRIGHT_GREEN)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawString(x + 18, y + height - 18, "YOUR NEXT FOCUS")
    pdf.setFillColor(HexColor("#FFFFFF"))
    _draw_wrapped_text(
        pdf,
        focus,
        x + 18,
        y + height - 36,
        width - 36,
        font_name="Helvetica-Bold",
        font_size=12,
        leading=14,
        color=HexColor("#FFFFFF"),
        max_lines=2,
    )
    _draw_wrapped_text(
        pdf,
        explanation,
        x + 18,
        y + 33,
        width - 36,
        font_size=7.6,
        leading=9.2,
        color=HexColor("#D2E6D3"),
        max_lines=2,
    )
    pdf.setFillColor(BRIGHT_GREEN)
    pdf.setFont("Helvetica-Bold", 6.5)
    pdf.drawString(x + 18, y + 16, "PRACTICE CUE")
    _draw_wrapped_text(
        pdf,
        cue,
        x + 76,
        y + 16,
        width - 94,
        font_name="Helvetica-Oblique",
        font_size=6.8,
        leading=8,
        color=HexColor("#FFFFFF"),
        max_lines=1,
    )


def _draw_strength_cards(pdf: canvas.Canvas, strengths: Sequence[str], top_y: float) -> None:
    cards = strengths[:3]
    count = len(cards)
    gap = 10
    card_width = (CONTENT_WIDTH - (gap * (count - 1))) / count
    card_height = 55
    for index, strength in enumerate(cards):
        x = MARGIN + index * (card_width + gap)
        y = top_y - card_height
        pdf.setFillColor(PALE_CARD)
        pdf.setStrokeColor(DIVIDER)
        pdf.setLineWidth(0.5)
        pdf.roundRect(x, y, card_width, card_height, 10, stroke=1, fill=1)
        pdf.setFillColor(GREEN)
        pdf.setFont("Helvetica-Bold", 7.4)
        pdf.drawString(x + 10, y + card_height - 14, _strength_title(strength))
        _draw_wrapped_text(
            pdf,
            strength,
            x + 10,
            y + card_height - 27,
            card_width - 20,
            font_size=6.8,
            leading=8,
            color=MUTED,
            max_lines=3,
        )


def _draw_priority_cards(pdf: canvas.Canvas, priorities: Sequence[tuple[str, str, str, str]], top_y: float) -> None:
    cards = priorities[:3]
    count = len(cards)
    gap = 9
    card_width = (CONTENT_WIDTH - (gap * (count - 1))) / count
    card_height = 126
    for index, (title, observed, why, thought) in enumerate(cards):
        x = MARGIN + index * (card_width + gap)
        y = top_y - card_height
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.setStrokeColor(DIVIDER)
        pdf.setLineWidth(0.6)
        pdf.roundRect(x, y, card_width, card_height, 10, stroke=1, fill=1)
        pdf.setFillColor(GREEN)
        pdf.setFont("Helvetica-Bold", 7.7)
        pdf.drawString(x + 10, y + card_height - 15, f"{index + 1}. {title}")
        sections = (
            ("OBSERVED", observed, y + card_height - 29),
            ("WHY IT MATTERS", why, y + card_height - 65),
            ("SWING THOUGHT", thought, y + card_height - 101),
        )
        for label, text, label_y in sections:
            pdf.setFillColor(MUTED)
            pdf.setFont("Helvetica-Bold", 5.6)
            pdf.drawString(x + 10, label_y, label)
            _draw_wrapped_text(
                pdf,
                text,
                x + 10,
                label_y - 9,
                card_width - 20,
                font_size=6.2,
                leading=7.1,
                color=INK,
                max_lines=2,
            )


def _draw_page_one(pdf: canvas.Canvas, result: Dict[str, Any], report_id: str) -> None:
    _draw_page_header(pdf, "Swing Analysis Report")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(MARGIN, 668, "Swing Analysis Report")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(MARGIN, 647, f"Analysis date: {_analysis_date(result)} | Club: {_club_name(result)}")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 647, "AI-Powered Swing Coaching")

    _draw_summary_card(pdf, result, MARGIN, 520, CONTENT_WIDTH, 105)
    _draw_coach_take(pdf, result, 495)

    _draw_focus_card(pdf, result, MARGIN, 335, CONTENT_WIDTH, 100)

    pdf.setFillColor(FOREST)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN, 313, "What You're Doing Well")
    _draw_strength_cards(pdf, _strengths(result), 300)

    pdf.setFillColor(FOREST)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN, 223, "Improvement Priorities")
    _draw_priority_cards(pdf, _priorities(result), 210)
    _draw_footer(pdf, 1, report_id)


def _score_rows(metrics: Dict[str, Any]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for key, label, description in SCORE_METRICS:
        value = _usable_metric(key, metrics.get(key))
        if value is None:
            continue
        rows.append((label, int(max(0, min(100, round(value)))), description))
    return rows


def _measurement_rows(metrics: Dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label, unit in MEASUREMENT_METRICS:
        value = _usable_metric(key, metrics.get(key))
        if value is None:
            continue
        if unit == "degrees":
            formatted = format_degrees(value)
        elif unit == "pixels":
            formatted = format_pixels(value)
        else:
            formatted = format_ratio(value)
        if formatted != "Not available":
            rows.append((label, formatted))
    return rows[:8]


def _draw_score_bar(pdf: canvas.Canvas, x: float, y: float, width: float, label: str, score: int, description: str) -> None:
    pdf.setFillColor(PALE_CARD)
    pdf.setStrokeColor(DIVIDER)
    pdf.setLineWidth(0.5)
    pdf.roundRect(x, y, width, 35, 9, stroke=1, fill=1)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 8.4)
    pdf.drawString(x + 12, y + 22, f"{label} - {score} / 100")
    _draw_wrapped_text(
        pdf,
        description,
        x + 12,
        y + 11,
        width - 146,
        font_size=6.8,
        leading=8,
        color=MUTED,
        max_lines=1,
    )
    bar_x = x + width - 116
    bar_y = y + 11
    bar_width = 92
    pdf.setFillColor(HexColor("#DCEBDC"))
    pdf.roundRect(bar_x, bar_y, bar_width, 6, 3, stroke=0, fill=1)
    pdf.setFillColor(GREEN)
    pdf.roundRect(bar_x, bar_y, max(3, bar_width * score / 100), 6, 3, stroke=0, fill=1)


def _draw_measurement_table(pdf: canvas.Canvas, rows: Sequence[tuple[str, str]], top_y: float) -> None:
    if not rows:
        pdf.setFillColor(PALE_CARD)
        pdf.roundRect(MARGIN, top_y - 43, CONTENT_WIDTH, 43, 10, stroke=0, fill=1)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 8.5)
        pdf.drawString(MARGIN + 14, top_y - 24, "No reliable movement measurements were available from this video.")
        return

    row_height = 27
    table_height = len(rows) * row_height
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.setStrokeColor(DIVIDER)
    pdf.setLineWidth(0.6)
    pdf.roundRect(MARGIN, top_y - table_height, CONTENT_WIDTH, table_height, 10, stroke=1, fill=1)
    for index, (label, value) in enumerate(rows):
        y = top_y - (index * row_height)
        if index:
            pdf.setStrokeColor(DIVIDER)
            pdf.line(MARGIN + 12, y, PAGE_WIDTH - MARGIN - 12, y)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(MARGIN + 14, y - 17, label)
        pdf.setFillColor(FOREST)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawRightString(PAGE_WIDTH - MARGIN - 14, y - 17, value)


def _draw_page_two(pdf: canvas.Canvas, result: Dict[str, Any], report_id: str) -> None:
    metrics = result.get("advanced_metrics") or result.get("advanced") or {}
    metrics = metrics if isinstance(metrics, dict) else {}
    _draw_page_header(pdf, "Swing Details")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(MARGIN, 668, "Swing Details")
    _draw_wrapped_text(
        pdf,
        "These measurements are estimated from the uploaded video and are intended to support coaching feedback. Results may vary based on camera angle, lighting, framing, and video quality.",
        MARGIN,
        646,
        CONTENT_WIDTH,
        font_size=8.5,
        leading=11,
        color=MUTED,
        max_lines=2,
    )

    pdf.setFillColor(FOREST)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN, 600, "Key Movement Scores")
    score_rows = _score_rows(metrics)
    if score_rows:
        y = 586
        for label, score, description in score_rows[:4]:
            _draw_score_bar(pdf, MARGIN, y - 35, CONTENT_WIDTH, label, score, description)
            y -= 44
    else:
        pdf.setFillColor(PALE_CARD)
        pdf.roundRect(MARGIN, 543, CONTENT_WIDTH, 43, 10, stroke=0, fill=1)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 8.5)
        pdf.drawString(MARGIN + 14, 561, "Key movement scores were not reliably measured from this video.")

    pdf.setFillColor(FOREST)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN, 390, "Measured Movement")
    _draw_measurement_table(pdf, _measurement_rows(metrics), 374)
    _draw_footer(pdf, 2, report_id)


def generate_pdf_report(result: Dict[str, Any], reports_dir: str) -> str:
    """Generate the single, branded SwingSight coaching report as a PDF."""
    output_dir = _ensure_reports_dir(reports_dir)
    report_path = output_dir / build_report_filename(_club_name(result))
    pdf = canvas.Canvas(str(report_path), pagesize=letter, pageCompression=1)
    pdf.setTitle("SwingSight Swing Analysis Report")
    pdf.setAuthor("SwingSight AI")
    pdf.setSubject("AI-powered golf swing coaching report")
    report_id = _short_report_id(result)

    _paint_page_background(pdf)
    _draw_page_one(pdf, result, report_id)
    pdf.showPage()
    _paint_page_background(pdf)
    _draw_page_two(pdf, result, report_id)
    pdf.save()
    return str(report_path)
