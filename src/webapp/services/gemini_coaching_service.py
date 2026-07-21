from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable

import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-3-flash-preview"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

COACHING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "A concise, encouraging summary grounded only in the provided evidence."},
        "next_focus": {"type": "string", "description": "The single highest-value practice focus."},
        "strengths": {"type": "array", "items": {"type": "string"}, "description": "Up to three evidence-based strengths."},
        "improvements": {"type": "array", "items": {"type": "string"}, "description": "Up to three evidence-based adjustments."},
        "tips": {"type": "array", "items": {"type": "string"}, "description": "Short, actionable on-range cues."},
        "drills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "description"],
            },
            "description": "Up to three simple drills tied to the evidence.",
        },
        "data_gaps": {"type": "array", "items": {"type": "string"}, "description": "Measurements that were not available and therefore were not assessed."},
    },
    "required": ["summary", "next_focus", "strengths", "improvements", "tips", "drills", "data_gaps"],
}


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _compact_mapping(source: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            output[key] = value
    return output


def build_gemini_evidence(analysis: Dict[str, Any], conventional_coaching: Dict[str, Any]) -> Dict[str, Any]:
    """Create a privacy-conscious, evidence-only payload for Gemini.

    The raw swing video, file paths, frames, and debug artifacts are intentionally
    excluded. Gemini receives only metrics already calculated by the local vision
    pipeline. Ball, speed, and impact fields remain optional until those CV models
    produce measured values.
    """
    metrics = analysis.get("advanced_metrics", {}) or {}
    tracking = analysis.get("tracking", {}) or {}
    quality = tracking.get("quality_metrics", {}) or {}
    club_detection = analysis.get("club_detection", {}) or {}

    body_keys = (
        "hip_rotation", "shoulder_rotation", "spine_angle", "spine_angle_deg",
        "head_stability", "head_movement_cm", "weight_shift", "balance_score",
        "tempo_estimate", "knee_flex_deg", "club_path",
    )
    speed_keys = (
        "club_speed_mph", "clubhead_speed_mph", "club_speed", "clubhead_speed",
        "hand_speed_mph", "swing_speed_mph",
    )
    impact_keys = (
        "impact_frame", "impact_timestamp", "impact_timestamp_sec", "impact_quality",
        "ball_detected", "ball_speed_mph", "ball_speed", "launch_angle_deg",
        "launch_angle", "smash_factor", "contact_point", "face_to_path_deg",
    )
    body_movement = _compact_mapping(metrics, body_keys)
    speed = _compact_mapping(metrics, speed_keys)
    impact = _compact_mapping(metrics, impact_keys)

    # Support future CV modules that publish grouped measurements without
    # requiring another Gemini integration change.
    for group_name, target in (("speed", speed), ("impact", impact), ("ball", impact)):
        group = metrics.get(group_name)
        if isinstance(group, dict):
            target.update(_compact_mapping(group, group.keys()))

    unavailable: list[str] = []
    if not speed:
        unavailable.append("No measured club or hand speed was provided by the vision pipeline.")
    if not impact:
        unavailable.append("No measured ball position, contact, launch, or impact data was provided by the vision pipeline.")
    if not body_movement:
        unavailable.append("No usable body-movement measurements were provided by the vision pipeline.")

    return {
        "measurement_source": "local computer-vision pipeline",
        "club": {
            "label": analysis.get("club") or club_detection.get("club") or "Not detected",
            "status": club_detection.get("status"),
            "confidence": _number(club_detection.get("confidence")),
        },
        "body_movement": body_movement,
        "speed": speed,
        "impact_and_ball": impact,
        "tracking_quality": _compact_mapping(
            quality,
            (
                "total_frames_processed", "frames_with_pose", "detection_rate",
                "average_pose_quality_score", "head_tracked_rate", "hands_tracked_rate",
                "feet_tracked_rate", "frames_interpolated", "frames_rejected",
            ),
        ),
        "existing_local_coaching": {
            "strengths": conventional_coaching.get("strengths", []),
            "improvements": conventional_coaching.get("improvements", []),
            "next_focus": conventional_coaching.get("next_focus"),
        },
        "unavailable_measurements": unavailable,
    }


def _coach_prompt(evidence: Dict[str, Any]) -> str:
    return """You are SwingSight's golf coaching layer. Create useful, concise practice
feedback from the JSON evidence below.

Rules:
- Treat only provided values as facts. Never estimate, fabricate, or convert an
  unavailable value into club speed, ball speed, impact, launch, or contact data.
- If ball/impact/speed data is unavailable, say it was not measured rather than
  diagnosing impact quality.
- Explain body-movement findings in golfer-friendly terms and keep the tone
  encouraging.
- Do not provide medical, injury, or equipment-fit advice.
- Give a maximum of three strengths, three improvements, three tips, and three drills.
- Each drill must be safe, simple, and usable on a practice range.
- Return only JSON that matches the requested schema.

Evidence:
""" + json.dumps(evidence, separators=(",", ":"), ensure_ascii=True)


def _as_string_list(value: Any, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()][:limit]


def validate_coaching(payload: Any) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    next_focus = payload.get("next_focus")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(next_focus, str) or not next_focus.strip():
        return None

    drills: list[Dict[str, str]] = []
    for drill in payload.get("drills", []) if isinstance(payload.get("drills"), list) else []:
        if not isinstance(drill, dict):
            continue
        title = drill.get("title")
        description = drill.get("description")
        if isinstance(title, str) and title.strip() and isinstance(description, str) and description.strip():
            drills.append({"title": title.strip(), "description": description.strip()})
        if len(drills) == 3:
            break

    return {
        "summary": summary.strip(),
        "next_focus": next_focus.strip(),
        "strengths": _as_string_list(payload.get("strengths")),
        "improvements": _as_string_list(payload.get("improvements")),
        "tips": _as_string_list(payload.get("tips")),
        "drills": drills,
        "data_gaps": _as_string_list(payload.get("data_gaps"), limit=6),
    }


class GeminiCoachingService:
    """Optional structured Gemini coaching over local computer-vision evidence."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        settings = config or {}
        self.enabled = bool(settings.get("enabled", True))
        self.model = str(settings.get("model") or DEFAULT_MODEL)
        self.timeout_seconds = max(3, int(settings.get("timeout_seconds", 35)))
        self.api_key_env = str(settings.get("api_key_env") or "GEMINI_API_KEY")
        self.api_key = os.getenv(self.api_key_env, "").strip()

    def coach(self, analysis: Dict[str, Any], conventional_coaching: Dict[str, Any]) -> Dict[str, Any]:
        evidence = build_gemini_evidence(analysis, conventional_coaching)
        base = {
            "enabled": self.enabled,
            "model": self.model,
            "input_mode": "structured_cv_evidence_only",
            "video_shared_with_gemini": False,
            "evidence_fields": {
                "body_movement": sorted(evidence["body_movement"].keys()),
                "speed": sorted(evidence["speed"].keys()),
                "impact_and_ball": sorted(evidence["impact_and_ball"].keys()),
            },
        }
        if not self.enabled:
            return {**base, "status": "disabled", "coaching": None}
        if not self.api_key:
            return {**base, "status": "not_configured", "coaching": None}

        request_body = {
            "contents": [{"parts": [{"text": _coach_prompt(evidence)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": COACHING_SCHEMA,
                "temperature": 0.2,
                "maxOutputTokens": 1200,
            },
        }
        try:
            response = requests.post(
                GEMINI_API_URL.format(model=self.model),
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                json=request_body,
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                LOGGER.warning("Gemini coaching request failed with HTTP %s", response.status_code)
                return {**base, "status": "unavailable", "coaching": None}
            payload = response.json()
            text = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            coaching = validate_coaching(json.loads(text))
            if coaching is None:
                LOGGER.warning("Gemini coaching response did not pass validation")
                return {**base, "status": "invalid_response", "coaching": None}
            return {**base, "status": "success", "coaching": coaching}
        except (requests.RequestException, ValueError, TypeError, KeyError, IndexError) as exc:
            LOGGER.warning("Gemini coaching unavailable: %s", type(exc).__name__)
            return {**base, "status": "unavailable", "coaching": None}
