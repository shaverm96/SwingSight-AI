from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import requests

from swingsight.config import load_dotenv


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-3-flash-preview"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

COACHING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Two or three concise, encouraging sentences in a natural on-range golf-coach voice. Address the player directly and use only provided evidence."},
        "overall_score": {"type": ["integer", "null"], "description": "A 0-100 score only when enough measured pose evidence exists; otherwise null."},
        "score_rationale": {"type": "string", "description": "A short player-friendly explanation of the score using only provided measurements, or why no score was produced. Avoid computer-vision jargon."},
        "next_focus": {"type": "string", "description": "The single highest-value player-facing practice focus, phrased as a simple move or feel."},
        "strengths": {"type": "array", "items": {"type": "string"}, "description": "Up to three observed swing actions stated in natural golf-coach language. Never describe tracking quality or processing."},
        "improvements": {"type": "array", "items": {"type": "string"}, "description": "Up to three specific player-facing adjustments with a clear movement or feel."},
        "tips": {"type": "array", "items": {"type": "string"}, "description": "Short, natural on-range cues a golfer can use on the next ball."},
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
    "required": ["summary", "overall_score", "score_rationale", "next_focus", "strengths", "improvements", "tips", "drills", "data_gaps"],
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

    tracking_quality = _compact_mapping(
        quality,
        (
            "total_frames_processed", "frames_with_pose", "detection_rate",
            "average_pose_quality_score", "head_tracked_rate", "hands_tracked_rate",
            "feet_tracked_rate", "frames_interpolated", "frames_rejected",
        ),
    )
    pose_frames = tracking.get("pose_frames_detected") or tracking.get("frames_with_pose")
    if tracking_quality.get("frames_with_pose") is None and pose_frames is not None:
        tracking_quality["frames_with_pose"] = pose_frames

    measured_body_metrics = {
        key: value for key, value in body_movement.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and _number(value) is not None
    }
    score_eligible = bool(
        (_number(tracking_quality.get("frames_with_pose")) or 0) >= 24
        and len(measured_body_metrics) >= 3
    )

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
        "tracking_quality": tracking_quality,
        "scoring_eligibility": {
            "eligible": score_eligible,
            "pose_frames": int(_number(tracking_quality.get("frames_with_pose")) or 0),
            "measured_body_metrics": sorted(measured_body_metrics.keys()),
            "rule": "Score when at least 24 pose frames and three numeric body-movement measurements are available.",
        },
        "unavailable_measurements": unavailable,
    }


def _coach_prompt(evidence: Dict[str, Any]) -> str:
    return """You are SwingSight's golf coaching layer. Create useful, concise practice
feedback from the JSON evidence below.

Voice and coaching rules:
- Speak like an experienced, encouraging golf coach standing right beside one
  player on the range. Use warm, direct language: "you", "feel", "setup", "turn",
  "finish", "alright", and other natural golf terms when supported by the evidence.
- Let the voice have some personality: conversational, confident, and practical,
  like a coach giving a quick between-balls cue. Use light, friendly golf humor
  sparingly—at most one gentle joke or playful comparison in the entire response.
  Keep it kind, never sarcastic, and never make the player the punchline.
- Lead with what the player can do on the next swing. Every strength, adjustment,
  tip, and drill must describe an observed swing action or a simple action to try.
- Do not mention vision pipelines, pose tracking, processed frames, tracking
  quality, measurement accuracy, reliability, or model outputs in player-facing
  coaching. Do not praise the player merely because data was captured successfully.
- Avoid robotic or clinical wording such as "generate measurable tempo",
  "stabilize the head to allow tracking", or "consistent tracking reliability".
  Say the golf move and the feel instead.
- Use camera/framing guidance only in data_gaps when missing evidence genuinely
  prevents a movement assessment; never make it the main coaching focus when
  coachable body-movement evidence exists.
- Keep each list item short, specific, and conversational. The next_focus should
  read like one clear move or feel, not a category or report heading.

Evidence and safety rules:
- Treat only provided values as facts. Never estimate, fabricate, or convert an
  unavailable value into club speed, ball speed, impact, launch, or contact data.
- If ball/impact/speed data is unavailable, say it was not measured rather than
  diagnosing impact quality.
- Do not provide medical, injury, or equipment-fit advice.
- Give a maximum of three strengths, three improvements, three tips, and three drills.
- Follow scoring_eligibility exactly. If eligible is true, overall_score MUST be a
  whole number from 0 to 100; never return null merely because some measurements
  are missing or zero. Use only the listed body-movement measurements to score.
  If eligible is false, overall_score must be null.
- Explain the score in score_rationale and name any limitations in data_gaps.
  Never use unavailable speed, ball, or impact data to score.
- Each drill must be safe, simple, and usable on a practice range.
- Return only JSON that matches the requested schema.

Evidence:
""" + json.dumps(evidence, separators=(",", ":"), ensure_ascii=True)


def _as_string_list(value: Any, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()][:limit]


def _response_text(payload: Any) -> str:
    """Collect text from Gemini candidates without exposing the raw response."""
    if not isinstance(payload, dict):
        return ""

    text_parts: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    return "\n".join(text_parts).strip()


def _json_object_from_text(text: Any) -> Dict[str, Any] | None:
    """Parse a JSON object from a JSON response that may be fenced by the model."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for value in (candidate, candidate[candidate.find("{"):] if "{" in candidate else ""):
        if not value:
            continue
        try:
            parsed, _ = decoder.raw_decode(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def validate_coaching(payload: Any) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    next_focus = payload.get("next_focus")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(next_focus, str) or not next_focus.strip():
        return None

    score_value = payload.get("overall_score")
    if isinstance(score_value, bool):
        score_value = None
    elif isinstance(score_value, (int, float)):
        score_value = int(round(max(0, min(100, score_value))))
    else:
        score_value = None
    score_rationale = payload.get("score_rationale")
    if not isinstance(score_rationale, str) or not score_rationale.strip():
        score_rationale = "No Gemini score rationale was returned."

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
        "overall_score": score_value,
        "score_rationale": score_rationale.strip(),
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
        # Gemini 3 may use part of the output budget for reasoning before returning
        # its structured answer. Leave enough room for the complete JSON object.
        self.max_output_tokens = max(1024, int(settings.get("max_output_tokens", 4096)))
        self.api_key_env = str(settings.get("api_key_env") or "GEMINI_API_KEY")
        self.api_key, self.api_key_source = self._read_api_key()

    def _read_api_key(self) -> tuple[str, str | None]:
        # Re-read the project .env so keys added after the local server starts
        # are available without a full Windows sign-out or application restart.
        load_dotenv(Path.cwd() / ".env")

        candidates = [self.api_key_env, "GEMINI_API_KEY", "GOOGLE_API_KEY"]

        for environment_name in dict.fromkeys(candidates):
            value = os.getenv(environment_name, "").strip()
            if value:
                return value, environment_name
        return "", None

    def coach(self, analysis: Dict[str, Any], conventional_coaching: Dict[str, Any]) -> Dict[str, Any]:
        self.api_key, self.api_key_source = self._read_api_key()
        evidence = build_gemini_evidence(analysis, conventional_coaching)
        base = {
            "enabled": self.enabled,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_detected": bool(self.api_key),
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
                "maxOutputTokens": self.max_output_tokens,
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
            response_data = _json_object_from_text(_response_text(payload))
            if response_data is None:
                LOGGER.warning("Gemini coaching returned non-JSON text")
                return {**base, "status": "invalid_response", "reason": "response_not_json", "coaching": None}

            coaching = validate_coaching(response_data)
            if coaching is None:
                LOGGER.warning("Gemini coaching response did not pass validation")
                return {**base, "status": "invalid_response", "reason": "response_schema_invalid", "coaching": None}
            return {**base, "status": "success", "coaching": coaching}
        except (requests.RequestException, ValueError, TypeError, KeyError, IndexError) as exc:
            LOGGER.warning("Gemini coaching unavailable: %s", type(exc).__name__)
            return {**base, "status": "unavailable", "reason": "request_failed", "coaching": None}
