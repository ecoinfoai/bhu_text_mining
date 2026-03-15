"""LLM Vision OCR module — data classes and utility functions.

Provides structured data types for LLM-based text recognition and
utility functions for token-to-word confidence mapping and prompt
construction.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WordConfidence:
    """Per-word recognition confidence derived from token logprobs.

    Attributes:
        word: The word text.
        confidence: Recognition confidence (0.0-1.0), geometric mean of token probs.
        token_count: Number of tokens composing this word.
    """

    word: str
    confidence: float
    token_count: int


@dataclass
class TokenUsage:
    """API call token usage.

    Attributes:
        input_tokens: Input token count (prompt + image).
        output_tokens: Output token count (recognized text).
    """

    input_tokens: int
    output_tokens: int


@dataclass
class LLMVisionResponse:
    """Full LLM Vision API response for a single image.

    Attributes:
        text: Recognized text.
        word_confidences: Per-word confidence list, None if logprobs unavailable.
        confidence_mean: Arithmetic mean of word confidences, None if unavailable.
        confidence_min: Minimum word confidence, None if unavailable.
        usage: Token usage for this call.
        finish_reason: Model stop reason (e.g. "STOP", "MAX_TOKENS").
        logprobs_raw: Raw token logprobs for reanalysis, None if unavailable.
        safety_ratings: Gemini safety ratings, None for Anthropic.
    """

    text: str
    word_confidences: list[WordConfidence] | None
    confidence_mean: float | None
    confidence_min: float | None
    usage: TokenUsage
    finish_reason: str
    logprobs_raw: list[dict] | None
    safety_ratings: list[dict] | None


def compute_word_confidence(
    logprobs_result: Any,
    text: str,
) -> list[WordConfidence]:
    """Convert token-level logprobs to per-word confidence scores.

    Uses sequential token text concatenation with whitespace boundary
    detection to map tokens to words. Each word's confidence is the
    geometric mean of its constituent token probabilities.

    Args:
        logprobs_result: Gemini logprobs result object with
            ``chosen_candidates`` attribute. None if logprobs unavailable.
        text: The full recognized text to split into words.

    Returns:
        List of WordConfidence, one per whitespace-delimited word.
        Empty list if logprobs_result is None or text is empty.
    """
    if logprobs_result is None or not text or not text.strip():
        return []

    chosen = getattr(logprobs_result, "chosen_candidates", None)
    if not chosen:
        return []

    words = text.split()
    if not words:
        return []

    # Collect (token_text, probability) pairs, skipping whitespace-only tokens
    token_probs: list[tuple[str, float]] = []
    for candidate in chosen:
        token_text = candidate.token
        log_prob = candidate.log_probability
        prob = min(max(math.exp(log_prob), 0.0), 1.0)
        # Skip pure whitespace tokens
        if token_text.strip():
            token_probs.append((token_text, prob))

    if not token_probs:
        return []

    # Map tokens to words by sequential concatenation
    results: list[WordConfidence] = []
    token_idx = 0
    for word in words:
        word_token_probs: list[float] = []
        accumulated = ""
        while token_idx < len(token_probs) and len(accumulated) < len(word):
            tok_text, tok_prob = token_probs[token_idx]
            accumulated += tok_text
            word_token_probs.append(tok_prob)
            token_idx += 1

        if not word_token_probs:
            # Fallback: no tokens matched this word
            continue

        # Geometric mean: (p1 * p2 * ... * pn)^(1/n)
        n = len(word_token_probs)
        log_sum = sum(math.log(max(p, 1e-300)) for p in word_token_probs)
        geo_mean = min(max(math.exp(log_sum / n), 0.0), 1.0)

        results.append(
            WordConfidence(
                word=word,
                confidence=geo_mean,
                token_count=n,
            )
        )

    return results


def build_recognition_prompt(
    context: dict[str, str] | None = None,
) -> str:
    """Build a structured Korean prompt for handwriting text recognition.

    Args:
        context: Optional exam context with keys ``subject``, ``question``,
            ``answer_keywords``. Missing keys are omitted from the prompt.

    Returns:
        Formatted prompt string for the LLM vision API.
    """
    lines = [
        "아래 이미지에서 학생이 손으로 작성한 텍스트를 정확히 읽어 그대로 옮겨 적으시오.",
        "- 이미지에 보이는 텍스트만 출력하시오. 추가 설명이나 해석을 붙이지 마시오.",
        "- 글씨가 불분명한 부분은 가장 가능성 높은 해석으로 작성하시오.",
        "- 텍스트가 없으면 빈 문자열을 반환하시오.",
    ]

    if context and any(context.get(k) for k in ("subject", "question", "answer_keywords")):
        lines.append("")
        lines.append("[시험 문맥 정보]")
        if context.get("subject"):
            lines.append(f"- 과목: {context['subject']}")
        if context.get("question"):
            lines.append(f"- 문항: {context['question']}")
        if context.get("answer_keywords"):
            lines.append(f"- 핵심 키워드: {context['answer_keywords']}")
        lines.append("")
        lines.append("위 문맥 정보를 참고하여 전문 용어를 정확히 인식하시오.")

    return "\n".join(lines)


def validate_llm_recognition(
    text: str,
    finish_reason: str,
    confidence_mean: float | None,
) -> dict[str, Any]:
    """Validate an LLM recognition result.

    Checks for empty text, non-STOP finish reason, hallucination (>200 chars),
    and low confidence (<0.3).

    Args:
        text: Recognized text.
        finish_reason: Model stop reason.
        confidence_mean: Average word confidence, None if unavailable.

    Returns:
        Dict with ``valid`` (bool) and ``warnings`` (list[str]).
    """
    warnings: list[str] = []
    valid = True

    if not text or not text.strip():
        warnings.append("빈 텍스트 — 인식 실패")
        valid = False

    # Gemini returns "FinishReason.STOP", Anthropic returns "end_turn"
    stop_reasons = {"STOP", "FinishReason.STOP", "end_turn"}
    if finish_reason not in stop_reasons:
        warnings.append(f"finish_reason={finish_reason} — 응답이 완료되지 않음")
        valid = False

    if text and len(text) > 500:
        warnings.append(f"텍스트 길이 {len(text)}자 > 500자 — 환각 가능성")

    if confidence_mean is not None and confidence_mean < 0.3:
        warnings.append(f"평균 confidence {confidence_mean:.2f} < 0.3 — 수동 검토 필요")

    return {"valid": valid, "warnings": warnings}


def _build_vision_response(full_resp: Any) -> LLMVisionResponse:
    """Convert an LLMFullResponse to LLMVisionResponse with computed confidences."""
    word_confs = compute_word_confidence(
        full_resp.logprobs_result, full_resp.text,
    )
    conf_mean: float | None = None
    conf_min: float | None = None
    if word_confs:
        conf_vals = [wc.confidence for wc in word_confs]
        conf_mean = sum(conf_vals) / len(conf_vals)
        conf_min = min(conf_vals)

    logprobs_raw: list[dict] | None = None
    if full_resp.logprobs_result is not None:
        chosen = getattr(full_resp.logprobs_result, "chosen_candidates", None)
        if chosen:
            logprobs_raw = [
                {"token": c.token, "log_probability": c.log_probability}
                for c in chosen
            ]

    safety_raw: list[dict] | None = None
    if full_resp.safety_ratings is not None:
        safety_raw = [
            {"category": str(getattr(r, "category", r)), "probability": str(getattr(r, "probability", ""))}
            if not isinstance(r, dict) else r
            for r in full_resp.safety_ratings
        ]

    return LLMVisionResponse(
        text=full_resp.text,
        word_confidences=word_confs or None,
        confidence_mean=conf_mean,
        confidence_min=conf_min,
        usage=TokenUsage(
            input_tokens=full_resp.usage.get("input_tokens", 0),
            output_tokens=full_resp.usage.get("output_tokens", 0),
        ),
        finish_reason=full_resp.finish_reason,
        logprobs_raw=logprobs_raw,
        safety_ratings=safety_raw,
    )


def extract_text_via_llm(
    image_paths: list[str],
    provider: str = "gemini",
    model: str | None = None,
    api_key: str | None = None,
    context: dict[str, str] | None = None,
    rate_limit_delay: float = 4.0,
    review_threshold: float = 0.75,
) -> dict[str, LLMVisionResponse]:
    """Extract text from images using LLM Vision API.

    Processes each image sequentially with rate limiting between calls.
    Failed images get an empty-text LLMVisionResponse rather than raising.

    Args:
        image_paths: List of image file paths to process.
        provider: LLM provider name (``"gemini"`` or ``"anthropic"``).
        model: Model ID override (uses provider default if None).
        api_key: API key (falls back to environment variable).
        context: Optional exam context for prompt enrichment.
        rate_limit_delay: Seconds to wait between API calls.
        review_threshold: Confidence threshold for review flagging.

    Returns:
        Dict mapping image path to LLMVisionResponse.
    """
    from forma.llm_provider import create_provider

    llm = create_provider(provider=provider, api_key=api_key, model=model)
    prompt = build_recognition_prompt(context=context)
    use_logprobs = provider.lower() == "gemini"

    results: dict[str, LLMVisionResponse] = {}
    total = len(image_paths)
    ok_count = 0
    err_count = 0

    for idx, image_path in enumerate(image_paths):
        if idx > 0:
            time.sleep(rate_limit_delay)

        image_name = os.path.basename(image_path)
        print(
            f"\r  [{idx + 1}/{total}] {image_name}",
            end="", flush=True,
        )

        try:
            # Try with logprobs first; fall back without if unsupported
            try:
                full_resp = llm.generate_with_image_full(
                    prompt=prompt,
                    image_path=image_path,
                    response_logprobs=use_logprobs,
                )
            except Exception as logprob_exc:
                if use_logprobs and "logprob" in str(logprob_exc).lower():
                    logger.info(
                        "logprobs not supported by model, retrying without: %s",
                        logprob_exc,
                    )
                    use_logprobs = False  # disable for all subsequent images
                    full_resp = llm.generate_with_image_full(
                        prompt=prompt,
                        image_path=image_path,
                        response_logprobs=False,
                    )
                else:
                    raise

            # Validate and retry if invalid
            validation = validate_llm_recognition(
                full_resp.text, full_resp.finish_reason, None,
            )
            if not validation["valid"]:
                logger.warning(
                    "LLM 인식 결과 검증 실패 (%s): %s — 재시도",
                    image_path, "; ".join(validation["warnings"]),
                )
                full_resp = llm.generate_with_image_full(
                    prompt=prompt,
                    image_path=image_path,
                    response_logprobs=use_logprobs,
                    temperature=0.1,
                )

            vision_resp = _build_vision_response(full_resp)
            results[image_path] = vision_resp
            ok_count += 1

            # Show preview of recognized text
            preview = full_resp.text[:40].replace("\n", " ")
            print(
                f"\r  [{idx + 1}/{total}] {image_name} — OK: {preview}..."
                + " " * 10,
                end="", flush=True,
            )
            print()  # newline after each result

        except Exception as exc:
            err_count += 1
            print(
                f"\r  [{idx + 1}/{total}] {image_name} — ERROR"
                + " " * 20,
            )
            logger.warning("LLM 인식 실패: %s — %s", image_path, exc)
            results[image_path] = LLMVisionResponse(
                text="",
                word_confidences=None,
                confidence_mean=None,
                confidence_min=None,
                usage=TokenUsage(input_tokens=0, output_tokens=0),
                finish_reason="ERROR",
                logprobs_raw=None,
                safety_ratings=None,
            )

    print(f"  완료: {ok_count}/{total} 성공, {err_count} 실패")
    return results
