import re
import time
import hashlib
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from app.rag import llm

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# --- PII redaction ---

def redact_pii_regex(text: str) -> str:
    """Fast regex pass for common PII patterns."""
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    text = re.sub(r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b', '[CARD]', text)
    text = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    return text


def redact_pii_presidio(text: str) -> str:
    """Presidio deep scan for names, emails, addresses etc."""
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def redact_pii(text: str) -> tuple[str, bool]:
    """Combined PII redaction. Returns: (redacted_text, pii_found)."""
    original = text
    text = redact_pii_regex(text)
    text = redact_pii_presidio(text)
    pii_found = text != original
    return text, pii_found


# --- Rate limiting ---

class RateLimiter:
    """Token bucket rate limiter — tracks requests per user per time window."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        now = time.time()
        window_start = now - self.window
        self.requests[user_id] = [t for t in self.requests[user_id] if t > window_start]

        remaining = self.max_requests - len(self.requests[user_id])
        if remaining <= 0:
            return False, 0

        self.requests[user_id].append(now)
        return True, remaining - 1

    def time_until_reset(self, user_id: str) -> float:
        if not self.requests[user_id]:
            return 0
        oldest = min(self.requests[user_id])
        return max(0, self.window - (time.time() - oldest))

    def reset(self, user_id: str):
        self.requests[user_id] = []


rate_limiter = RateLimiter(max_requests=5, window_seconds=60)


# --- Guardrail cache ---

guardrail_cache = {}
cache_hits = 0
cache_misses = 0


def get_cache_key(text: str) -> str:
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def check_cache(redacted_question: str):
    global cache_hits
    cache_key = get_cache_key(redacted_question)
    if cache_key in guardrail_cache:
        cache_hits += 1
        return guardrail_cache[cache_key]
    return None


def store_cache(redacted_question: str, classification: str):
    global cache_misses
    cache_misses += 1
    guardrail_cache[get_cache_key(redacted_question)] = classification


def cache_stats():
    total = cache_hits + cache_misses
    hit_rate = cache_hits / total if total > 0 else 0
    return {
        "hits": cache_hits,
        "misses": cache_misses,
        "hit_rate": hit_rate,
        "tokens_saved": cache_hits * 200,
    }


# --- Guardrail LLMs ---

class GuardrailResult(BaseModel):
    classification: Literal["SAFE", "PROMPT_INJECTION", "UNSAFE"]
    reason: str


class OutputGuardrailResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    reason: str


guardrail_llm = llm.with_structured_output(GuardrailResult, method="json_mode")
output_guardrail_llm = llm.with_structured_output(OutputGuardrailResult, method="json_mode")

JAILBREAK_PATTERNS = [
    r"ignore (all |previous |your )?instructions",
    r"reveal (your |the )?system prompt",
    r"pretend you (are|have no)",
    r"act as (if )?you (have no|don't have) restrictions",
    r"DAN mode", r"developer mode",
]