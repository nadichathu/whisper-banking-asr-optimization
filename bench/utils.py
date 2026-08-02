import re
import os
import platform
import time
import json


def normalize_transcript(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    # remove punctuation, including apostrophes
    text = re.sub(r"[^\w\s]", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(ref_units, hyp_units) -> int:
    """Standard unit-cost Levenshtein distance. Used for both word-level
    (WER) and character-level (CER) error rate computation."""
    n = len(ref_units)
    m = len(hyp_units)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_units[i - 1] == hyp_units[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )
    return dp[n][m]


def _wer_words(ref_words, hyp_words) -> float:
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    return _edit_distance(ref_words, hyp_words) / float(n)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = (substitutions + deletions + insertions) / reference_word_count.

    NOTE: this is NOT clipped to [0, 1]. WER can legitimately exceed 1.0 when
    the hypothesis contains substantially more words than the reference --
    e.g. hallucinated/over-generated output on a short reference utterance.
    Clipping would silently hide exactly this failure mode, which is
    directly relevant to known Whisper hallucination behaviour on short
    audio (see Chapter 2). Only a floor at 0.0 is applied, since WER cannot
    be negative.
    """
    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)
    ref_words = ref.split() if ref else []
    hyp_words = hyp.split() if hyp else []
    wer = _wer_words(ref_words, hyp_words)
    return max(0.0, float(wer))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = (substitutions + deletions + insertions) / reference_char_count,
    computed on normalized (lowercased, punctuation-stripped) text with
    spaces retained as characters, consistent with common CER conventions.

    Same non-clipping rationale as compute_wer() applies here.
    """
    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    distance = _edit_distance(list(ref), list(hyp))
    cer = distance / float(len(ref))
    return max(0.0, float(cer))


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def env_metadata():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def json_dumps_safe(obj):
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return json.dumps(str(obj))
