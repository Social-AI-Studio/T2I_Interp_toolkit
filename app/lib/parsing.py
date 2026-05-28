"""Shared parser for the `pos | neg` (and `prompt_a | prompt_b`) textareas
on the Steering and Stitching pages."""

from __future__ import annotations

import re

# Placeholder tokens like <ATTRIBUTE>, <CONCEPT>, <DEMO>, <STYLE> that the
# intent-template starter pairs use. Detected so the workflow page can warn
# (or block) before submitting unreplaced placeholders to the trainer.
_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z_]*>")

# Words that pad most prompts and shouldn't be reported as "the concept".
_CONCEPT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def parse_pipe_lines(
    raw: str, *, require_separator: bool = True
) -> tuple[list[dict[str, str] | str], list[int]]:
    """Parse pipe-separated lines.

    Returns `(entries, skipped_line_numbers)`.

    When `require_separator=True` (Steering's `pos | neg` shape), every line
    must contain a `|`. The entry is a dict `{"pos": str, "neg": str}`.

    When `require_separator=False` (Stitching's optional `a | b` shape),
    lines without `|` become plain strings (one prompt fed into both
    models). Lines with `|` become dicts `{"a": str, "b": str}`.

    Empty lines are silently skipped (not counted in `skipped_line_numbers`).
    Lines with a `|` but empty side(s) are reported as skipped.
    """
    out: list[dict[str, str] | str] = []
    skipped: list[int] = []
    for idx, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            left, right = (s.strip() for s in line.split("|", 1))
            if left and right:
                # Field name follows the workflow's convention: Steering uses
                # pos/neg, Stitching uses a/b. The caller knows which shape
                # it wants based on `require_separator`.
                if require_separator:
                    out.append({"pos": left, "neg": right})
                else:
                    out.append({"a": left, "b": right})
            else:
                skipped.append(idx)
        else:
            if require_separator:
                skipped.append(idx)
            else:
                out.append(line)
    return out, skipped


def detect_concept(pairs: list[dict[str, str]]) -> str | None:
    """Best-effort: pull the word or short phrase that's unique to every `pos`
    prompt but absent from the matching `neg` prompt.

    The intuition: in "add an attribute" pairs the `pos` adds the attribute
    word and otherwise looks like the `neg`. In demographic pairs the `pos`
    inserts the demographic word. In style pairs the `pos` swaps "photo" for
    the style. By intersecting the "in pos but not in neg" tokens across
    every pair, what's left is the concept the steering direction will encode.

    Returns the detected concept as a lowercase string, or None when the
    pairs share no common discriminating token (e.g. an empty list, or
    deliberately diverse pairs that don't share one concept).
    """
    if not pairs:
        return None
    common: set[str] | None = None
    for p in pairs:
        pos_tokens = {t.lower().strip(",.!?;:\"'") for t in p["pos"].split()}
        neg_tokens = {t.lower().strip(",.!?;:\"'") for t in p["neg"].split()}
        diff = (pos_tokens - neg_tokens) - _CONCEPT_STOPWORDS
        if common is None:
            common = diff
        else:
            common &= diff
    if not common:
        return None
    # Sort for stable output, drop empty strings from stripping artifacts.
    return " ".join(sorted(t for t in common if t)) or None


def has_unresolved_placeholders(raw: str) -> list[str]:
    """Return the unique placeholder tokens (e.g. `<ATTRIBUTE>`) found in `raw`.

    Empty list means the textarea has no unresolved placeholders. Used by the
    Steering page to block Run when the user clicked *Use template starter*
    but never filled in the angle-bracket placeholder with a real concept.
    """
    seen: list[str] = []
    for match in _PLACEHOLDER_RE.findall(raw):
        if match not in seen:
            seen.append(match)
    return seen
