"""Unit tests for the pure-Python helpers in app/lib/.

These cover the surfaces the Streamlit pages depend on: the inline-pair
parser, the placeholder detector, the recipe-payload intake helper, and
the baseline/modified image pairer. None of these touch Streamlit at
runtime, so they're cheap to test outside an AppTest harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.lib.outputs import collect_images, load_fingerprint, pair_baseline_modified
from app.lib.pages import apply_payload
from app.lib.parsing import detect_concept, has_unresolved_placeholders, parse_pipe_lines

# ── parse_pipe_lines ─────────────────────────────────────────────────────────


class TestParsePipeLines:
    def test_steering_shape_basic(self):
        out, skipped = parse_pipe_lines("a | b\nc | d", require_separator=True)
        assert out == [{"pos": "a", "neg": "b"}, {"pos": "c", "neg": "d"}]
        assert skipped == []

    def test_steering_shape_skips_lines_without_separator(self):
        out, skipped = parse_pipe_lines("a | b\nno pipe here\nc | d", require_separator=True)
        assert out == [{"pos": "a", "neg": "b"}, {"pos": "c", "neg": "d"}]
        assert skipped == [2]

    def test_steering_shape_skips_empty_pos_or_neg(self):
        out, skipped = parse_pipe_lines("a |\n | b\n | \nc | d", require_separator=True)
        assert out == [{"pos": "c", "neg": "d"}]
        assert skipped == [1, 2, 3]

    def test_steering_shape_ignores_blank_lines(self):
        out, skipped = parse_pipe_lines("\na | b\n\n\nc | d\n", require_separator=True)
        assert out == [{"pos": "a", "neg": "b"}, {"pos": "c", "neg": "d"}]
        assert skipped == []

    def test_stitching_shape_optional_separator(self):
        # Stitching: plain string lines stay as strings, lines with | become {a, b}.
        out, skipped = parse_pipe_lines(
            "a plain prompt\nleft | right\nanother plain", require_separator=False
        )
        assert out == [
            "a plain prompt",
            {"a": "left", "b": "right"},
            "another plain",
        ]
        assert skipped == []

    def test_stitching_shape_still_skips_half_empty_pipes(self):
        out, skipped = parse_pipe_lines(
            "good plain\nbroken |\nleft | right", require_separator=False
        )
        assert out == ["good plain", {"a": "left", "b": "right"}]
        assert skipped == [2]

    def test_empty_input(self):
        out, skipped = parse_pipe_lines("", require_separator=True)
        assert out == []
        assert skipped == []

    def test_whitespace_only(self):
        out, skipped = parse_pipe_lines("   \n\t\n  ", require_separator=True)
        assert out == []
        assert skipped == []


# ── detect_concept ───────────────────────────────────────────────────────────


class TestDetectConcept:
    def test_add_attribute_spectacles(self):
        pairs = [
            {"pos": "a man with spectacles", "neg": "a man"},
            {"pos": "a woman with spectacles", "neg": "a woman"},
            {"pos": "a child with spectacles", "neg": "a child"},
        ]
        assert detect_concept(pairs) == "spectacles"

    def test_shift_demographic_black(self):
        pairs = [
            {"pos": "photo of a Black man", "neg": "photo of a man"},
            {"pos": "portrait of a Black man", "neg": "portrait of a man"},
            {"pos": "photo of a Black woman", "neg": "photo of a woman"},
        ]
        # Tokens are lowercased before set ops.
        assert detect_concept(pairs) == "black"

    def test_suppress_concept_cigarette(self):
        # The concept lives on the pos side; the neg has a benign substitute.
        # detect_concept reports the pos-only token (cigarette), not the
        # neg-only one (pen/coffee/...), which is the right thing for a
        # "what is the direction encoding" answer.
        pairs = [
            {"pos": "a man holding a cigarette", "neg": "a man holding a pen"},
            {"pos": "close-up of a cigarette", "neg": "close-up of a flower"},
            {"pos": "a hand holding a cigarette", "neg": "a hand holding a phone"},
        ]
        assert detect_concept(pairs) == "cigarette"

    def test_apply_style_painterly(self):
        pairs = [
            {"pos": "a painterly portrait of a man", "neg": "a photo of a man"},
            {"pos": "a painterly landscape", "neg": "a photo of a landscape"},
            {"pos": "a painterly still life", "neg": "a photo of a still life"},
        ]
        # Across all three, only "painterly" is uniformly in pos but not in neg.
        # (Pair 1 also adds "portrait" but pair 2 doesn't.)
        assert detect_concept(pairs) == "painterly"

    def test_returns_none_on_empty(self):
        assert detect_concept([]) is None

    def test_returns_none_when_no_common_diff(self):
        # Each pair adds a different word — no shared concept across all pairs.
        pairs = [
            {"pos": "a red apple", "neg": "an apple"},
            {"pos": "a green pear", "neg": "a pear"},
        ]
        # "red" only in pair 1, "green" only in pair 2; intersection empty.
        assert detect_concept(pairs) is None

    def test_returns_none_when_pos_equals_neg(self):
        pairs = [{"pos": "a photo of a person", "neg": "a photo of a person"}]
        assert detect_concept(pairs) is None

    def test_strips_punctuation_and_lowercases(self):
        pairs = [
            {"pos": "a man with Spectacles!", "neg": "a man."},
            {"pos": "a woman with spectacles,", "neg": "a woman."},
        ]
        assert detect_concept(pairs) == "spectacles"

    def test_filters_stopwords(self):
        # The word "with" is in every pos but never in neg — it would
        # show up if we didn't filter stopwords. We do, so it doesn't.
        pairs = [
            {"pos": "a man with spectacles", "neg": "a man"},
            {"pos": "a woman with spectacles", "neg": "a woman"},
        ]
        out = detect_concept(pairs)
        assert "with" not in (out or "")
        assert out == "spectacles"


# ── has_unresolved_placeholders ──────────────────────────────────────────────


class TestHasUnresolvedPlaceholders:
    def test_finds_template_placeholders(self):
        text = "a man with <ATTRIBUTE> | a man\na woman with <ATTRIBUTE> | a woman"
        assert has_unresolved_placeholders(text) == ["<ATTRIBUTE>"]

    def test_finds_multiple_distinct(self):
        text = "<CONCEPT> | <SUBSTITUTE>\n<DEMO> photo | photo"
        # Order = first-seen.
        assert has_unresolved_placeholders(text) == ["<CONCEPT>", "<SUBSTITUTE>", "<DEMO>"]

    def test_dedupes_repeats(self):
        text = "<X> | <X>\n<X> apple | <X> orange"
        assert has_unresolved_placeholders(text) == ["<X>"]

    def test_clean_text_returns_empty(self):
        text = "a man with spectacles | a man\na woman with glasses | a woman"
        assert has_unresolved_placeholders(text) == []

    def test_underscores_in_token_ok(self):
        text = "the <TARGET_CONCEPT> | the plain version"
        assert has_unresolved_placeholders(text) == ["<TARGET_CONCEPT>"]

    def test_lowercase_brackets_are_not_placeholders(self):
        # `<a>` is not an unresolved placeholder; we only flag UPPERCASE.
        text = "an <a> tag | another <b>"
        assert has_unresolved_placeholders(text) == []


# ── apply_payload ────────────────────────────────────────────────────────────


class TestApplyPayload:
    def _defaults(self):
        return {"x_goal": "", "x_method": "loreft", "x_alpha": 1.0}

    def test_initialises_defaults_on_empty_state(self):
        state: dict = {}
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        assert state == {"x_goal": "", "x_method": "loreft", "x_alpha": 1.0}

    def test_preserves_existing_state_values(self):
        # setdefault must not overwrite a key the user already changed.
        state = {"x_method": "caa", "x_alpha": 5.0}
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        assert state["x_method"] == "caa"
        assert state["x_alpha"] == 5.0

    def test_consumes_matching_payload(self):
        state: dict = {
            "recipe_payload": {
                "workflow": "X",
                "goal": "My run",
                "fields": {"method": "ksteer", "alpha": 7.5},
            }
        }
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        assert "recipe_payload" not in state
        assert state["x_goal"] == "My run"
        assert state["x_method"] == "ksteer"
        assert state["x_alpha"] == 7.5

    def test_leaves_payload_for_other_workflow(self):
        # An X-page should NOT pop a Y-bound payload.
        original = {"workflow": "Y", "goal": "for Y", "fields": {"method": "caa"}}
        state = {"recipe_payload": original}
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        assert state["recipe_payload"] is original

    def test_ignores_unknown_field_keys(self):
        # Schema drift: unknown keys in payload.fields don't blow up.
        state = {
            "recipe_payload": {
                "workflow": "X",
                "fields": {"method": "caa", "future_knob": 99, "another": "yo"},
            }
        }
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        assert state["x_method"] == "caa"
        assert "x_future_knob" not in state
        assert "x_another" not in state

    def test_empty_goal_does_not_overwrite(self):
        state = {
            "x_goal": "pre-existing label",
            "recipe_payload": {"workflow": "X", "goal": "", "fields": {}},
        }
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        # An empty payload-goal must not blank out the existing label.
        assert state["x_goal"] == "pre-existing label"

    def test_missing_payload_keys_are_safe(self):
        # Payload without "fields" or "goal" still consumes cleanly.
        state = {"recipe_payload": {"workflow": "X"}}
        apply_payload(state, prefix="x", defaults=self._defaults(), workflow_name="X")
        assert "recipe_payload" not in state
        # Defaults still installed.
        assert state["x_method"] == "loreft"


# ── pair_baseline_modified ───────────────────────────────────────────────────


class TestPairBaselineModified:
    def _paths(self, names: list[str]) -> list[Path]:
        # The pairer only reads .name, so Path() of bare names is enough.
        return [Path(n) for n in names]

    def test_steering_baseline_steered_pair_per_prompt(self):
        imgs = self._paths(["baseline_0.png", "baseline_1.png", "steered_0.png", "steered_1.png"])
        out = pair_baseline_modified(imgs, modified_kinds=("steered",), label_prefix="prompt")
        assert len(out) == 2
        labels = [t[0] for t in out]
        assert labels == ["prompt 0", "prompt 1"]
        assert all(b is not None and m is not None for _, b, m in out)

    def test_localisation_single_baseline_shared_across_heads(self):
        # The Localisation bug: only one baseline image, multiple modified.
        # The baseline must be paired into EVERY triple, not stranded at "0".
        imgs = self._paths(["baseline.png", "modified_0.png", "modified_3.png", "modified_5.png"])
        out = pair_baseline_modified(imgs, modified_kinds=("modified",), label_prefix="head")
        assert [t[0] for t in out] == ["head 0", "head 3", "head 5"]
        # All three triples must have the shared baseline filled in.
        assert all(b is not None and b.name == "baseline.png" for _, b, _ in out)

    def test_indexed_baseline_wins_over_shared(self):
        # If both bare baseline AND indexed baseline_<idx> exist, the indexed
        # one wins for that index, the shared one fills the rest.
        imgs = self._paths(
            [
                "baseline.png",  # shared
                "baseline_3.png",  # indexed
                "modified_0.png",
                "modified_3.png",
            ]
        )
        out = pair_baseline_modified(imgs, modified_kinds=("modified",), label_prefix="head")
        by_label = {label: (b, m) for label, b, m in out}
        assert by_label["head 0"][0].name == "baseline.png"
        assert by_label["head 3"][0].name == "baseline_3.png"

    def test_modified_without_baseline_renders_with_none_baseline(self):
        imgs = self._paths(["modified_5.png"])
        out = pair_baseline_modified(imgs, modified_kinds=("modified",), label_prefix="head")
        assert len(out) == 1
        assert out[0][0] == "head 5"
        assert out[0][1] is None
        assert out[0][2] is not None

    def test_baseline_only_still_displayed(self):
        # If nothing matched as modified, the lone baseline still gets shown.
        imgs = self._paths(["baseline.png"])
        out = pair_baseline_modified(imgs, modified_kinds=("modified",), label_prefix="head")
        assert len(out) == 1
        assert out[0][1] is not None
        assert out[0][2] is None

    def test_unrecognised_filenames_show_up_as_leftovers(self):
        # Things like `mapper.pt` and `random.png` go to a leftover triple
        # with (name, None, image) so the page still displays them.
        imgs = self._paths(["stitched_xyz.png", "report.png"])
        out = pair_baseline_modified(imgs, modified_kinds=("steered",), label_prefix="prompt")
        # Both pass through as leftovers; no baseline pairs constructed.
        labels = [t[0] for t in out]
        assert "stitched_xyz.png" in labels
        assert "report.png" in labels

    def test_case_insensitive_filenames(self):
        imgs = self._paths(["BASELINE_0.PNG", "Steered_0.png"])
        out = pair_baseline_modified(imgs, modified_kinds=("steered",), label_prefix="prompt")
        assert len(out) == 1
        assert out[0][1] is not None
        assert out[0][2] is not None


# ── collect_images + load_fingerprint with prefix-sibling search ─────────────


class TestCollectImagesSiblings:
    """Regression tests for the Steering output-dir suffix bug.

    `run_steer.py` rewrites cfg.output_dir to `<orig>_<block>_alpha=<a>`,
    so images and fingerprint.json end up in a sibling of the directory
    the Streamlit page created. The helpers need to find them.
    """

    def test_collect_walks_sibling_with_prefix(self, tmp_path):
        # The directory the Streamlit page created (empty after the script ran).
        out_dir = tmp_path / "streamlit_steer_xxx"
        out_dir.mkdir()
        # The directory the script actually wrote to (sibling with suffix).
        actual = tmp_path / "streamlit_steer_xxx_up_blocks_alpha=12"
        actual.mkdir()
        (actual / "baseline_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (actual / "steered_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        # Default behaviour: only walks out_dir, misses the images.
        assert collect_images(out_dir) == []
        # With the flag, finds them in the sibling.
        images = collect_images(out_dir, include_prefix_siblings=True)
        assert sorted(p.name for p in images) == ["baseline_0.png", "steered_0.png"]

    def test_collect_does_not_grab_unrelated_siblings(self, tmp_path):
        out_dir = tmp_path / "streamlit_steer_xxx"
        out_dir.mkdir()
        # A sibling that doesn't share the prefix should be ignored.
        unrelated = tmp_path / "streamlit_steer_OTHER"
        unrelated.mkdir()
        (unrelated / "wrong.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        assert collect_images(out_dir, include_prefix_siblings=True) == []

    def test_load_fingerprint_walks_sibling(self, tmp_path):
        out_dir = tmp_path / "streamlit_steer_yyy"
        out_dir.mkdir()
        actual = tmp_path / "streamlit_steer_yyy_mid_block_alpha=8"
        actual.mkdir()
        (actual / "fingerprint.json").write_text(
            '{"fingerprint_hash": "abc123", "workflow": "steer"}'
        )

        # Default: doesn't find it.
        assert load_fingerprint(out_dir) is None
        # With the flag: finds it.
        fp = load_fingerprint(out_dir, include_prefix_siblings=True)
        assert fp is not None
        assert fp["fingerprint_hash"] == "abc123"

    def test_collect_with_flag_off_does_not_break_existing(self, tmp_path):
        # Sanity: the default-flag-off behaviour still walks the dir itself
        # (this is what the Fingerprints page relies on).
        d = tmp_path / "X"
        d.mkdir()
        (d / "baseline_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        assert [p.name for p in collect_images(d)] == ["baseline_0.png"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
