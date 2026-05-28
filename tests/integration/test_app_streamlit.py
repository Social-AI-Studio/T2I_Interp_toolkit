"""End-to-end Streamlit tests for the demo app.

These render every page with `streamlit.testing.v1.AppTest` (no actual
model load) and exercise the user-facing flows:

- every page renders without raising
- recipe-payload pipeline pre-fills the workflow page for each workflow
- Steering Step 1 scenario radio + "Use example pairs" loads a clean
  template and the goal/Run-button labels reflect the detected concept
- the placeholder-block and degenerate-strength safety nets disable Run
- home-page CTA wires the FIG2_SPECTACLES_PAYLOAD into session_state
  (the actual page navigation is not exercised because AppTest treats
  `st.switch_page` as a script-stop and discards subsequent state, but
  the payload pipeline is tested separately).

These cover the surfaces the unit tests can't: widget keys, intent
labels, button labels, scenario radios, and the cross-page payload
contract. They take ~3 seconds total on a normal laptop.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

PAGES = [
    "app/streamlit_app.py",
    "app/pages/0_Recipes.py",
    "app/pages/1_Localisation.py",
    "app/pages/2_Steering.py",
    "app/pages/3_Stitching.py",
    "app/pages/4_SAE.py",
    "app/pages/5_Fingerprints.py",
    "app/pages/6_Glossary.py",
]


# ── Every page renders ───────────────────────────────────────────────────────


class TestPagesRender:
    """Render-only smoke test for every page in the sidebar."""

    def test_all_pages_render_without_exception(self):
        # Iterating a list inside one test keeps the failure message helpful
        # (which page broke?) without spamming the runner with 8 separate
        # tests for what is essentially the same check.
        failures = []
        for page in PAGES:
            at = AppTest.from_file(page, default_timeout=20).run()
            if at.exception:
                failures.append((page, [str(e.value) for e in at.exception]))
        assert not failures, f"Pages raised:\n{failures}"


# ── Recipe payload pipeline ──────────────────────────────────────────────────


class TestRecipePayloadPipeline:
    """For each workflow page, set st.session_state['recipe_payload'] before
    the page runs and check that its widget keys get pre-filled."""

    def test_steering_payload_pre_fill(self):
        at = AppTest.from_file("app/pages/2_Steering.py", default_timeout=20)
        at.session_state["recipe_payload"] = {
            "workflow": "Steering",
            "goal": "Test label",
            "fields": {
                "method": "caa",
                "alpha": -10.0,
                "inline_pairs": "X | Y",
                "model_preset": "sd15",
            },
        }
        at.run()
        assert not at.exception
        assert at.session_state["steer_goal"] == "Test label"
        assert at.session_state["steer_method"] == "caa"
        assert at.session_state["steer_alpha"] == -10.0
        assert at.session_state["steer_inline_pairs"] == "X | Y"
        # Payload is consumed so a script-rerun doesn't reapply it.
        try:
            assert at.session_state["recipe_payload"] is None
        except (KeyError, AttributeError):
            pass  # absent is fine; the assertion is "not present"

    def test_localisation_payload_pre_fill(self):
        at = AppTest.from_file("app/pages/1_Localisation.py", default_timeout=20)
        at.session_state["recipe_payload"] = {
            "workflow": "Localisation",
            "goal": "Probe heads",
            "fields": {"prompt": "a unicorn", "factor": 0.0, "target_head": 5},
        }
        at.run()
        assert not at.exception
        assert at.session_state["loc_goal"] == "Probe heads"
        assert at.session_state["loc_prompt"] == "a unicorn"
        assert at.session_state["loc_target_head"] == 5

    def test_stitching_payload_pre_fill(self):
        at = AppTest.from_file("app/pages/3_Stitching.py", default_timeout=20)
        at.session_state["recipe_payload"] = {
            "workflow": "Stitching",
            "goal": "Try a stitch",
            "fields": {"hidden_dim": 384, "num_steps": 150},
        }
        at.run()
        assert not at.exception
        assert at.session_state["stitch_goal"] == "Try a stitch"
        assert at.session_state["stitch_hidden_dim"] == 384
        assert at.session_state["stitch_num_steps"] == 150

    def test_sae_payload_pre_fill(self):
        at = AppTest.from_file("app/pages/4_SAE.py", default_timeout=20)
        at.session_state["recipe_payload"] = {
            "workflow": "SAE",
            "goal": "Find shininess",
            "fields": {"prompt": "a glossy apple", "n_features_to_plot": 6},
        }
        at.run()
        assert not at.exception
        assert at.session_state["sae_goal"] == "Find shininess"
        assert at.session_state["sae_prompt"] == "a glossy apple"
        assert at.session_state["sae_n_features_to_plot"] == 6

    def test_payload_for_wrong_workflow_is_not_consumed(self):
        # A Steering-bound payload landing on the Localisation page must
        # stay untouched so the matching page can pick it up later.
        at = AppTest.from_file("app/pages/1_Localisation.py", default_timeout=20)
        payload = {
            "workflow": "Steering",
            "goal": "for Steering only",
            "fields": {"method": "caa"},
        }
        at.session_state["recipe_payload"] = payload
        at.run()
        assert not at.exception
        # Steering's goal must NOT have leaked into loc_goal.
        assert at.session_state["loc_goal"] == ""
        # Payload still in place.
        assert at.session_state["recipe_payload"]["workflow"] == "Steering"


# ── Steering intent helpers ──────────────────────────────────────────────────


class TestSteeringIntentHelpers:
    """Per-intent flow: pick scenario → click Use example pairs →
    check inline_pairs are loaded, no placeholder warning fires, and the
    Run button label reflects the concept."""

    _INTENT_TO_EXPECTED = {
        "Add an attribute (spectacles, beard, long hair)": ("add", "spectacles"),
        "Suppress a concept (cigarettes, weapons, NSFW)": ("suppress", "cigarette"),
        "Shift toward a demographic (Black, women, older adults)": (
            "shift toward",
            "black",
        ),
        "Apply an art style (painterly, watercolor, anime)": ("apply", "painterly"),
    }

    def test_each_intent_loads_clean_template_and_enables_run(self):
        for intent_label, (verb, concept) in self._INTENT_TO_EXPECTED.items():
            at = AppTest.from_file("app/pages/2_Steering.py", default_timeout=20)
            at.run()
            assert not at.exception, f"initial render failed for {intent_label!r}"

            radio = next(r for r in at.radio if r.label == "I want to:")
            radio.set_value(intent_label)
            at.run()

            use_btn = next(b for b in at.button if b.label == "Use example pairs")
            use_btn.click()
            at.run()

            pairs_text = at.session_state["steer_inline_pairs"]
            assert pairs_text, f"no inline pairs loaded for {intent_label!r}"
            assert len(pairs_text.splitlines()) == 8, f"expected 8 pairs, got {pairs_text!r}"
            # No <UPPERCASE> tokens — templates ship as working examples.
            assert "<" not in pairs_text, f"placeholders in {intent_label!r}: {pairs_text!r}"

            run_btn = next(b for b in at.button if b.label and "Train and" in b.label)
            assert not run_btn.disabled, f"Run disabled for {intent_label!r}"
            assert run_btn.label == f"▶ Train and {verb} {concept}", (
                f"unexpected Run label for {intent_label!r}: {run_btn.label!r}"
            )

    def test_empty_inline_pairs_falls_back_to_generic_run_label(self):
        at = AppTest.from_file("app/pages/2_Steering.py", default_timeout=20)
        at.session_state["steer_inline_pairs"] = ""
        at.run()
        assert not at.exception
        run_btn = next(b for b in at.button if b.label and "Train and" in b.label)
        assert run_btn.label == "▶ Train and generate"


# ── Safety net guards ───────────────────────────────────────────────────────


class TestSafetyNets:
    """The placeholder block (Steering) and degenerate-strength block (SAE)
    disable Run with a clear error/warning. The unit-test version of these
    checks just exercises the helper functions; here we verify the actual
    Run button widget state."""

    def test_steering_run_blocked_by_unresolved_placeholders(self):
        at = AppTest.from_file("app/pages/2_Steering.py", default_timeout=20)
        at.session_state["steer_inline_pairs"] = (
            "a man with <ATTRIBUTE> | a man\na woman with <ATTRIBUTE> | a woman"
        )
        at.run()
        assert not at.exception
        run_btn = next(b for b in at.button if b.label and "Train and" in b.label)
        assert run_btn.disabled, "Run should be disabled while <ATTRIBUTE> placeholders remain"

    def test_sae_run_blocked_when_strengths_degenerate_to_baseline_only(self):
        at = AppTest.from_file("app/pages/4_SAE.py", default_timeout=20)
        at.session_state["sae_strength_lo"] = 0.0
        at.session_state["sae_strength_hi"] = 0.0
        at.run()
        assert not at.exception
        run_btn = next(b for b in at.button if b.label and "Capture and modulate" in b.label)
        assert run_btn.disabled, "SAE Run should be disabled when sweep collapses to [0.0]"


# ── Home page CTA wiring ─────────────────────────────────────────────────────


class TestHomeCTA:
    """The home page's "Reproduce paper Fig 2" button sets
    st.session_state['recipe_payload'] before calling st.switch_page.

    AppTest doesn't preserve session_state across switch_page (the
    runtime treats it as a script stop), so we can't observe the payload
    after the click directly. Instead we verify the payload exported by
    app.lib (FIG2_SPECTACLES_PAYLOAD) lands on the Steering page when
    used as the recipe payload — i.e. the same dict the home CTA sets.
    """

    def test_fig2_payload_pre_fills_steering(self):
        from app.lib import FIG2_SPECTACLES_PAYLOAD

        at = AppTest.from_file("app/pages/2_Steering.py", default_timeout=20)
        at.session_state["recipe_payload"] = FIG2_SPECTACLES_PAYLOAD
        at.run()
        assert not at.exception
        assert at.session_state["steer_method"] == "loreft"
        assert at.session_state["steer_alpha"] == 10.0
        assert at.session_state["steer_model_preset"] == "sdxl_turbo"
        # Inline pairs end up with the spectacles set.
        pairs = at.session_state["steer_inline_pairs"]
        assert "spectacles" in pairs
        assert len(pairs.splitlines()) == 12  # SPECTACLES_INLINE_PAIRS has 12

    def test_home_page_exposes_fig2_button(self):
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=20).run()
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("Reproduce paper Fig 2" in (lbl or "") for lbl in labels)
