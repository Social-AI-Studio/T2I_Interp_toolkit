"""Shared prompt-pair constants used by both the Recipes page and the
workflow pages. Centralised so the two never drift apart."""

from __future__ import annotations

# Inference prompts often paired with the spectacles recipe.
SPECTACLES_INFERENCE_PROMPTS = "A photo of Jack Sparrow\nA photo of Simba\nA photo of Mickey Mouse"

# Inline pairs that let the spectacles recipe run fully offline (no HF download).
# Each line is `<teacher with-spectacles> | <base without-spectacles>`.
# run_steer.py maps pos to teacher_prompt and neg to base_prompt for LoReFT.
SPECTACLES_INLINE_PAIRS = (
    "A photo of Jack Sparrow with spectacles | A photo of Jack Sparrow\n"
    "A photo of Simba with spectacles | A photo of Simba\n"
    "A photo of Mickey Mouse with spectacles | A photo of Mickey Mouse\n"
    "A photo of Spider-Man with spectacles | A photo of Spider-Man\n"
    "A photo of a businessman with spectacles | A photo of a businessman\n"
    "A photo of a scientist with spectacles | A photo of a scientist\n"
    "A photo of a doctor with spectacles | A photo of a doctor\n"
    "A photo of a librarian with spectacles | A photo of a librarian\n"
    "A photo of a professor with spectacles | A photo of a professor\n"
    "A photo of a teacher with spectacles | A photo of a teacher\n"
    "A photo of a student with spectacles | A photo of a student\n"
    "A photo of a man with spectacles | A photo of a man"
)

# Generic prompts the Stitching mapper trains on. One per line, fed to both
# models since cross-model behaviour transfer typically uses the same prompt.
STITCH_GENERIC_PROMPTS = (
    "a photo of a person\n"
    "a photo of a cat\n"
    "a photo of a landscape\n"
    "a photo of a still life\n"
    "a photo of a city street\n"
    "a photo of a forest\n"
    "a photo of a beach\n"
    "a portrait of a woman\n"
    "a portrait of a man\n"
    "a photo of a sunset"
)

# Inline pair catalogues. Used by recipes whose target concept doesn't have
# a published HuggingFace dataset in the CAA or LoReFT schema. run_steer.py
# builds an in-memory Dataset from these pairs.
DEMOGRAPHIC_PAIRS = (
    "photo of a Black man | photo of a man\n"
    "portrait of a Black man | portrait of a man\n"
    "photograph of a Black father | photograph of a father\n"
    "headshot of a Black man | headshot of a man\n"
    "photo of a Black businessman | photo of a businessman\n"
    "portrait of a Black athlete | portrait of an athlete\n"
    "photo of a Black doctor | photo of a doctor\n"
    "photo of a Black teacher | photo of a teacher"
)
CIGARETTE_PAIRS = (
    "a man holding a cigarette | a man holding a pen\n"
    "a person smoking | a person resting\n"
    "a man smoking outside | a man standing outside\n"
    "a woman with a cigarette | a woman with a coffee cup\n"
    "a person lighting a cigarette | a person lighting a candle\n"
    "close-up of someone smoking | close-up of someone yawning\n"
    "a hand holding a cigarette | a hand holding a phone\n"
    "a character with a cigarette | a character with a coffee"
)
PAINTERLY_PAIRS = (
    "a painterly portrait of a man | a photo of a man\n"
    "an impressionist painting of a woman | a photo of a woman\n"
    "a painterly landscape with mountains | a photo of a landscape with mountains\n"
    "an oil painting of a still life with apples | a photo of a still life with apples\n"
    "a painterly portrait of a child | a photo of a child\n"
    "an impressionist garden scene | a photo of a garden scene\n"
    "a painterly seascape at sunset | a photo of a seascape at sunset\n"
    "a painterly street market | a photo of a street market"
)
OCCUPATION_PAIRS = (
    "a woman doctor | a doctor\n"
    "a woman CEO | a CEO\n"
    "a woman engineer | an engineer\n"
    "a woman scientist | a scientist\n"
    "a woman pilot | a pilot\n"
    "a woman programmer | a programmer\n"
    "a woman surgeon | a surgeon\n"
    "a woman professor | a professor"
)
