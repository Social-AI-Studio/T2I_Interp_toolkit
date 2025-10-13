# text2image-10k-with-spectacles-pairs

A 10k **text-only** dataset combining:
- **9,000** prompts sampled (streaming) from `jackyhate/text-to-image-2M`
- **1,000** rows drawn from user-provided *spectacles* **pairs** (both base and with "wearing spectacles" versions are included as separate rows)

## Schema
- `text`: the prompt
- `is_from_pair`: whether this row came from the spectacles pairs
- `has_spectacles_phrase`: whether the text explicitly includes "wearing spectacles"
- `source`: `"jackyhate/text-to-image-2M"` or `"user_specs"`

## Construction
- Spectacles rows are **equally spaced** in the 10k corpus (one every ~10 rows).
- 80/20 split performed **without shuffling** to preserve spacing.

## Intended Use
- Bias/steering analyses for T2I diffusion, cross-attention & UNet activation studies.

## License
- Base data inherits license of `jackyhate/text-to-image-2M`.
- User spectacles prompts: specify your license here.
