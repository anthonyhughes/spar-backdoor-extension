# Datasets

## BeaverTails

**Always use `datasets/beaver_tails_sample.json`.** This is the only file used for dataset generation — by all triggers, including the semantic trigger. Using the full dataset makes generation prohibitively slow (especially for LLM-based triggers and refusal generation).

`datasets/beaver_tails_full.json` exists solely as a source to regenerate the sample from if needed. It is never loaded directly by any generation code.

The sample file is a flat list of `{"instruction", "output"}` dicts. The full file is a category-grouped dict (`{category: [...]}`). `load_beavertails()` in `craft.py` handles both formats transparently.
