# Configuration for newsagencies run v1-1-0
# Override MODEL_ID_NEWSAGENCIES and HF_MODEL_NEWSAGENCIES to target a different model.

# Path-safe label used in S3 output paths: <model-short-name>_<commit-short-hash>
MODEL_ID_NEWSAGENCIES ?= ner-newsagency-bert-multilingual_0b5d750

# Full HuggingFace model ID passed to NewsAgenciesPipeline.
# Append @<commit-hash> to pin an exact revision.
HF_MODEL_NEWSAGENCIES ?= impresso-project/ner-newsagency-bert-multilingual

RUN_VERSION_NEWSAGENCIES ?= v1-1-0
