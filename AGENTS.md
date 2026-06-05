# AGENTS.md — Impresso Newsagencies Cookbook

## Project Overview

This repository is a **Make-based processing pipeline** for Named Entity Recognition (NER) of news agencies in historical newspaper content, built within the [Impresso project](https://impresso-project.ch/) ecosystem. It detects and links news agency mentions (e.g. "Reuters", "AFP") in digitised newspaper articles to Wikidata entries, processing data stored on S3.

The pipeline is derived from the `impresso-make-cookbook` template and follows its conventions closely. All orchestration is done via GNU Make; Python handles the NLP processing.

---

## Repository Layout

```
.
├── Makefile                        # Top-level entry point; includes cookbook/*.mk files
├── Pipfile / requirements.txt      # Python dependencies
├── dotenv.sample                   # Template for .env (S3 credentials)
├── mypy.ini / pyrightconfig.json   # Python type-checking config
├── config/
│   └── config_v1-0-0.mk           # Versioned Make configuration overrides
├── lib/
│   ├── cli_newsagencies.py         # Main NER processing script
│   └── cli_TEMPLATE.py             # Template for new CLI scripts
└── cookbook/                       # Shared Make include files (the "cookbook")
    ├── paths_newsagencies.mk       # S3/local path definitions for this pipeline
    ├── processing_newsagencies.mk  # Make rules: rebuilt → newsagencies output
    ├── sync_newsagencies.mk        # S3 ↔ local sync targets
    ├── setup_newsagencies.mk       # Setup/check targets for this pipeline
    ├── main_targets.mk             # Top-level `newspaper` and parallelisation targets
    ├── newspaper_list.mk           # Newspaper list discovery from S3
    ├── local_to_s3.mk              # Path conversion utilities
    ├── log.mk                      # Makefile logging (DEBUG/INFO/WARN/ERROR)
    ├── make_settings.mk            # Core Make settings
    ├── setup.mk / setup_python.mk  # General and Python setup targets
    ├── sync.mk / sync_rebuilt.mk   # Generic and rebuilt-data sync
    ├── clean.mk                    # Cleanup targets
    ├── aws.mk / setup_aws.mk       # AWS CLI configuration
    └── lib/                        # Shared Python package (impresso-cookbook)
        ├── common.py               # Logging, timestamps, S3 transport helpers
        ├── s3_aggregator.py        # S3 data aggregation
        ├── s3_to_local_stamps.py   # S3 → local stamp file sync
        └── local_to_s3.py          # Local → S3 upload helper
```

---

## Core Processing Script

**`lib/cli_newsagencies.py`** is the main Python entry point. Key components:

| Class / Function                  | Purpose                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `NewsAgencyProcessorV2`           | Main processor: reads a `.jsonl.bz2` input file in batches, calls the external pipeline, writes annotated output               |
| `NewsAgenciesPipeline` (external) | Imported from `impresso_pipelines.newsagencies`; handles all NER and Wikidata linking internally                               |
| `NewsAgencyTokenClassifier`       | Custom `PreTrainedModel` (BERT backbone + token classification head) — kept for reference, not used by `NewsAgencyProcessorV2` |
| `ChunkAwareTokenClassification`   | `Pipeline` subclass with 512-token/stride-64 chunking — kept for reference, not used by `NewsAgencyProcessorV2`                |
| `main()`                          | CLI entry point; instantiates `NewsAgencyProcessorV2` and calls `.run()`                                                       |

**Model**: `impresso-project/ner-newsagency-bert-multilingual` (multilingual BERT, token classification), loaded by the external `impresso_pipelines` package.  
**Output**: Each article that contains at least one agency is written as a JSON line with an `agencies` list (each entry has `wikidata_id`, `surface`, `offset`, and relevance score) plus a `ts` timestamp and `id` field.

---

## Make Pipeline

### Key Variables (set in `config.local.mk` or env)

| Variable                   | Default                                    | Description                                |
| -------------------------- | ------------------------------------------ | ------------------------------------------ |
| `NEWSPAPER`                | _(required)_                               | Newspaper ID to process (e.g. `GDL`)       |
| `S3_BUCKET_REBUILT`        | `122-rebuilt-final`                        | Input S3 bucket (rebuilt content)          |
| `S3_BUCKET_newsagencies`   | `140-processed-data-sandbox`               | Output S3 bucket                           |
| `MODEL_ID_NEWSAGENCIES`    | `ner-newsagency-bert-multilingual_0b5d750` | Model identifier used in output paths      |
| `RUN_VERSION_NEWSAGENCIES` | `v1-0-0`                                   | Run version string                         |
| `TASK_NEWSAGENCIES`        | `nel`                                      | Task label (`nel` = Named Entity Linking)  |
| `COLLECTION_JOBS`          | `2`                                        | Number of newspapers processed in parallel |
| `LOGGING_LEVEL`            | `INFO`                                     | Makefile and Python log verbosity          |

### Essential Make Targets

| Target                         | Description                                                           |
| ------------------------------ | --------------------------------------------------------------------- |
| `make help`                    | List all available targets                                            |
| `make setup`                   | Verify environment and dependencies                                   |
| `make sync-input`              | Sync rebuilt (input) data from S3 to local stamps                     |
| `make sync-output`             | Sync newsagencies output data from/to S3                              |
| `make newsagencies-target`     | Run NER on all locally available rebuilt files                        |
| `make newspaper NEWSPAPER=GDL` | Full pipeline: sync input → process → upload output for one newspaper |
| `make collection`              | Process all newspapers in parallel (`COLLECTION_JOBS` at a time)      |
| `make clean-sync`              | Remove local S3 stamp files                                           |

### Data Flow

```
S3 (rebuilt .jsonl.bz2)
        │  sync-input
        ▼
Local stamp files  (build.d/.../rebuilt/)
        │  processing rule
        ▼
lib/cli_newsagencies.py
        │
        ▼
Local output files (build.d/.../newsagencies/*.jsonl.bz2)
        │  sync-output / local_to_s3.py
        ▼
S3 (newsagencies output bucket)
```

---

## Environment & Credentials

Create a `.env` file from `dotenv.sample`:

```bash
cp dotenv.sample .env
# Edit .env and fill in:
SE_ACCESS_KEY=<your-s3-access-key>
SE_SECRET_KEY=<your-s3-secret-key>
```

Never commit `.env` or `config.local.mk` to the repository.

---

## Python Dependencies

Python 3.11 is required. Install with:

```bash
pipenv install          # from Pipfile
# or
pip install -r requirements.txt
```

Key packages: `torch`, `transformers`, `smart-open[s3,http]`, `boto3==1.35.95`, `impresso-pipelines[newsagencies]`, `impresso-cookbook` (local editable from `cookbook/lib`).

---

## Conventions & Constraints

- **Never edit cookbook `*.mk` files** unless fixing a bug that applies to all pipelines. Pipeline-specific logic belongs in `cookbook/paths_newsagencies.mk`, `cookbook/processing_newsagencies.mk`, `cookbook/sync_newsagencies.mk`, and `cookbook/setup_newsagencies.mk`.
- **`config.local.mk`** is the correct place for local overrides (S3 buckets, paths, model variants). It is git-ignored.
- **Stamp files**: rebuilt input files are synced as exact-name local stubs (`.jsonl.bz2`, no extra suffix). They mirror the S3 object tree locally without downloading actual content and serve as Make dependency triggers.
- **Output paths** are derived deterministically from `PROCESS_LABEL`, `TASK`, `MODEL_ID`, and `RUN_VERSION` variables, producing a `RUN_ID` like `newsagency-nel-ner-newsagency-bert-multilingual_0b5d750_v1-0-0`.
- **Type checking**: run `mypy lib/` (config in `mypy.ini`) and `pyright` (config in `pyrightconfig.json`) before committing Python changes.
- All Python files use `impresso_cookbook.setup_logging()` for log configuration — do not use `logging.basicConfig()` directly.

---

## Adding or Modifying Pipeline Logic

1. **Change NER model or parameters** → edit `cookbook/paths_newsagencies.mk` (update `MODEL_ID_NEWSAGENCIES`, `RUN_VERSION_NEWSAGENCIES`, etc.)
2. **Change processing CLI arguments** → edit the recipe in `cookbook/processing_newsagencies.mk`
3. **Add a new processing stage** → copy the four `_TEMPLATE.mk` files, rename, and include them from the top-level `Makefile`
4. **Update Python NER logic** → edit `lib/cli_newsagencies.py`; run type checks and test with a small input file before committing
