#!/usr/bin/env python3
"""
News Agencies NER CLI Script

This module implements a CLI script for extracting news agencies from text using
Named Entity Recognition (NER). It includes:

1. **Batch GPU Processing**: Optimized NewsAgenciesPipeline with chunk-aware token
   classification for efficient processing of large texts and batches.

2. **Custom Model Architecture**: NewsAgencyTokenClassifier with proper token
   classification head and support for overlapping chunks.

3. **File I/O Operations**: Uses smart_open for seamless handling of both local files
   and S3 URIs, with automatic transport parameter configuration.

4. **Logging Configuration**: Integrates with impresso_cookbook's setup_logging
   function for consistent logging across all project tools.

5. **Error Handling**: Implements robust error handling with proper logging of
   failures during file processing operations.

Example:
    $ python cli_newsagencies.py -i input.jsonl -o output.jsonl --log-level INFO
    $ python cli_newsagencies.py -i s3://bucket/input.jsonl -o s3://bucket/output.jsonl
"""

import logging
import argparse
import json
import sys
from smart_open import open as smart_open  # type: ignore
from typing import List, Optional, Dict, Any, Sequence, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    Pipeline,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput

from impresso_cookbook import (  # type: ignore
    get_timestamp,
    setup_logging,
    get_transport_params,
)

from impresso_pipelines.newsagencies.config import AGENCY_LINKS

log = logging.getLogger(__name__)


class NewsAgencyTokenClassifier(PreTrainedModel):
    """
    Custom token classification model for news agencies.
    """

    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        # Backbone encoder
        self.bert = AutoModel.from_config(config)
        # Dropout
        dropout_prob = (
            getattr(config, "classifier_dropout", None)
            or getattr(config, "hidden_dropout_prob", 0.0)
            or 0.0
        )
        self.dropout = nn.Dropout(dropout_prob)
        # Token classification head
        self.token_classifier = nn.Linear(config.hidden_size, len(config.id2label))
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> TokenClassifierOutput:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.token_classifier(sequence_output)
        if not return_dict:
            return (logits,) + outputs[2:]
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ChunkAwareTokenClassification(Pipeline):
    """
    Chunk-aware token classification supporting batch input and proper reassembly.
    """

    def __init__(
        self,
        model: NewsAgencyTokenClassifier,
        tokenizer: AutoTokenizer,
        min_score: float = 0.50,
        device: Optional[Union[int, str]] = 0,
    ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.min_score = min_score

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        return inputs

    def _forward(self, model_inputs):
        return model_inputs

    def postprocess(self, model_outputs):
        return model_outputs

    def __call__(
        self,
        texts: Union[str, List[str]],
        text_ids: Optional[List[str]] = None,
    ) -> Tuple[List[List[Dict[str, Any]]], int]:
        # Always normalize to list - treat single string as batch of 1
        if isinstance(texts, str):
            texts = [texts]

        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]
        elif isinstance(text_ids, str):
            text_ids = [text_ids]

        # Filter out empty texts completely - don't process or return them
        non_empty_texts = []
        non_empty_text_ids = []

        for i, (text, text_id) in enumerate(zip(texts, text_ids)):
            if len(text.strip()) == 0:
                log.debug("Skipping empty text at position %s (text_id=%s)", i, text_id)
            else:
                non_empty_texts.append(text)
                non_empty_text_ids.append(text_id)

        # If all texts are empty, return empty results
        if not non_empty_texts:
            log.info("All texts are empty, returning empty results")
            return [], 0

        log.debug("ChunkAware processing %s non-empty text(s)", len(non_empty_texts))
        log.debug("Text character counts: %s", [len(text) for text in non_empty_texts])

        # Batch tokenize with overflow handling - only process non-empty texts
        log.debug(
            "Starting tokenization for %s non-empty text(s)...", len(non_empty_texts)
        )
        encodings = self.tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            stride=50,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        sample_mapping = encodings.pop("overflow_to_sample_mapping")
        actual_chunks = len(sample_mapping)
        log.info(
            "🔄 Tokenized %s text(s) into %s actual chunks",
            len(non_empty_texts),
            actual_chunks,
        )
        log.debug("Input shape: %s", encodings["input_ids"].shape)

        # Move to device
        for k, v in encodings.items():
            encodings[k] = v.to(self.model.device)

        # Forward pass
        log.debug("Starting model inference on %s chunks...", len(sample_mapping))
        with torch.no_grad():
            outputs = self.model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
        logits = outputs.logits  # [num_chunks, seq_len, num_labels]
        log.debug("Model inference complete. Processing entities...")

        # Bring tensor inputs back to CPU for postprocess
        offset_mapping = encodings["offset_mapping"].cpu()
        input_ids = encodings["input_ids"].cpu()
        attention_mask = encodings["attention_mask"].cpu()

        # Prepare result containers - only for non-empty texts
        results: List[List[Dict[str, Any]]] = [[] for _ in non_empty_texts]
        seen: List[Set[Tuple[int, int, str]]] = [set() for _ in non_empty_texts]

        log.debug("Starting token-level processing...")
        num_chunks = logits.size(0)
        for chunk_idx in range(num_chunks):
            sample_idx = sample_mapping[chunk_idx]
            seq_logits = logits[chunk_idx]  # [seq_len, num_labels]
            offsets = offset_mapping[chunk_idx]
            mask = attention_mask[chunk_idx]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[chunk_idx])

            chunk_info = "chunk %s/%s for sample %s"
            log.debug(
                "Processing " + chunk_info,
                chunk_idx + 1,
                num_chunks,
                sample_idx,
            )
            log.debug("Active tokens in chunk: %s", mask.sum().item())

            current_word: List[str] = []
            current_logits = None
            current_offsets: List[Tuple[int, int]] = []

            for tok, m, offs, logit in zip(tokens, mask, offsets, seq_logits):
                if m.item() == 0 or tok in {"[CLS]", "[SEP]", "[PAD]"}:
                    continue
                start, end = offs.tolist()
                if not tok.startswith("##"):
                    # finalize previous token
                    if current_word and current_logits is not None:
                        self._finalise_word(
                            sample_idx,
                            current_word,
                            current_logits,
                            current_offsets,
                            seen,
                            results,
                        )
                    # start new
                    current_word = [tok]
                    current_logits = logit
                    current_offsets = [(start, end)]
                else:
                    current_word.append(tok)
                    current_offsets.append((start, end))
            # finalize last token in chunk
            if current_word and current_logits is not None:
                self._finalise_word(
                    sample_idx,
                    current_word,
                    current_logits,
                    current_offsets,
                    seen,
                    results,
                )

        log.debug("Token-level processing complete")
        entity_counts = [len(result) for result in results]

        # Log completion with text details and entity counts
        if len(non_empty_texts) == 1:
            text_len = len(non_empty_texts[0])
            total_entities = sum(entity_counts)
            log.info(
                "Completed processing text_id=%s (length=%s chars): found %s entities",
                non_empty_text_ids[0],
                text_len,
                total_entities,
            )
        else:
            total_entities = sum(entity_counts)
            log.info(
                "Completed processing %s texts: found %s total entities",
                len(non_empty_texts),
                total_entities,
            )

        log.debug("Found entities per text: %s", entity_counts)

        # Clean up GPU memory
        del encodings, outputs, logits
        torch.cuda.empty_cache()

        return results, actual_chunks

    def _finalise_word(
        self,
        sample_idx: int,
        tokens: List[str],
        first_logits: torch.Tensor,
        offsets: List[Tuple[int, int]],
        seen: List[Set[Tuple[int, int, str]]],
        results: List[List[Dict[str, Any]]],
    ) -> None:
        # Confidence & label
        probs = F.softmax(first_logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        label = self.model.config.id2label[int(idx)]

        # Reconstruct word and span
        word = self.tokenizer.convert_tokens_to_string(tokens)
        start = offsets[0][0]
        end = offsets[-1][1]

        log.debug("Word: '%s' | Label: %s | Score: %.3f", word, label, conf)
        log.debug("Span: [%s, %s]", start, end)

        if label == "O" or conf < self.min_score:
            log.debug(
                "Skipping word '%s': label=%s, score=%.3f, threshold=%s",
                word,
                label,
                conf,
                self.min_score,
            )
            return

        key = (start, end, label)
        if key in seen[sample_idx]:
            log.debug("Duplicate entity '%s' at [%s, %s]", word, start, end)
            return
        seen[sample_idx].add(key)

        entity_info = {
            "word": word,
            "entity": label,
            "score": float(conf),
            "start": int(start),
            "stop": int(end),
        }
        results[sample_idx].append(entity_info)
        log.debug("Added entity: %s", entity_info)


class NewsAgenciesPipeline:
    """
    High-level wrapper that batches texts, invokes ChunkAwareTokenClassification,
    and assembles final output with summary and wikidata links.
    """

    def __init__(
        self,
        model_id: str = "impresso-project/ner-newsagency-bert-multilingual",
        min_relevance: float = 0.1,
    ):
        log.info("Initializing NewsAgenciesPipeline with model: %s", model_id)
        log.debug("Min relevance threshold: %s", min_relevance)

        config = AutoConfig.from_pretrained(model_id)
        log.debug("Loaded config: %s", config.name_or_path)
        log.debug("Model supports %s entity labels", len(config.id2label))
        log.debug("Labels: %s", list(config.id2label.values()))

        model = NewsAgencyTokenClassifier.from_pretrained(model_id, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        log.debug("Model max length: %s", tokenizer.model_max_length)

        device = (
            0
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else -1
        )
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("Using GPU: %s (%.1f GB)", torch.cuda.get_device_name(), mem_gb)
        else:
            log.info("Using device: %s", device)

        self.ner = ChunkAwareTokenClassification(
            model=model,
            tokenizer=tokenizer,
            min_score=min_relevance,
            device=device,
        )
        log.info("Pipeline initialization complete")

    def __call__(
        self,
        input_texts: Union[str, List[str]],
        min_relevance: float = 0.1,
        diagnostics: bool = False,
        suppress_entities: Optional[Sequence[str]] = None,
        batch_size: int = 8,
        text_ids: Optional[Union[str, List[str]]] = None,
    ) -> Union[Tuple[Dict[str, Any], int], Tuple[List[Dict[str, Any]], int]]:
        # Track if we need to return single result
        return_single = isinstance(input_texts, str)

        # Always normalize to lists
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Normalize text_ids
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(input_texts))]
        elif isinstance(text_ids, str):
            text_ids = [text_ids]

        log.debug(
            "Processing %s text(s), return_single=%s", len(input_texts), return_single
        )
        log.debug("Min relevance: %s, diagnostics: %s", min_relevance, diagnostics)

        # Log text lengths for debugging
        text_lengths = [len(text) for text in input_texts]
        log.debug("Text lengths: %s", text_lengths)

        # Prepare suppression set
        if suppress_entities is None:
            suppress_entities = []
        suppress_set = list(suppress_entities) + [
            "org.ent.pressagency.unk",
            "ag",
            "pers.ind.articleauthor",
        ]
        SUPPRESS = frozenset(suppress_set)
        log.debug("Suppressing %s entity types: %s", len(SUPPRESS), list(SUPPRESS))

        # Run batch-aware NER
        log.debug("Starting NER processing...")
        raw_entities, actual_chunks = self.ner(input_texts, text_ids=text_ids)

        # Handle case where all texts were empty
        if not raw_entities:
            log.debug("No entities returned (all texts were empty)")
            if return_single:
                return {"agencies": []}, 0
            else:
                return [], 0

        entity_counts = [len(ents) for ents in raw_entities]
        log.debug("NER processing complete. Found entities per text: %s", entity_counts)

        # Since we filtered out empty texts, we need to filter input_texts too
        # to match the raw_entities results
        non_empty_input_texts = [text for text in input_texts if len(text.strip()) > 0]

        # Postprocess each text
        outputs = [
            self._postprocess(ents, text, diagnostics, SUPPRESS)
            for ents, text in zip(raw_entities, non_empty_input_texts)
        ]

        # Log final results
        if return_single:
            result = outputs[0]
            agency_count = len(result.get("agencies", []))
            log.debug("Final result: %s agencies found", agency_count)
        else:
            agency_counts = [len(output.get("agencies", [])) for output in outputs]
            log.debug("Final results: %s agencies found per text", agency_counts)

        if return_single:
            return outputs[0], actual_chunks
        else:
            return outputs, actual_chunks

    def _postprocess(
        self,
        entities: List[Dict[str, Any]],
        input_text: str,
        diagnostics: bool,
        SUPPRESS: frozenset,
    ) -> Dict[str, Any]:
        log.debug("Postprocessing %s raw entities", len(entities))
        log.debug("Input text length: %s characters", len(input_text))
        log.debug("Diagnostics mode: %s", diagnostics)

        # Merge B/I tokens into entities
        merged: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        suppressed_count = 0

        for tok in entities:
            iob, base = tok["entity"].split("-", 1)
            if base in SUPPRESS:
                suppressed_count += 1
                log.debug("Suppressing entity type: %s", base)
                continue
            if iob == "B":
                if current:
                    merged.append(current)
                start, stop = tok["start"], tok["stop"]
                current = {
                    "surface": input_text[start:stop],
                    "entity": base,
                    "start": start,
                    "stop": stop,
                    "relevance": round(tok["score"], 3),
                }
                log.debug("Starting new entity: %s at [%s:%s]", base, start, stop)
            elif iob == "I" and current and current["entity"] == base:
                stop = tok["stop"]
                current["surface"] = input_text[current["start"] : stop]
                current["stop"] = stop
                current["relevance"] = round(max(current["relevance"], tok["score"]), 3)
                log.debug("Extending entity to [%s:%s]", current["start"], stop)
        if current:
            merged.append(current)

        log.debug(
            "Merged into %s entities, suppressed %s", len(merged), suppressed_count
        )

        # Build summary
        summary_dict: Dict[str, float] = {}
        for ent in merged:
            summary_dict[ent["entity"]] = max(
                summary_dict.get(ent["entity"], 0.0), ent["relevance"]
            )
        summary = [
            {"uid": uid, "relevance": relevance}
            for uid, relevance in sorted(
                summary_dict.items(), key=lambda x: x[1], reverse=True
            )
        ]

        log.debug("Created summary with %s unique entities", len(summary))

        # Attach wikidata links
        for agency in merged:
            agency["wikidata_link"] = AGENCY_LINKS.get(
                agency["entity"].replace("org.ent.pressagency.", ""), None
            )
        for agency in summary:
            agency["wikidata_link"] = AGENCY_LINKS.get(
                agency["uid"].replace("org.ent.pressagency.", ""), None
            )

        if diagnostics:
            # Rename 'entity' → 'uid' in diagnostics
            merged = [
                {("uid" if k == "entity" else k): v for k, v in a.items()}
                for a in merged
            ]
            log.debug("Returning diagnostic output with full entity details")
            return {"agencies": merged}

        log.debug("Returning summary output")
        return {"agencies": summary}


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="News Agencies NER CLI script with adaptive chunk-aware batching."
    )
    parser.add_argument(
        "--log-file", dest="log_file", help="Write log to FILE", metavar="FILE"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Input file (required)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Output file (required)",
        required=True,
    )
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=16,
        help=(
            "Target number of chunks per batch for GPU memory optimization "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args(args)


class NewsAgencyProcessor:
    """
    A processor class that uses the NewsAgenciesPipeline to extract entities from text.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str,
        target_chunks: int = 16,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initializes the NewsAgencyProcessor with explicit parameters.

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file
            target_chunks (int): Target number of chunks per batch (default: 16)
            log_level (str): Logging level (default: "INFO")
            log_file (Optional[str]): Path to log file (default: None)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.target_chunks = target_chunks
        self.log_level = log_level
        self.log_file = log_file

        # Configure the module-specific logger
        setup_logging(self.log_level, self.log_file, logger=log)

        log.info("Initializing NewsAgencyProcessor: %s -> %s", input_file, output_file)
        log.debug("Log level: %s", log_level)
        log.debug("Log file: %s", log_file)

        # Initialize timestamp
        self.timestamp = get_timestamp()
        log.debug("Processing timestamp: %s", self.timestamp)

        log.debug("Initializing NewsAgenciesPipeline...")
        self.pipeline = NewsAgenciesPipeline()
        log.info("NewsAgencyProcessor ready")

    def run(self) -> None:
        """
        Runs the processor with length-based batching for optimal GPU utilization.
        """
        import time

        start_time = time.time()
        log.info(
            "Starting file processing: %s -> %s", self.input_file, self.output_file
        )
        log.info("Using chunk-aware batching with target_chunks=%s", self.target_chunks)

        batch_size = 8  # Target batch size (fallback)
        target_chunks = self.target_chunks  # Use configurable target chunks
        read_ahead_size = batch_size * 100  # Read 800 lines ahead for sorting

        line_count = 0
        processed_count = 0
        skipped_count = 0
        total_chunks_processed = 0

        # Adaptive estimation - learns from actual tokenization results
        estimation_history = []  # Store (char_count, actual_chunks) pairs
        chars_per_token_ratio = 4.0  # Starting conservative estimate for French OCR

        def estimate_tokens(text: str) -> int:
            """Adaptive estimate based on observed char/token ratios for French OCR"""
            return int(len(text) / chars_per_token_ratio) + 10  # +10 for special tokens

        def estimate_chunks(text: str) -> int:
            """Estimate number of chunks this text will create"""
            tokens = estimate_tokens(text)
            if tokens <= 512:
                return 1
            # Account for stride overlap when calculating chunks
            return ((tokens - 512) // (512 - 50)) + 1

        def update_estimation(char_count: int, actual_chunks: int) -> None:
            """Update estimation parameters based on actual results"""
            nonlocal chars_per_token_ratio, estimation_history

            # Store recent history (keep last 50 samples)
            estimation_history.append((char_count, actual_chunks))
            if len(estimation_history) > 50:
                estimation_history.pop(0)

            # Calculate actual chars per chunk from recent samples
            if len(estimation_history) >= 5:  # Need minimum samples
                total_chars = sum(chars for chars, _ in estimation_history)
                total_chunks = sum(chunks for _, chunks in estimation_history)

                if total_chunks > 0:
                    # Estimate chars per chunk, then derive chars per token
                    chars_per_chunk = total_chars / total_chunks
                    # Each chunk ~= 462 effective tokens (512 - 50 overlap)
                    new_ratio = chars_per_chunk / 462

                    # Smooth the update to avoid oscillation
                    chars_per_token_ratio = (
                        0.8 * chars_per_token_ratio + 0.2 * new_ratio
                    )

                    log.debug(
                        "Updated estimation: %.2f chars/token "
                        "(from %s samples, %.1f chars/chunk)",
                        chars_per_token_ratio,
                        len(estimation_history),
                        chars_per_chunk,
                    )

        def process_batch(batch_items):
            nonlocal processed_count, skipped_count, total_chunks_processed
            if not batch_items:
                return

            batch_texts = [item["text"] for item in batch_items]
            batch_ids = [item["content_id"] for item in batch_items]
            batch_data = [{"content_id": item["content_id"]} for item in batch_items]

            batch_len = len(batch_texts)
            estimated_chunks = [estimate_chunks(text) for text in batch_texts]
            total_chunks = sum(estimated_chunks)
            max_tokens = max(estimate_tokens(text) for text in batch_texts)

            log.info(
                "🔄 Processing batch of %s texts → %s estimated chunks (max_tokens=%s)",
                batch_len,
                total_chunks,
                max_tokens,
            )

            # Process batch through pipeline
            batch_start_time = time.time()
            results, actual_chunks = self.pipeline(
                input_texts=batch_texts, diagnostics=True, text_ids=batch_ids
            )
            batch_end_time = time.time()

            # Track total chunks processed
            total_chunks_processed += actual_chunks

            # Calculate batch timing
            batch_duration = batch_end_time - batch_start_time
            chunks_per_second = (
                actual_chunks / batch_duration if batch_duration > 0 else 0
            )

            log.info(
                "⏱️  Batch completed in %.2fs (%.1f chunks/sec, %.1fms per chunk)",
                batch_duration,
                chunks_per_second,
                (batch_duration * 1000) / actual_chunks if actual_chunks > 0 else 0,
            )

            # Update estimation based on actual results
            total_chars = sum(len(text) for text in batch_texts)
            update_estimation(total_chars, actual_chunks)

            # Log estimation accuracy
            if total_chunks != actual_chunks:
                accuracy = (1 - abs(total_chunks - actual_chunks) / actual_chunks) * 100
                log.debug(
                    "Estimation accuracy: %.1f%% (predicted %s, actual %s chunks)",
                    accuracy,
                    total_chunks,
                    actual_chunks,
                )

            # Write results for texts that have agencies
            for result, data in zip(results, batch_data):
                agencies_found = len(result.get("agencies", []))
                log.debug(
                    "Found %s agencies for %s", agencies_found, data["content_id"]
                )

                if result.get("agencies"):
                    result["ts"] = self.timestamp
                    result["id"] = data["content_id"]
                    log.debug("Writing result for content %s", data["content_id"])
                    output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    log.debug("No agencies found for %s, skipping", data["content_id"])
                    skipped_count += 1

        try:
            with smart_open(
                self.input_file,
                "r",
                encoding="utf-8",
                transport_params=get_transport_params(self.input_file),
            ) as f, smart_open(
                self.output_file,
                "w",
                encoding="utf-8",
                transport_params=get_transport_params(self.output_file),
            ) as output_stream:

                pending_items = []

                for line in f:
                    line_count += 1
                    log.debug("Processing line %s", line_count)

                    data = json.loads(line.strip())
                    input_text = data.get("ft", "")
                    content_id = data.get("id", data.get("c_id", ""))

                    log.debug("Content ID: %s", content_id)
                    log.debug("Text length: %s characters", len(input_text))

                    # Skip empty texts immediately at input stage
                    if len(input_text.strip()) == 0:
                        log.debug("Skipping empty text for %s", content_id)
                        skipped_count += 1
                        continue

                    # Add to pending items
                    pending_items.append(
                        {
                            "text": input_text,
                            "content_id": content_id,
                            "length": len(input_text),
                        }
                    )

                    # When we have enough items, sort by length and process in batches
                    if len(pending_items) >= read_ahead_size:
                        log.info(
                            "Read-ahead buffer full: %s items collected, "
                            "starting length-based processing",
                            len(pending_items),
                        )

                        # Sort by text length for optimal batching
                        pending_items.sort(key=lambda x: x["length"])

                        # Log length distribution for debugging
                        lengths = [item["length"] for item in pending_items]
                        min_len, max_len = min(lengths), max(lengths)
                        log.debug(
                            "Text length range: %s - %s chars (sorted %s items)",
                            min_len,
                            max_len,
                            len(pending_items),
                        )

                        log.info("🚀 ENTERING CHUNK-AWARE BATCHING SECTION")
                        log.info("Target chunks per batch: %s", target_chunks)

                        # Process in chunk-aware batches
                        current_batch = []
                        current_chunks = 0
                        batch_count = 0

                        log.info(
                            "Starting chunk-aware batching with target=%s chunks",
                            target_chunks,
                        )

                        for i, item in enumerate(pending_items):
                            item_chunks = estimate_chunks(item["text"])

                            log.debug(
                                "Item %s: %s chars → %s chunks (current: %s)",
                                i,
                                item["length"],
                                item_chunks,
                                current_chunks,
                            )

                            # If adding this item would exceed target chunks,
                            # process current batch
                            if (
                                current_batch
                                and current_chunks + item_chunks > target_chunks
                            ):
                                batch_count += 1
                                log.info(
                                    "🔥 CHUNK LIMIT: batch %s with %s texts → %s"
                                    " chunks "
                                    "(would be %s with next item)",
                                    batch_count,
                                    len(current_batch),
                                    current_chunks,
                                    current_chunks + item_chunks,
                                )
                                process_batch(current_batch)
                                current_batch = [item]
                                current_chunks = item_chunks
                                log.debug(
                                    "Started new batch with item %s (%s chunks)",
                                    i,
                                    item_chunks,
                                )
                            else:
                                current_batch.append(item)
                                current_chunks += item_chunks
                                log.debug(
                                    "Added item %s to batch (total: %s texts, %s"
                                    " chunks)",
                                    i,
                                    len(current_batch),
                                    current_chunks,
                                )

                        # Process final batch if any items remain
                        if current_batch:
                            batch_count += 1
                            log.info(
                                "🏁 FINAL BATCH: batch %s with %s texts → %s chunks",
                                batch_count,
                                len(current_batch),
                                current_chunks,
                            )
                            process_batch(current_batch)

                        log.info(
                            "Completed processing %s items in %s chunk-aware batches",
                            len(pending_items),
                            batch_count,
                        )
                        pending_items.clear()

                    if line_count % 1000 == 0:
                        msg = "Read %s lines, %s processed with agencies" % (
                            line_count,
                            processed_count,
                        )
                        log.info(msg)

                # Process remaining items in final batches
                if pending_items:
                    # Sort final batch by length
                    pending_items.sort(key=lambda x: x["length"])
                    log.debug("Processing final %s items", len(pending_items))

                    # Use same chunk-aware batching for final items
                    current_batch = []
                    current_chunks = 0
                    batch_count = 0

                    for item in pending_items:
                        item_chunks = estimate_chunks(item["text"])

                        # If adding this item would exceed target chunks,
                        # process current batch
                        if (
                            current_batch
                            and current_chunks + item_chunks > target_chunks
                        ):
                            batch_count += 1
                            log.debug(
                                "Final chunk-aware batch %s: %s texts → %s chunks",
                                batch_count,
                                len(current_batch),
                                current_chunks,
                            )
                            process_batch(current_batch)
                            current_batch = [item]
                            current_chunks = item_chunks
                        else:
                            current_batch.append(item)
                            current_chunks += item_chunks

                    # Process final batch if any items remain
                    if current_batch:
                        batch_count += 1
                        log.debug(
                            "Final chunk-aware batch %s: %s texts → %s chunks",
                            batch_count,
                            len(current_batch),
                            current_chunks,
                        )
                        process_batch(current_batch)

                    log.info(
                        "Completed processing final %s items in %s chunk-aware batches",
                        len(pending_items),
                        batch_count,
                    )

        except Exception as e:
            log.error("Error processing file: %s", e, exc_info=True)
            sys.exit(1)

        # Calculate final timing metrics
        end_time = time.time()
        total_duration = end_time - start_time
        avg_time_per_chunk = (
            total_duration / total_chunks_processed if total_chunks_processed > 0 else 0
        )
        chunks_per_second = (
            total_chunks_processed / total_duration if total_duration > 0 else 0
        )

        final_msg = "Processing complete: %s lines total, %s processed, %s skipped" % (
            line_count,
            processed_count,
            skipped_count,
        )
        log.info(final_msg)

        log.info("📊 PERFORMANCE SUMMARY:")
        log.info("   Total processing time: %.2f seconds", total_duration)
        log.info("   Total chunks processed: %s", total_chunks_processed)
        log.info("   Average time per chunk: %.2f ms", avg_time_per_chunk * 1000)
        log.info("   Overall throughput: %.1f chunks/second", chunks_per_second)


def main(args: Optional[List[str]] = None) -> None:
    """
    Main function to run the NewsAgencyProcessor.

    Args:
        args: Command-line arguments (uses sys.argv if None)
    """
    options: argparse.Namespace = parse_arguments(args)

    processor: NewsAgencyProcessor = NewsAgencyProcessor(
        input_file=options.input,
        output_file=options.output,
        target_chunks=options.target_chunks,
        log_level=options.log_level,
        log_file=options.log_file,
    )

    # Log the parsed options after logger is configured
    log.info("%s", options)

    processor.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Processing error: %s", e, exc_info=True)
        sys.exit(2)
