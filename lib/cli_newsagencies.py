#!/usr/bin/env python3
"""
News Agencies NER CLI Script using impresso_pipelines.newsagencies

This module implements a CLI script for extracting news agencies from text using
the NewsAgenciesPipeline from impresso_pipelines.newsagencies package.

Key features:
1. **External Pipeline Integration**: Uses NewsAgenciesPipeline from impresso_pipelines
2. **Batch Processing**: Optimized batching for efficient processing of large datasets
3. **File I/O Operations**: Uses smart_open for seamless handling of local files and S3 URIs
4. **Logging Configuration**: Consistent logging across all project tools
5. **Error Handling**: Robust error handling with proper logging

Example:
    $ python cli_newsagencies_v_pipeline.py -i input.jsonl -o output.jsonl --log-level INFO
    $ python cli_newsagencies_v_pipeline.py -i s3://bucket/input.jsonl -o s3://bucket/output.jsonl
"""

import logging
import argparse
import json
import sys
import time
from smart_open import open as smart_open  # type: ignore
from typing import List, Optional, Dict, Any

# NEW: optionally use Hugging Face Datasets for efficient streaming + batching
try:
    from datasets import load_dataset  # type: ignore
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

from impresso_pipelines.newsagencies import NewsAgenciesPipeline # type: ignore
from impresso_cookbook import ( # type: ignore
    get_timestamp,
    setup_logging,
    get_transport_params,
)

log = logging.getLogger(__name__)


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="News Agencies NER CLI script using impresso_pipelines.newsagencies."
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing texts (default: %(default)s)",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.1,
        help="Minimum relevance threshold for entities (default: %(default)s)",
    )
    # NEW: engine selector
    parser.add_argument(
        "--engine",
        choices=["datasets", "legacy"],
        default="datasets",
        help="Processing engine: 'datasets' (streaming, recommended) or 'legacy' (in-memory). Default: %(default)s",
    )
    return parser.parse_args(args)


class NewsAgencyProcessorV2:
    """
    A processor class that uses the external NewsAgenciesPipeline to extract entities from text.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 8,
        min_relevance: float = 0.1,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        engine: str = "datasets",  # NEW
    ) -> None:
        """
        Initializes the NewsAgencyProcessorV2 with explicit parameters.

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file
            batch_size (int): Batch size for processing texts (default: 8)
            min_relevance (float): Minimum relevance threshold (default: 0.1)
            log_level (str): Logging level (default: "INFO")
            log_file (Optional[str]): Path to log file (default: None)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.min_relevance = min_relevance
        self.log_level = log_level
        self.log_file = log_file
        self.engine = engine  # NEW

        # Configure the module-specific logger
        setup_logging(self.log_level, self.log_file, logger=log)

        log.info("Initializing NewsAgencyProcessorV2: %s -> %s", input_file, output_file)
        log.debug("Log level: %s", log_level)
        log.debug("Log file: %s", log_file)
        log.debug("Batch size: %s", batch_size)
        log.debug("Min relevance: %s", min_relevance)
        log.debug("Engine: %s (datasets available: %s)", self.engine, HAS_DATASETS)  # NEW

        # Initialize timestamp
        self.timestamp = get_timestamp()
        log.debug("Processing timestamp: %s", self.timestamp)

        log.debug("Initializing NewsAgenciesPipeline...")
        self.pipeline = NewsAgenciesPipeline()
        log.info("NewsAgencyProcessorV2 ready")

    def _run_with_datasets(self) -> None:
        """
        Streaming processing using Hugging Face Datasets to keep GPU busy and reduce memory use.
        """
        if not HAS_DATASETS:
            log.warning("Datasets is not available. Falling back to legacy engine.")
            return self._run_legacy()

        start_time = time.time()
        log.info("Starting streaming processing (datasets): %s -> %s", self.input_file, self.output_file)
        log.info("Using batch_size=%s", self.batch_size)

        line_count = 0
        processed_count = 0
        skipped_count = 0
        entity_type_counts: Dict[str, int] = {}

        # Build a streaming dataset over the JSONL file
        try:
            # Streaming avoids reading all into memory and works with local paths or URLs/S3 (if supported)
            ds = load_dataset(
                "json",
                data_files=self.input_file,
                split="train",
                streaming=True,
            )
        except Exception as e:
            log.error("Failed to initialize datasets streaming: %s", e, exc_info=True)
            sys.exit(1)

        def iter_items():
            nonlocal line_count, skipped_count
            for row in ds:
                line_count += 1
                text = row.get("ft", "") or ""
                content_id = row.get("id", row.get("c_id", ""))
                if len(text.strip()) == 0:
                    skipped_count += 1
                    continue
                yield {
                    "text": text,
                    "content_id": content_id,
                    "length": len(text),
                }

        def process_batch(batch_items, output_stream):
            nonlocal processed_count, skipped_count, entity_type_counts
            if not batch_items:
                return

            batch_texts = [it["text"] for it in batch_items]
            batch_ids = [it["content_id"] for it in batch_items]

            text_lengths = [len(t) for t in batch_texts]
            log.info(
                "🔄 Processing batch of %s texts (total: %s chars, max: %s chars)",
                len(batch_texts),
                sum(text_lengths),
                max(text_lengths) if text_lengths else 0,
            )

            batch_start_time = time.time()
            results = self.pipeline(
                input_texts=batch_texts,
                min_relevance=self.min_relevance,
                diagnostics=True,
            )
            batch_duration = time.time() - batch_start_time
            log.info(
                "⏱️  Batch completed in %.2fs (%.1f texts/sec)",
                batch_duration,
                len(batch_texts) / batch_duration if batch_duration > 0 else 0,
            )

            if isinstance(results, tuple):
                batch_results = results[0]
            else:
                batch_results = results
            if not isinstance(batch_results, list):
                batch_results = [batch_results]

            for result, cid in zip(batch_results, batch_ids):
                agencies = result.get("agencies", [])
                if agencies:
                    for agency in agencies:
                        entity_type = agency.get("uid", agency.get("entity", "unknown"))
                        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                    result["ts"] = self.timestamp
                    result["id"] = cid
                    output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    skipped_count += 1

        # Iterate, batch, and write results
        current_batch: List[Dict[str, Any]] = []
        batch_count = 0

        try:
            with smart_open(
                self.output_file,
                "w",
                encoding="utf-8",
                transport_params=get_transport_params(self.output_file),
            ) as output_stream:
                for item in iter_items():
                    current_batch.append(item)
                    if len(current_batch) >= self.batch_size:
                        batch_count += 1
                        process_batch(current_batch, output_stream)
                        current_batch = []
                if current_batch:
                    batch_count += 1
                    log.info("🏁 FINAL BATCH: batch %s with %s texts", batch_count, len(current_batch))
                    process_batch(current_batch, output_stream)
        except Exception as e:
            log.error("Error during datasets streaming run: %s", e, exc_info=True)
            sys.exit(1)

        # Final stats and logs
        total_duration = time.time() - start_time
        items_per_second = (line_count / total_duration) if total_duration > 0 else 0.0
        log.info("Processing complete: %s lines total, %s processed, %s skipped", line_count, processed_count, skipped_count)

        if entity_type_counts:
            log.info("📈 ENTITY TYPE SUMMARY:")
            total_entities = sum(entity_type_counts.values())
            log.info("   Total entities found: %s", total_entities)
            for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_entities) * 100 if total_entities > 0 else 0
                log.info("   %s: %s entities (%.1f%%)", entity_type, count, pct)
        else:
            log.info("📈 ENTITY TYPE SUMMARY: No entities found")

        log.info("📊 PERFORMANCE SUMMARY:")
        log.info("   Total processing time: %.2f seconds", total_duration)
        log.info("   Total lines read: %s", line_count)
        log.info("   Overall throughput: %.1f items/second", items_per_second)

    def _run_legacy(self) -> None:
        """
        Previous in-memory batching implementation (unchanged).
        """
        start_time = time.time()
        log.info(
            "Starting file processing: %s -> %s", self.input_file, self.output_file
        )
        log.info("Using batch processing with batch_size=%s", self.batch_size)

        line_count = 0
        processed_count = 0
        skipped_count = 0

        # Track entity types across all processed texts
        entity_type_counts = {}  # entity_type -> count

        def process_batch(batch_items):
            nonlocal processed_count, skipped_count, entity_type_counts
            if not batch_items:
                return

            batch_texts = [item["text"] for item in batch_items]
            batch_ids = [item["content_id"] for item in batch_items]
            batch_data = [{"content_id": item["content_id"]} for item in batch_items]

            batch_len = len(batch_texts)
            text_lengths = [len(text) for text in batch_texts]
            total_chars = sum(text_lengths)
            max_chars = max(text_lengths)

            log.info(
                "🔄 Processing batch of %s texts (total: %s chars, max: %s chars)",
                batch_len,
                total_chars,
                max_chars,
            )

            # Process batch through external pipeline
            batch_start_time = time.time()
            
            # Call the external pipeline with the batch of texts
            results = self.pipeline(
                input_texts=batch_texts,
                min_relevance=self.min_relevance,
                diagnostics=True,  # Get detailed results like the original script
            )
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            log.info(
                "⏱️  Batch completed in %.2fs (%.1f texts/sec)",
                batch_duration,
                batch_len / batch_duration if batch_duration > 0 else 0,
            )

            # Handle the pipeline results - it might return a tuple or just results
            if isinstance(results, tuple):
                batch_results = results[0]  # Get the actual results from tuple
            else:
                batch_results = results

            # Ensure we have a list of results matching our batch
            if not isinstance(batch_results, list):
                batch_results = [batch_results]  # Single result to list

            # Write results for texts that have agencies
            for result, data in zip(batch_results, batch_data):
                agencies_found = len(result.get("agencies", []))
                log.debug(
                    "Found %s agencies for %s", agencies_found, data["content_id"]
                )

                if result.get("agencies"):
                    # Count entity types for statistics
                    for agency in result.get("agencies", []):
                        entity_type = agency.get("uid", agency.get("entity", "unknown"))
                        entity_type_counts[entity_type] = (
                            entity_type_counts.get(entity_type, 0) + 1
                        )

                    result["ts"] = self.timestamp
                    result["id"] = data["content_id"]
                    log.debug("Writing result for content %s", data["content_id"])
                    output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    log.debug("No agencies found for %s, skipping", data["content_id"])
                    skipped_count += 1

        # Read all data and process in batches
        log.info("Reading all data from %s...", self.input_file)
        all_items = []

        try:
            with smart_open(
                self.input_file,
                "r",
                encoding="utf-8",
                transport_params=get_transport_params(self.input_file),
            ) as f:
                for line in f:
                    line_count += 1
                    data = json.loads(line.strip())
                    input_text = data.get("ft", "")
                    content_id = data.get("id", data.get("c_id", ""))

                    # Skip empty texts immediately at input stage
                    if len(input_text.strip()) == 0:
                        log.debug("Skipping empty text for %s", content_id)
                        skipped_count += 1
                        continue

                    # Only keep essential fields to minimize memory usage
                    all_items.append(
                        {
                            "text": input_text,
                            "content_id": content_id,
                            "length": len(input_text),
                        }
                    )

                    if line_count % 10000 == 0:
                        log.info(
                            "Read %s lines, collected %s valid items",
                            line_count,
                            len(all_items),
                        )

            log.info(
                "Completed reading %s lines, collected %s valid items",
                line_count,
                len(all_items),
            )

            # Log length distribution
            if all_items:
                lengths = [item["length"] for item in all_items]
                min_len, max_len = min(lengths), max(lengths)
                median_len = lengths[len(lengths) // 2]
                log.info(
                    "Text length range: %s - %s chars (median: %s)",
                    min_len,
                    max_len,
                    median_len,
                )

            log.info("🚀 STARTING BATCH PROCESSING")
            log.info("Batch size: %s", self.batch_size)

            # Process all items in batches
            with smart_open(
                self.output_file,
                "w",
                encoding="utf-8",
                transport_params=get_transport_params(self.output_file),
            ) as output_stream:
                current_batch = []
                batch_count = 0

                for i, item in enumerate(all_items):
                    current_batch.append(item)

                    if i % 1000 == 0 and i > 0:
                        log.debug(
                            "Processed %s/%s items (%s batches so far)",
                            i,
                            len(all_items),
                            batch_count,
                        )

                    # Process when batch is full
                    if len(current_batch) >= self.batch_size:
                        batch_count += 1
                        log.debug(
                            "Batch %s: %s texts",
                            batch_count,
                            len(current_batch),
                        )
                        process_batch(current_batch)
                        current_batch = []

                # Process final batch if any items remain
                if current_batch:
                    batch_count += 1
                    log.info(
                        "🏁 FINAL BATCH: batch %s with %s texts",
                        batch_count,
                        len(current_batch),
                    )
                    process_batch(current_batch)

                log.info(
                    "Completed processing %s items in %s batches",
                    len(all_items),
                    batch_count,
                )

        except Exception as e:
            log.error("Error processing file: %s", e, exc_info=True)
            sys.exit(1)

        # Calculate final timing metrics
        end_time = time.time()
        total_duration = end_time - start_time
        items_per_second = (
            len(all_items) / total_duration if total_duration > 0 else 0
        )

        final_msg = "Processing complete: %s lines total, %s processed, %s skipped" % (
            line_count,
            processed_count,
            skipped_count,
        )
        log.info(final_msg)

        # Report entity type statistics
        if entity_type_counts:
            log.info("📈 ENTITY TYPE SUMMARY:")
            total_entities = sum(entity_type_counts.values())
            log.info("   Total entities found: %s", total_entities)

            # Sort entity types by count (descending)
            sorted_entities = sorted(
                entity_type_counts.items(), key=lambda x: x[1], reverse=True
            )
            for entity_type, count in sorted_entities:
                percentage = (count / total_entities) * 100 if total_entities > 0 else 0
                log.info("   %s: %s entities (%.1f%%)", entity_type, count, percentage)
        else:
            log.info("📈 ENTITY TYPE SUMMARY: No entities found")

        log.info("📊 PERFORMANCE SUMMARY:")
        log.info("   Total processing time: %.2f seconds", total_duration)
        log.info("   Total items processed: %s", len(all_items))
        log.info("   Overall throughput: %.1f items/second", items_per_second)

    def run(self) -> None:
        """
        Dispatch to the selected engine.
        """
        if self.engine == "datasets":
            if not HAS_DATASETS:
                log.warning("Engine 'datasets' selected but datasets is not installed. Falling back to 'legacy'.")
                return self._run_legacy()
            return self._run_with_datasets()
        return self._run_legacy()


def main(args: Optional[List[str]] = None) -> None:
    """
    Main function to run the NewsAgencyProcessorV2.

    Args:
        args: Command-line arguments (uses sys.argv if None)
    """
    options: argparse.Namespace = parse_arguments(args)

    processor: NewsAgencyProcessorV2 = NewsAgencyProcessorV2(
        input_file=options.input,
        output_file=options.output,
        batch_size=options.batch_size,
        min_relevance=options.min_relevance,
        log_level=options.log_level,
        log_file=options.log_file,
        engine=options.engine,  # NEW
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
