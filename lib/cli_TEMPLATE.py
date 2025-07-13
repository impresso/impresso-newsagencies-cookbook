#!/usr/bin/env python3
"""
Dummy CLI Script Template with smart_open and logging

This module serves as a template for creating command-line interface scripts in the
Impresso project ecosystem. It demonstrates best practices for:

1. **File I/O Operations**: Uses smart_open for seamless handling of both local files
   and S3 URIs, with automatic transport parameter configuration via get_transport_params().

2. **Logging Configuration**: Integrates with impresso_cookbook's setup_logging function
   to provide consistent logging across all project tools, supporting both console and
   file output (including S3 destinations).

3. **S3 Integration**: Automatically configures S3 client connections using environment
   variables (SE_ACCESS_KEY, SE_SECRET_KEY, SE_HOST_URL) through get_s3_client().

4. **Command-line Interface**: Follows argparse patterns with proper type hints and
   documentation, including standard options for log level and log file output.

5. **Error Handling**: Implements robust error handling with proper logging of failures
   during file processing operations.

6. **Code Organization**: Separates concerns with a processor class that encapsulates
   the main logic, making the code testable and maintainable.

Template Usage:
- Copy this file as a starting point for new CLI tools
- Modify the TemplateProcessor class to implement your specific logic
- Add command-line arguments as needed in parse_arguments()
- Update the module docstring and class documentation

Template Philosophy:
This template embodies the principle of "convention over configuration" by providing
a standardized structure that reduces boilerplate and ensures consistency across
Impresso project tools. Key philosophical principles include:

**Separation of Concerns**: The processor class isolates business logic from CLI
infrastructure, making code easier to test, maintain, and reason about.

**Fail-Fast Principle**: Configuration and setup happen early in the process,
with comprehensive error handling and logging to quickly identify issues.

**Unified I/O Model**: By abstracting file operations through smart_open and
get_transport_params(), the same code seamlessly handles local files, S3 objects,
and stdin/stdout without modification.

**Observability by Default**: Comprehensive logging is built-in rather than
added as an afterthought, enabling better debugging and monitoring in production.

**Environment-Aware**: S3 credentials and endpoints are configured through
environment variables, supporting different deployment environments without
code changes.

**Extensibility**: The template provides a solid foundation that can be extended
with additional features while maintaining the core architectural patterns.

Integration with impresso_cookbook:
- Uses get_s3_client() for S3 operations
- Uses get_timestamp() for consistent timestamping
- Uses read_json() for JSON file reading with S3 support
- Uses setup_logging() for standardized logging configuration
- Uses get_transport_params() for automatic S3/local file handling

Example:
    $ python cli.py -i input.txt --log-level DEBUG --log-file process.log
    $ python cli.py -i s3://bucket/input.jsonl --log-level INFO
"""

import logging
import argparse
import json
import re
import sys
from smart_open import open as smart_open  # type: ignore
from typing import List, Optional

from impresso_cookbook import (  # type: ignore
    get_s3_client,
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
        description="Template CLI script with smart_open and logging."
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
    return parser.parse_args(args)


class TemplateProcessor:
    """
    A template processor class that reads input lines and logs them.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initializes the TemplateProcessor with explicit parameters.

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file
            log_level (str): Logging level (default: "INFO")
            log_file (Optional[str]): Path to log file (default: None)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.log_level = log_level
        self.log_file = log_file

        # Configure the module-specific logger
        setup_logging(self.log_level, self.log_file, logger=log)

        # Initialize S3 client and timestamp
        self.s3_client = get_s3_client()
        self.timestamp = get_timestamp()

    def run(self) -> None:
        """
        Runs the template processor, reading from the input file
        and processing each line.
        """
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
                for line in f:
                    result = self.process_line(line)
                    output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Error processing file: {e}", exc_info=True)
            sys.exit(1)

    def process_line(self, line: str) -> dict:
        """
        Processes a single line of JSON input data.

        Args:
            line (str): The input line to process.

        Returns:
            dict: The processed result.
        """
        data = json.loads(line.strip())
        uid = data.get("id", data.get("c_id", ""))
        fulltext = data.get("ft", "")
        length = len(fulltext)
        alphabetic_ratio = (
            len(re.sub(r"\W+", "", fulltext)) / length if length > 0 else 0
        )
        return {
            "id": uid,
            "length": length,
            "alphabetic_ratio": round(alphabetic_ratio, 3),
            "ts": self.timestamp,
        }


def main(args: Optional[List[str]] = None) -> None:
    """
    Main function to run the Template Processor.

    Args:
        args: Command-line arguments (uses sys.argv if None)
    """
    options: argparse.Namespace = parse_arguments(args)

    processor: TemplateProcessor = TemplateProcessor(
        input_file=options.input,
        output_file=options.output,
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
        log.error(f"Processing error: {e}", exc_info=True)
        sys.exit(2)
