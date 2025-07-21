# Changelog

## [1.0.0] - 2025-07-21

### Added

- **NER Model Integration**: Custom `NewsAgencyTokenClassifier` for newsagency recognition in historical newspaper articles
- **GPU Batch Processing**: Optimized batch processing with automatic text length sorting for efficient GPU memory utilization
- **Chunk-Aware Processing**: Automatic text chunking with overlap handling for long documents exceeding model limits
- **Adaptive Batching**: Dynamic batch sizing based on target chunk count (configurable via `--target-chunks`)
- **S3 Integration**: Direct processing from/to S3 URIs using smart_open with automatic transport parameter configuration
- **Wikidata Linking**: Automatic linking of recognized newsagencies to Wikidata entries
- **Entity Suppression**: Configurable suppression of unwanted entity types (e.g., unknown agencies, article authors)
- **Performance Monitoring**: Real-time tracking of processing speed (chunks/second) and memory usage
- **Distributed Processing**: Make-based build system supporting multi-machine processing without conflicts
- **Quality Assessment**: Confidence scoring and validation for NER results

### Features

- **Model**: `impresso-project/ner-newsagency-bert-multilingual` - Multilingual BERT model for newsagency NER
- **Chunking Strategy**: 512 tokens with 64-token stride overlap for seamless processing of long texts
- **Batch Optimization**: Length-based text sorting to minimize padding and maximize GPU efficiency
- **Memory Management**: Automatic GPU memory cleanup and CUDA cache emptying
- **Logging**: Comprehensive logging with configurable levels (DEBUG, INFO, WARNING, ERROR)
- **Output Formats**: Structured JSON output with entity details, relevance scores, and timestamps

### Performance

- **Adaptive Estimation**: Machine learning-based estimation of tokenization for optimal batch sizing
- **GPU Acceleration**: CUDA support with fallback to MPS (Apple Silicon) and CPU
- **Parallel Processing**: Multi-core CPU utilization through Make's parallel build system
- **Scalability**: Processes newspapers independently enabling horizontal scaling

### Configuration

- `--target-chunks`: Control GPU memory usage (default: 16 chunks per batch)
- `NEWSPAPER`: Target newspaper selection for processing
- `COLLECTION_JOBS`: Number of parallel newspaper processing jobs

### Infrastructure

- **Build System**: Sophisticated Makefile system with modular includes
- **Dependency Management**: Automatic resolution and stamp-file based tracking
- **Data Synchronization**: Bidirectional sync between local storage and S3
- **Error Handling**: Robust error recovery with detailed logging

### Requirements

- Python 3.11+
- PyTorch with CUDA support (optional)
- Transformers library
- GNU Make, Git with git-lfs
- AWS CLI (optional for direct S3 access)

---

_This release represents the collaborative effort of the Impresso project team, funded by the Swiss National Science Foundation (grants CRSII5_173719 and CRSII5_213585) and Luxembourg National Research Fund (grant 17498891)._
