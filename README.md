# file_indexer

A high-performance Rust CLI tool to recursively index files, extract metadata, compute checksums, and search contents.

## Features

- Fast recursive file indexing with metadata
- Optional SHA256 checksums for duplicate detection
- Content previews (text files)
- JSON and CSV export support
- Search interface with regex/wildcard matching

## Usage

### Index a directory

```bash
file_indexer index /some/path --depth 3 --content --checksums
```

### Search in indexed files

```bash
file_indexer search "main.rs" --regex
```

### Export

```bash
file_indexer export json output.json
```

### Check for duplicates

```bash
file_indexer duplicates
```

---

## Build

```bash
cargo build --release
```