"""
I/O Utilities Module for ElectriQ
Handles file operations, data serialization, and path management.
"""

import json
import csv
import yaml
import logging
from typing import List, Dict, Any, Optional, Union, Iterator, TypeVar
from pathlib import Path
from datetime import datetime
import hashlib
import gzip
import pickle
from contextlib import contextmanager

T = TypeVar('T')

logger = logging.getLogger(__name__)


class DataFormatError(Exception):
    """Custom exception for data format errors."""
    pass


class IOUtils:
    """Utility class for file I/O operations."""

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Path: The resolved path object
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {p}")
        return p

    @staticmethod
    def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        Calculate hash of a file.

        Args:
            filepath: Path to the file
            algorithm: Hash algorithm ('md5', 'sha256', etc.)

        Returns:
            str: Hexadecimal hash string
        """
        hash_func = getattr(hashlib, algorithm, None)
        if not hash_func:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hasher = hash_func()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    @contextmanager
    def safe_open(filepath: Union[str, Path], mode: str = 'r', encoding: str = 'utf-8', **kwargs):
        """
        Context manager for safe file opening with error handling.

        Args:
            filepath: File path
            mode: File mode ('r', 'w', 'a', etc.)
            encoding: File encoding
            **kwargs: Additional arguments for open()

        Yields:
            File object
        """
        f = None
        try:
            # Ensure directory exists for write modes
            if 'w' in mode or 'a' in mode:
                IOUtils.ensure_dir(Path(filepath).parent)

            f = open(filepath, mode, encoding=encoding, **kwargs)
            yield f
        except Exception as e:
            logger.error(f"Error accessing file {filepath}: {e}")
            raise
        finally:
            if f and not f.closed:
                f.close()

    # --- JSON Operations ---

    @staticmethod
    def load_json(filepath: Union[str, Path], encoding: str = 'utf-8') -> Any:
        """
        Load JSON from file.

        Args:
            filepath: Path to JSON file
            encoding: File encoding

        Returns:
            Any: Parsed JSON data
        """
        with IOUtils.safe_open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Any, filepath: Union[str, Path], indent: int = 2, encoding: str = 'utf-8', **kwargs) -> None:
        """
        Save data to JSON file.

        Args:
            data: Data to save
            filepath: Output path
            indent: JSON indentation level
            encoding: File encoding
            **kwargs: Additional arguments for json.dump
        """
        with IOUtils.safe_open(filepath, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, **kwargs)
        logger.info(f"Saved JSON to {filepath}")

    @staticmethod
    def load_jsonl(filepath: Union[str, Path], encoding: str = 'utf-8') -> List[Dict]:
        """
        Load JSONL (JSON Lines) file.

        Args:
            filepath: Path to JSONL file
            encoding: File encoding

        Returns:
            list: List of dictionaries
        """
        data = []
        with IOUtils.safe_open(filepath, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        return data

    @staticmethod
    def save_jsonl(data: List[Dict], filepath: Union[str, Path], encoding: str = 'utf-8') -> None:
        """
        Save list of dictionaries to JSONL file.

        Args:
            data: List of dictionaries
            filepath: Output path
            encoding: File encoding
        """
        with IOUtils.safe_open(filepath, 'w', encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved JSONL to {filepath} ({len(data)} items)")

    # --- CSV Operations ---

    @staticmethod
    def load_csv(filepath: Union[str, Path], delimiter: str = ',', **kwargs) -> List[Dict]:
        """
        Load CSV file into list of dictionaries.

        Args:
            filepath: Path to CSV file
            delimiter: Column delimiter
            **kwargs: Additional arguments for csv.DictReader

        Returns:
            list: List of row dictionaries
        """
        with IOUtils.safe_open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter, **kwargs)
            return list(reader)

    @staticmethod
    def save_csv(data: List[Dict], filepath: Union[str, Path], fieldnames: Optional[List[str]] = None,
                 **kwargs) -> None:
        """
        Save list of dictionaries to CSV file.

        Args:
            data: List of dictionaries
            filepath: Output path
            fieldnames: Column names (if None, inferred from first row)
            **kwargs: Additional arguments for csv.DictWriter
        """
        if not data:
            logger.warning("Attempting to save empty data to CSV")
            return

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        with IOUtils.safe_open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, **kwargs)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Saved CSV to {filepath} ({len(data)} rows)")

    # --- YAML Operations ---

    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Any:
        """Load YAML file."""
        with IOUtils.safe_open(filepath, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_yaml(data: Any, filepath: Union[str, Path]) -> None:
        """Save data to YAML file."""
        with IOUtils.safe_open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved YAML to {filepath}")

    # --- Compressed Files ---

    @staticmethod
    def save_json_gz(data: Any, filepath: Union[str, Path], **kwargs) -> None:
        """Save data to gzipped JSON file."""
        path = Path(filepath)
        if not path.suffix == '.gz':
            path = path.with_suffix(path.suffix + '.gz')

        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, **kwargs)
        logger.info(f"Saved gzipped JSON to {path}")

    @staticmethod
    def load_json_gz(filepath: Union[str, Path]) -> Any:
        """Load gzipped JSON file."""
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)

    # --- Pickle (Use with caution) ---

    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        """Save data to pickle file."""
        with IOUtils.safe_open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        logger.info(f"Saved pickle to {filepath}")

    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Any:
        """Load pickle file."""
        with IOUtils.safe_open(filepath, 'rb') as f:
            return pickle.load(f)

    # --- Batch Processing ---

    @staticmethod
    def batch_iterate(
            filepath: Union[str, Path],
            batch_size: int = 100,
            loader_func: callable = None
    ) -> Iterator[List[Any]]:
        """
        Iterate over a large file in batches.

        Args:
            filepath: Path to file
            batch_size: Number of items per batch
            loader_func: Custom loader function (default: auto-detect JSONL)

        Yields:
            list: Batch of items
        """
        if loader_func is None:
            # Default to JSONL line-by-line reading for memory efficiency
            def loader_func(fp):
                with open(fp, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)

        batch = []
        for item in loader_func(filepath):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


if __name__ == '__main__':
    # Example usage
    sample_data = [
        {"id": 1, "text": "Hello"},
        {"id": 2, "text": "World"}
    ]

    # Save/Load JSON
    IOUtils.save_json(sample_data, "output/test.json")
    loaded = IOUtils.load_json("output/test.json")
    print(f"JSON Loaded: {loaded}")

    # Save/Load JSONL
    IOUtils.save_jsonl(sample_data, "output/test.jsonl")
    loaded_list = IOUtils.load_jsonl("output/test.jsonl")
    print(f"JSONL Loaded: {loaded_list}")

    # Hash
    h = IOUtils.get_file_hash("output/test.json")
    print(f"File Hash: {h}")