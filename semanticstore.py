import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Tokens that commonly appear when XLSX zip bytes are accidentally parsed as JSON text.
_BINARY_TOKENS = (
    "pk\x03\x04",
    "[content_types].xml",
    "content_typesxml",
    "_rels/.rels",
    "rels/rels",
    "docprops",
    "sharedstrings",
    "stylesxml",
    "xl/workbook",
    "xl/worksheets",
)


# ---------------------------------------------------------
# NORMALIZE KEYS
# ---------------------------------------------------------
def normalize_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if not isinstance(row, dict):
        return normalized

    for raw_key, value in row.items():
        key = str(raw_key or "").replace("\x00", " ").strip().lower()
        key = re.sub(r"\s+", " ", key)
        key = re.sub(r"[^a-z0-9 _\-/().]", "", key).strip()
        key = key.replace(" ", "_")
        key = re.sub(r"_+", "_", key).strip("._-/")

        if not key:
            continue
        if _looks_binary_like(key):
            continue
        if not _is_valid_field_name(key):
            continue

        normalized[key] = value

    return normalized


def _is_valid_field_name(key: str) -> bool:
    """Keep realistic dynamic fields and drop gibberish key names."""
    if not key:
        return False
    if len(key) > 80:
        return False
    if not re.search(r"[a-z]", key):
        return False

    separators = sum(1 for ch in key if ch in "_-/().")
    digits = sum(1 for ch in key if ch.isdigit())
    sep_ratio = separators / max(len(key), 1)
    digit_ratio = digits / max(len(key), 1)

    if sep_ratio > 0.35:
        return False
    if len(key) > 20 and digit_ratio > 0.40:
        return False
    if re.search(r"_{3,}", key):
        return False
    return True


def _looks_binary_like(text: str) -> bool:
    if text is None:
        return False

    s = str(text).strip()
    if not s:
        return False

    lower = s.lower()
    compact = re.sub(r"[^a-z0-9/_\-.]", "", lower)

    if any(token in lower or token in compact for token in _BINARY_TOKENS):
        return True
    if compact.startswith("pk") and ("content" in compact or "rels" in compact):
        return True

    sample = s[:250]
    printable_ratio = sum(1 for ch in sample if ch.isprintable()) / max(len(sample), 1)
    alnum_ratio = sum(1 for ch in sample if ch.isalnum()) / max(len(sample), 1)
    symbol_ratio = sum(1 for ch in sample if not ch.isalnum()) / max(len(sample), 1)

    if printable_ratio < 0.85:
        return True
    if len(sample) > 25 and alnum_ratio < 0.25:
        return True
    if len(sample) > 30 and symbol_ratio > 0.60:
        return True
    return False


# ---------------------------------------------------------
# AUTO TYPE CONVERSION
# ---------------------------------------------------------
def auto_convert(value: Any):
    try:
        if value is None:
            return None

        if isinstance(value, (dict, list, tuple, set)):
            value = str(value)

        value = str(value).replace("\x00", " ").strip()
        if not value:
            return None
        if _looks_binary_like(value):
            return None

        numeric_candidate = re.sub(r"[,\$\u20B9\u20AC\u00A3%]", "", value).strip()
        if re.fullmatch(r"-?\d+\.\d+", numeric_candidate):
            return float(numeric_candidate)
        if re.fullmatch(r"-?\d+", numeric_candidate):
            return int(numeric_candidate)

        return value

    except Exception:
        return value


# ---------------------------------------------------------
# SPLIT NUMERIC / CATEGORICAL
# ---------------------------------------------------------
def split_fields(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    numeric: Dict[str, Any] = {}
    categorical: Dict[str, Any] = {}

    for k, v in row.items():
        if isinstance(v, (int, float)):
            numeric[k] = v
        else:
            categorical[k] = v

    return numeric, categorical


# ---------------------------------------------------------
# CREATE HUMAN FRIENDLY TEXT (FOR EMBEDDING)
# ---------------------------------------------------------
def create_semantic_text(row: Dict[str, Any]) -> str:
    if "month" in row:
        base = f"For {row['month']}, "
    else:
        base = "Data point: "

    parts = []
    for k, v in row.items():
        if k != "month":
            parts.append(f"{k.replace('_', ' ')} is {v}")

    return base + ", ".join(parts) + "."


def _prepare_rows_and_key_set(data: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, Dict[str, Any]]], set]:
    """
    Build normalized rows and keep only stable keys seen repeatedly.
    This prevents random ZIP-like garbage keys from entering Mongo documents.
    """
    normalized_rows: List[Tuple[int, Dict[str, Any]]] = []
    key_frequency: Dict[str, int] = {}

    for idx, raw_row in enumerate(data):
        if not isinstance(raw_row, dict):
            continue

        row = normalize_keys(raw_row)
        if not row:
            continue

        normalized_rows.append((idx, row))
        for key in row:
            key_frequency[key] = key_frequency.get(key, 0) + 1

    if not normalized_rows:
        raise ValueError(
            "No valid structured rows detected. Uploaded data looks corrupted or not row-wise JSON."
        )

    min_key_frequency = 1 if len(normalized_rows) < 5 else 2
    valid_keys = [
        key for key, count in key_frequency.items()
        if count >= min_key_frequency and _is_valid_field_name(key)
    ]
    valid_keys.sort(key=lambda k: (-key_frequency[k], k))
    valid_keys = valid_keys[:300]
    valid_key_set = set(valid_keys)

    if not valid_key_set:
        raise ValueError(
            "No reliable fields found after sanitization. Uploaded content appears malformed."
        )

    return normalized_rows, valid_key_set


# ---------------------------------------------------------
# MAIN FUNCTION (UPDATED FOR JSON INPUT)
# ---------------------------------------------------------
def process_dataset(
    data: List[Dict[str, Any]],
    file_name: str,
    embedding_client,
    mongo_client
):
    try:
        logger.info(f"[PROCESS] Processing dataset: {file_name}")

        normalized_rows, valid_key_set = _prepare_rows_and_key_set(data)
        documents = []

        for idx, row in normalized_rows:
            try:
                structured_data = {}

                # Clean + convert values on valid/stable keys only.
                for k, v in row.items():
                    if k not in valid_key_set:
                        continue

                    converted_value = auto_convert(v)
                    if converted_value is None:
                        continue

                    structured_data[k] = converted_value

                if not structured_data:
                    logger.warning(f"[SKIP] Row {idx} has no valid values after conversion")
                    continue

                numeric_fields, categorical_fields = split_fields(structured_data)
                content = create_semantic_text(structured_data)
                embedding = embedding_client.generate_embedding(content)

                if not embedding:
                    logger.warning(f"[SKIP] Embedding failed at row {idx}")
                    continue

                document = {
                    "type": "embedding",
                    "file_name": file_name,
                    "row_index": idx,
                    "data": structured_data,
                    "content": content,
                    "metadata": {
                        "numeric": numeric_fields,
                        "categorical": categorical_fields,
                        "field_names": list(structured_data.keys()),
                    },
                    "embedding": embedding,
                }
                documents.append(document)

            except Exception as e:
                logger.error(f"[ROW_ERR] Row {idx} failed: {e}")
                continue

        if not documents:
            raise ValueError("No embeddings generated because all rows were filtered as invalid.")

        mongo_client.insert_documents(documents)
        logger.info(f"[PROCESS] {len(documents)} embeddings stored")

    except Exception as e:
        logger.error(f"[PROCESS_ERR] {e}")
        raise
