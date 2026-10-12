"""CSV reading, validation, type inference and profiling (deterministic, no LLM).

CSV files are untrusted input. They are read in Python with the standard `csv` module, streamed in
batches (bounded memory), and inserted with parameterized statements, so SQL never touches the file
system and no file content is ever interpreted as SQL. Identifiers (table and column names) are
normalized, validated against a strict pattern and always double-quoted.

Reports describe structure only (types, counts, percentages, flags). They never contain cell values,
so they are safe to print, log, store in metadata or show to an agent.

Type inference rules (applied to the non-missing values of a column, in this order):
  1. no non-missing values                         -> VARCHAR  (warning: column is empty)
  2. every value is an integer, some have a leading zero (e.g. '00123') -> VARCHAR (codes keep zeros)
  3. every value is a plain integer (optional sign, <= 18 digits)      -> BIGINT
  4. every value is a plain decimal number (optional exponent)        -> DOUBLE
  5. every value is an ISO date YYYY-MM-DD         -> DATE
  6. every value is true/false (any case)          -> BOOLEAN
  7. otherwise                                     -> VARCHAR  (warning if >= 50% looked numeric or date-like:
                                                                 the column has mixed / ambiguous values)
Missing values are empty fields plus any extra `null_values` the caller lists (none by default).
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import duckdb

ALLOWED_EXTENSIONS = {".csv": ",", ".tsv": "\t"}
DEFAULT_MAX_BYTES = 200 * 1024 * 1024  # override with CB_INGEST_MAX_BYTES
MAX_COLUMNS = 500
MAX_REPORTED_ERRORS = 20
BATCH_ROWS = 5000
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")  # table names (and dataset names, see registry)
COLUMN_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
TYPES = ("BIGINT", "DOUBLE", "DATE", "BOOLEAN", "VARCHAR")
_ID_NAME = re.compile(r"(^|_)(id|key|identifier|mrn)$")
_PATIENT_NAME = re.compile(r"(person|patient|subject|member|mrn)")
_DATE_NAME = re.compile(r"(date|_dt$|_dt_|^dt_|_time$|timestamp)")
_INT_RE = "[+-]?[0-9]{1,18}"  # fits BIGINT without overflow
_NUM_RE = "[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]{1,3})?"
_CODE_NAME = re.compile(r"(code|concept|icd|ndc|loinc|cpt|hcpcs|snomed|rxnorm|atc|vocabulary)")


class IngestError(ValueError):
    """Ingestion refused. `problems` are structured {code, message, line?} entries without cell values."""

    def __init__(self, message: str, problems: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.problems = problems or [{"code": "error", "message": message}]


def max_bytes_from_env() -> int:
    raw = os.environ.get("CB_INGEST_MAX_BYTES")
    return int(raw) if raw and raw.strip().isdigit() else DEFAULT_MAX_BYTES


def qi(name: str) -> str:
    """Quote an identifier that has already been validated (defense in depth: refuses quotes)."""
    if '"' in name or "\x00" in name:
        raise IngestError(f"unsafe identifier {name!r}")
    return f'"{name}"'


def validate_name(name: str, what: str = "table name") -> str:
    if not NAME_PATTERN.match(name or "") or name.startswith("_") or "__" in name:
        raise IngestError(
            f"invalid {what} {name!r}: use lower-case letters, digits and single underscores, "
            "starting with a letter (max 48 characters)",
            [{"code": "invalid_name", "message": f"invalid {what}"}],
        )
    return name


def normalize_column(raw: str) -> str:
    """Header -> SQL-safe column name: lower case, non-alphanumerics -> '_', no leading digit."""
    name = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if name and name[0].isdigit():
        name = f"c_{name}"
    return name


def default_table_name(path: Path) -> str:
    stem = normalize_column(path.stem)
    return stem if stem and stem[0].isalpha() else f"t_{stem}" if stem else "table"


# ---- file checks -------------------------------------------------------------------------------
@dataclass
class FileInfo:
    name: str  # basename only (never a full path in reports)
    delimiter: str
    size_bytes: int
    sha256: str


def check_file(path: Path, max_bytes: int | None = None) -> FileInfo:
    """Existence, regular file, extension, size and strict UTF-8 (with line number of the first bad line)."""
    limit = max_bytes or max_bytes_from_env()
    if not path.exists():
        raise IngestError(f"file not found: {path.name}", [{"code": "not_found", "message": "file not found"}])
    if not path.is_file():
        raise IngestError(f"not a regular file: {path.name}", [{"code": "not_a_file", "message": "not a file"}])
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError(
            f"unsupported file type {ext or '(none)'!r}; expected one of {sorted(ALLOWED_EXTENSIONS)}",
            [{"code": "unsupported_extension", "message": f"unsupported extension {ext!r}"}],
        )
    size = path.stat().st_size
    if size == 0:
        raise IngestError(f"{path.name} is empty", [{"code": "empty_file", "message": "file is empty"}])
    if size > limit:
        raise IngestError(
            f"{path.name} is {size} bytes; the limit is {limit} (CB_INGEST_MAX_BYTES)",
            [{"code": "too_large", "message": "file exceeds the size limit"}],
        )
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for i, line in enumerate(fh, start=1):
            digest.update(line)
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                raise IngestError(
                    f"{path.name}: line {i} is not valid UTF-8 (re-save the file as UTF-8)",
                    [{"code": "invalid_encoding", "message": "not valid UTF-8", "line": i}],
                ) from None
    return FileInfo(path.name, ALLOWED_EXTENSIONS[ext], size, "sha256:" + digest.hexdigest())


def check_text(name: str, text: str, max_bytes: int) -> FileInfo:
    """Same checks for uploaded content (API): basename, extension and size only (text is already decoded)."""
    base = Path(name).name
    if base != name or not base or base.startswith("."):
        raise IngestError(
            "invalid file name: give a plain file name, not a path",
            [{"code": "invalid_file_name", "message": "file names must not contain paths"}],
        )
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError(
            f"unsupported file type {ext or '(none)'!r}; expected one of {sorted(ALLOWED_EXTENSIONS)}",
            [{"code": "unsupported_extension", "message": f"unsupported extension {ext!r}"}],
        )
    data = text.encode("utf-8")
    if not data:
        raise IngestError(f"{base} is empty", [{"code": "empty_file", "message": "file is empty"}])
    if len(data) > max_bytes:
        raise IngestError(
            f"{base} is {len(data)} bytes; the limit is {max_bytes}",
            [{"code": "too_large", "message": "file exceeds the size limit"}],
        )
    return FileInfo(base, ALLOWED_EXTENSIONS[ext], len(data), "sha256:" + hashlib.sha256(data).hexdigest())


# ---- header + rows -------------------------------------------------------------------------------
@dataclass
class Header:
    source: list[str]
    columns: list[str]


def parse_header(raw: list[str]) -> Header:
    problems: list[dict[str, Any]] = []
    if not raw or all(not h.strip() for h in raw):
        raise IngestError("missing header row", [{"code": "no_header", "message": "first line is empty", "line": 1}])
    if len(raw) > MAX_COLUMNS:
        raise IngestError(
            f"too many columns ({len(raw)} > {MAX_COLUMNS})",
            [{"code": "too_many_columns", "message": "too many columns"}],
        )
    cols = [normalize_column(h) for h in raw]
    for i, (src, col) in enumerate(zip(raw, cols, strict=True), start=1):
        if not src.strip():
            problems.append({"code": "empty_column_name", "message": f"column {i} has no name", "line": 1})
        elif not COLUMN_PATTERN.match(col):
            problems.append({"code": "invalid_column_name", "message": f"column {i} name cannot be used", "line": 1})
    seen: dict[str, int] = {}
    for i, col in enumerate(cols, start=1):
        if col and col in seen:
            problems.append(
                {
                    "code": "duplicate_column",
                    "line": 1,
                    "message": f"columns {seen[col]} and {i} have the same name {col!r} (after normalization)",
                }
            )
        seen.setdefault(col, i)
    if problems:
        raise IngestError("invalid header: " + "; ".join(p["message"] for p in problems), problems)
    return Header(raw, cols)


def iter_rows(stream: TextIO, delimiter: str, null_values: Sequence[str]) -> Iterator[tuple[Header, list[list[Any]]]]:
    """Yield (header, batch) pairs. Raises IngestError listing malformed rows (line numbers, no values);
    nothing is yielded after the first problem is known to be fatal, and callers roll back."""
    reader = csv.reader(stream, delimiter=delimiter, strict=True)
    nulls = set(null_values) | {""}
    try:
        raw_header = next(reader)
    except StopIteration:
        raise IngestError("file is empty", [{"code": "empty_file", "message": "file is empty"}]) from None
    except csv.Error as exc:
        raise IngestError(f"malformed header: {exc}", [{"code": "malformed", "message": str(exc), "line": 1}]) from None
    header = parse_header(raw_header)
    width = len(header.columns)
    batch: list[list[Any]] = []
    problems: list[dict[str, Any]] = []
    bad = 0
    data_rows = 0
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:  # unterminated quote etc.: the rest of the file cannot be trusted
            problems.append({"code": "malformed_row", "line": reader.line_num, "message": f"unparseable CSV ({exc})"})
            bad += 1
            break
        if not row:  # blank line
            continue
        if len(row) != width:
            bad += 1
            if len(problems) < MAX_REPORTED_ERRORS:
                problems.append(
                    {
                        "code": "malformed_row",
                        "line": reader.line_num,
                        "message": f"expected {width} fields, found {len(row)}",
                    }
                )
            continue
        data_rows += 1
        if not bad:
            batch.append([None if v in nulls else v for v in row])
            if len(batch) >= BATCH_ROWS:
                yield header, batch
                batch = []
    if bad:
        raise IngestError(
            f"{bad} malformed row(s); nothing was loaded (first problems at lines "
            f"{', '.join(str(p['line']) for p in problems[:5])})",
            problems,
        )
    if data_rows == 0:
        raise IngestError(
            "the file has a header but no data rows", [{"code": "no_data_rows", "message": "header only, no data rows"}]
        )
    yield header, batch


# ---- staging + inference -----------------------------------------------------------------------
@dataclass
class ColumnProfile:
    name: str
    source_name: str
    type: str
    non_null: int
    null_count: int
    null_pct: float
    distinct_count: int
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class TableProfile:
    table: str
    file: str
    sha256: str
    size_bytes: int
    rows: int
    columns: list[ColumnProfile]
    duplicate_rows: int
    candidate_keys: list[str]
    warnings: list[str]
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "file": self.file,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "duplicate_rows": self.duplicate_rows,
            "candidate_keys": self.candidate_keys,
            "columns": [c.__dict__ for c in self.columns],
            "warnings": self.warnings,
            "errors": self.errors,
            "suggestions": {
                "patient_identifier": [c.name for c in self.columns if "patient_identifier" in c.flags],
                "date": [c.name for c in self.columns if "date" in c.flags],
                "code": [c.name for c in self.columns if "code" in c.flags],
                "note": "suggestions only: mappings are never inferred automatically",
            },
        }


def _infer(con: duckdb.DuckDBPyConnection, raw: str, col: str) -> tuple[str, list[str]]:
    c = qi(col)
    # strict text patterns first: DuckDB's own casts accept '7.3' as BIGINT 7, '0x1F', '1_000', ...
    n, ints, dbls, dates, bools, zeros = con.execute(
        f"SELECT count({c}), "
        f"count(*) FILTER (WHERE regexp_full_match({c}, '{_INT_RE}')), "
        f"count(*) FILTER (WHERE regexp_full_match({c}, '{_NUM_RE}')), "
        f"count(*) FILTER (WHERE regexp_full_match({c}, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}') "
        f"AND TRY_CAST({c} AS DATE) IS NOT NULL), "
        f"count(*) FILTER (WHERE lower({c}) IN ('true', 'false')), "
        f"count(*) FILTER (WHERE regexp_full_match({c}, '[+-]?0[0-9]+')) FROM {raw}"
    ).fetchone() or (0, 0, 0, 0, 0, 0)
    notes: list[str] = []
    if n == 0:
        return "VARCHAR", ["column has no values"]
    if ints == n and zeros:
        return "VARCHAR", ["integer-like values with leading zeros kept as text (codes)"]
    if ints == n:
        return "BIGINT", notes
    if dbls == n:
        return "DOUBLE", notes
    if dates == n:
        return "DATE", notes
    if bools == n:
        return "BOOLEAN", notes
    for label, k in (("numeric", dbls), ("ISO date", dates)):
        if k >= n / 2:
            notes.append(f"mixed values: {k} of {n} look {label}; kept as text (fix the source or map explicitly)")
    return "VARCHAR", notes


def stage_table(
    con: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    stream: TextIO,
    info: FileInfo,
    null_values: Sequence[str] = (),
) -> TableProfile:
    """Load one CSV into `schema.table` (typed) and profile it. Runs inside the caller's transaction."""
    validate_name(table)
    raw = f"{qi(schema)}.{qi('raw_' + table)}"
    created = False
    header: Header | None = None
    for header, batch in iter_rows(stream, info.delimiter, null_values):
        if not created:
            con.execute(f"CREATE TABLE {raw} ({', '.join(qi(c) + ' VARCHAR' for c in header.columns)})")
            created = True
        if batch:
            con.executemany(f"INSERT INTO {raw} VALUES ({', '.join('?' for _ in header.columns)})", batch)
    assert header is not None  # iter_rows always yields at least once or raises
    types: dict[str, tuple[str, list[str]]] = {c: _infer(con, raw, c) for c in header.columns}

    def cast(c: str) -> str:
        t = types[c][0]
        if t == "VARCHAR":
            return f"{qi(c)}"
        if t == "BOOLEAN":
            return f"CAST(lower({qi(c)}) AS BOOLEAN) AS {qi(c)}"
        return f"CAST({qi(c)} AS {t}) AS {qi(c)}"

    target = f"{qi(schema)}.{qi(table)}"
    con.execute(f"CREATE TABLE {target} AS SELECT {', '.join(cast(c) for c in header.columns)} FROM {raw}")
    con.execute(f"DROP TABLE {raw}")
    return profile_table(con, schema, table, header, types, info)


def profile_table(
    con: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    header: Header,
    types: dict[str, tuple[str, list[str]]],
    info: FileInfo,
) -> TableProfile:
    target = f"{qi(schema)}.{qi(table)}"
    rows = int((con.execute(f"SELECT count(*) FROM {target}").fetchone() or (0,))[0])
    distinct_rows = int((con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {target})").fetchone() or (0,))[0])
    warnings: list[str] = []
    columns: list[ColumnProfile] = []
    for src, col in zip(header.source, header.columns, strict=True):
        non_null, distinct = con.execute(
            f"SELECT count({qi(col)}), count(DISTINCT {qi(col)}) FROM {target}"
        ).fetchone() or (0, 0)
        typ, notes = types[col]
        flags = []
        if _ID_NAME.search(col) or _PATIENT_NAME.search(col):
            flags.append("identifier")
            if _PATIENT_NAME.search(col):
                flags.append("patient_identifier")
        if typ == "DATE" or _DATE_NAME.search(col):
            flags.append("date")
            if typ != "DATE":
                notes.append("name suggests a date but values are not ISO YYYY-MM-DD (use date_format in the mapping)")
        if _CODE_NAME.search(col):
            flags.append("code")
        if rows and non_null == rows and distinct == rows:
            flags.append("unique")
        cp = ColumnProfile(
            col,
            src,
            typ,
            int(non_null),
            rows - int(non_null),
            round(100.0 * (rows - non_null) / rows, 2) if rows else 0.0,
            int(distinct),
            flags,
            notes,
        )
        columns.append(cp)
        if "identifier" in flags and "patient_identifier" not in flags and col.endswith("_id") and distinct < non_null:
            warnings.append(f"candidate key {col!r} has {non_null - distinct} duplicate value(s)")
        warnings += [f"{col}: {n}" for n in notes]
    if rows - distinct_rows:
        warnings.append(f"{rows - distinct_rows} duplicate record(s) (identical rows)")
    keys = [c.name for c in columns if "unique" in c.flags]
    return TableProfile(
        table, info.name, info.sha256, info.size_bytes, rows, columns, rows - distinct_rows, keys, warnings
    )


def profile_csv(
    path: Path, table: str | None = None, null_values: Sequence[str] = (), max_bytes: int | None = None
) -> dict[str, Any]:
    """Profile a CSV without touching any persistent database (in-memory DuckDB)."""
    path = Path(path)
    name = table or default_table_name(path)
    try:
        info = check_file(path, max_bytes)
        con = duckdb.connect(":memory:")
        con.execute("CREATE SCHEMA preview")
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            prof = stage_table(con, "preview", validate_name(name), fh, info, null_values)
        con.close()
        return {"ok": True, **prof.as_dict()}
    except IngestError as exc:
        return {"ok": False, "table": name, "file": path.name, "errors": exc.problems}


def profile_text(
    name: str, text: str, table: str | None = None, null_values: Sequence[str] = (), max_bytes: int | None = None
) -> dict[str, Any]:
    """Profile uploaded CSV content (API) in an in-memory database."""
    table_name = table or default_table_name(Path(Path(name).name))
    try:
        info = check_text(name, text, max_bytes or max_bytes_from_env())
        con = duckdb.connect(":memory:")
        con.execute("CREATE SCHEMA preview")
        prof = stage_table(con, "preview", validate_name(table_name), text_stream(text), info, null_values)
        con.close()
        return {"ok": True, **prof.as_dict()}
    except IngestError as exc:
        return {"ok": False, "table": table_name, "file": Path(name).name, "errors": exc.problems}


def open_text(path: Path) -> TextIO:
    return path.open("r", encoding="utf-8-sig", newline="")


def text_stream(text: str) -> TextIO:
    return io.StringIO(text.removeprefix("﻿"), newline="")
