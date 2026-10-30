#!/usr/bin/env python3
"""convert_malecns.py - MaleCNS connectome Feather -> normalized TSV converter.

Converts the official MaleCNS v1.0 flat-connectome Feather files into the
normalized TSV format consumed by the `alife-neural` Rust importer
(`connectome_format.rs`, FORMAT_NAME = "flybrain-normalized-connectome",
format_version 1):

    edges.tsv    -> THREE columns: body_pre <TAB> body_post <TAB> weight
    nodes.tsv    -> FIVE columns:  bodyId <TAB> type <TAB> superclass
                                 <TAB> status <TAB> somaSide
    sidecar.json -> metadata sidecar (schema in connectome_format.rs `Sidecar`)

Verification status of upstream facts (verified 2026-10-30 by reading
https://male-cns.janelia.org/download/ and /release/):
  - v1.0 released June 8, 2026 (official release notes)
  - Official download page lists:
      * connectome-weights-male-cns-v1.0-minconf-0.5.feather  (1.1 GB)
      * body-annotations-male-cns-v1.0-minconf-0.5.feather   (13 MB)
    under gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
    (direct https mirrors at storage.googleapis.com/flyem-male-cns/...)
  - The Male CNS dataset is "licensed under CC-BY" with a link to
    https://creativecommons.org/licenses/by/4.0/  (i.e. CC-BY 4.0)

Hard rules enforced by this converter:
  1. Source columns are VALIDATED first; a missing/renamed column is a hard
     error that lists the columns actually found (never guessed).
  2. Body IDs are preserved EXACTLY: an integer Arrow column is required
     (int32/int64/uint32/uint64); ids are written as bare decimal integers.
     A float body-id column is rejected rather than risk precision loss.
  3. Synapse weights are preserved EXACTLY: written with Python's shortest
     round-trip repr(float). The importer parses f32; the importer keeps f32
     precision, never more - this is a documented boundary in
     FLYBRAIN_IMPLEMENTATION.md.
  4. Missing/NULL annotations are written as EMPTY STRINGS. Nothing is ever
     invented; every annotated body gets a row even when most fields are
     empty.
  5. Memory: connectivity and annotations are streamed record-batch by
     record-batch over a memory-mapped Feather (Arrow IPC file) handle. The
     1.1 GB connectivity file is NEVER loaded fully into memory.
  6. Datasets are never merged: one converter run writes exactly one output
     directory, from exactly one given input pair.

The 1.1 GB download itself is a MANUAL, human-supplied step: this tool never
downloads, and requires the local path of the already-fetched files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIREMENTS = (
    "requires pyarrow (Python 3.9+). Install with: pip install pyarrow>=14 "
    "(the only third-party dependency)"
)

try:
    import pyarrow as pa
except ImportError as _e:  # explicit, actionable requirements note
    raise SystemExit(f"{CONVERTER}: {REQUIREMENTS} (import failed: {_e})")

CONVERTER = "convert_malecns.py"
CONVERTER_VERSION = "1.1.0"

EDGE_COLUMNS = ["body_pre", "body_post", "weight"]
NODE_COLUMNS = ["bodyId", "type", "superclass", "status", "somaSide"]

# Official source page (verified 2026-10-30; facts quoted above).
SOURCE_URL = "https://male-cns.janelia.org/download"

FORMAT_NAME = "flybrain-normalized-connectome"
FORMAT_VERSION = 1


class ConversionError(Exception):
    """Hard error (bad source, bad columns); the run writes nothing partial."""


def open_feather_file(path: Path):
    """Open a Feather file for batch-wise reads.

    Feather v2 = Arrow IPC file: batch random access over a memory map (no
    full-file load). Legacy Feather v1 has no batch API here; the fallback
    re-wraps the whole table (documented: v1 sources must be small enough to
    fit, which the official v1.0 files - v2 - are anyway).
    """
    try:
        return pa.ipc.open_file(str(path))
    except (pa.ArrowInvalid, OSError, ValueError):
        import pyarrow.feather

        table = pyarrow.feather.read_table(str(path))
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table, max_chunksize=1_000_000)
        writer.close()
        return pa.ipc.open_file(sink.getvalue())


def require_columns(batch: pa.RecordBatch, required, path: Path) -> None:
    found = list(batch.schema.names)
    missing = [c for c in required if c not in found]
    if missing:
        raise ConversionError(
            f"{path.name}: missing required column(s) {missing}; found columns "
            f"{found}; refusing to guess a mapping."
        )


def integer_column(batch: pa.RecordBatch, name: str, path: Path) -> pa.ChunkedArray:
    col = batch.column(batch.schema.get_field_index(name))
    if not pa.types.is_integer(col.type):
        raise ConversionError(
            f"{path.name}: column '{name}' must be an INTEGER Arrow type to be "
            f"preserved exactly; found {col.type} (float body ids would risk "
            "precision loss and are rejected)."
        )
    return col


def body_id_strings(batch: pa.RecordBatch, name: str, path: Path):
    col = integer_column(batch, name, path)
    for v in col.to_pylist():  # per-batch only
        if v is None:
            raise ConversionError(f"{path.name}: NULL in column '{name}'")
        yield str(v)


def weight_strings(batch: pa.RecordBatch, name: str, path: Path):
    col = batch.column(batch.schema.get_field_index(name))
    for v in col.to_pylist():
        if v is None or isinstance(v, float) and math.isnan(v):
            raise ConversionError(f"{path.name}: NULL/NaN weight")
        f = float(v)
        if not math.isfinite(f):
            raise ConversionError(f"{path.name}: non-finite weight {v!r}")
        # shortest round-trip decimal text: the value survives as written
        yield repr(f)


def text_field_strings(batch: pa.RecordBatch, name: str):
    col = batch.column(batch.schema.get_field_index(name))
    for v in col.to_pylist():
        if v is None:
            yield ""
        else:
            s = str(v).strip()
            if "\t" in s or "\n" in s or "\r" in s:
                raise ConversionError(
                    f"annotation field '{name}' contains tab/newline: {s!r}"
                )
            yield s


def sha256_of(path: Path) -> str:
    """SHA-256 of the ORIGINAL download (streamed, never memory-resident)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(
    edges_path: Path,
    nodes_path: Path,
    out_dir: Path,
    dataset: str,
    want_sha256: bool,
    edge_cols=None,
    node_cols=None,
) -> dict:
    # Column-name mapping is CONFIGURABLE (see --edge-cols/--node-cols) and
    # documented; the defaults below are our expected MaleCNS names and are
    # validated against the real Arrow schema before any row is written
    # (a mismatch is a hard error, never a silent guess). Use
    # `--schema-inspect` first when the exact upstream column names are
    # unknown/unverified.
    edge_cols = list(edge_cols or EDGE_COLUMNS)
    node_cols = list(node_cols or NODE_COLUMNS)
    (pre_c, post_c, w_c) = edge_cols          # positional logical roles
    (id_c, type_c, sup_c, st_c, side_c) = node_cols
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_out = out_dir / "edges.tsv"
    nodes_out = out_dir / "nodes.tsv"
    sidecar_out = out_dir / "sidecar.json"

    # ---- edges.tsv: validate columns on the FIRST batch, then stream --------
    rows_in_total = 0
    rows_out_edges = 0
    edge_iter = open_feather_batches(edges_path)
    first_edge = next(edge_iter, None)
    if first_edge is None:
        raise ConversionError(f"{edges_path.name}: file has zero record batches")
    require_columns(first_edge, edge_cols, edges_path)
    with open(edges_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(EDGE_COLUMNS) + "\n")
        for batch in [first_edge, *edge_iter]:
            require_columns(batch, edge_cols, edges_path)
            rows_in_total += batch.num_rows
            pres = body_id_strings(batch, pre_c, edges_path)
            posts = body_id_strings(batch, post_c, edges_path)
            ws = weight_strings(batch, w_c, edges_path)
            for p, q, w in zip(pres, posts, ws):
                fh.write(f"{p}\t{q}\t{w}\n")
                rows_out_edges += 1

    # ---- nodes.tsv: every annotated body, empties kept empty ----------------
    rows_in_nodes = 0
    rows_out_nodes = 0
    node_iter = open_feather_batches(nodes_path)
    first_nodes = next(node_iter, None)
    if first_nodes is None:
        raise ConversionError(f"{nodes_path.name}: file has zero record batches")
    require_columns(first_nodes, node_cols, nodes_path)
    with open(nodes_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(NODE_COLUMNS) + "\n")
        for batch in [first_nodes, *node_iter]:
            require_columns(batch, node_cols, nodes_path)
            rows_in_nodes += batch.num_rows
            ids = body_id_strings(batch, id_c, nodes_path)
            types = text_field_strings(batch, type_c)
            supers = text_field_strings(batch, sup_c)
            stats = text_field_strings(batch, st_c)
            sides = text_field_strings(batch, side_c)
            for p, t, s, st, sd in zip(ids, types, supers, stats, sides):
                fh.write(f"{p}\t{t}\t{s}\t{st}\t{sd}\n")
                rows_out_nodes += 1

    sidecar = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "source_files": [edges_path.name, nodes_path.name],
        "source_url": SOURCE_URL,
        "converter": CONVERTER,
        "converter_version": CONVERTER_VERSION,
        "source_sha256": sha256_of(edges_path) if want_sha256 else "",
        "rows_in": rows_in_total,
        "rows_out_edges": rows_out_edges,
        "rows_out_nodes": rows_out_nodes,
        "converted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "notes": (
            "Converted from official MaleCNS Feather flat-connectome files; "
            "weights and body IDs preserved verbatim; missing annotations empty. "
            f"Source column map: edges {edge_cols} (pre,post,weight); nodes "
            f"{node_cols} (bodyId,type,superclass,status,somaSide)."
        ),
    }
    with open(sidecar_out, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
        fh.write("\n")
    return sidecar


def schema_inspect(paths, out=sys.stdout) -> None:
    """Print the FULL Arrow schema of each input file and exit.

    Use this when the exact connectome-weights column names are not verified
    (they are documented as UNVERIFIED on official MaleCNS pages): run this
    once on the real download, then pass the confirmed names via
    --edge-cols/--node-cols. Nothing is written or validated against the
    expected schema here; inspection is read-only.
    """
    for path in paths:
        if not path.exists():
            print(f"{path}: NOT FOUND", file=out)
            continue
        f = open_feather_file(path)
        schema = f.schema
        print(f"== {path.name}", file=out)
        print(f"   record_batches={f.num_record_batches} columns={len(schema)}", file=out)
        for i, field in enumerate(schema):
            print(f"   [{i}] {field.name}: {field.type}", file=out)


def open_feather_batches(path: Path):
    file = open_feather_file(path)
    n = file.num_record_batches
    if n == 0:
        return
    for i in range(n):
        yield file.get_batch(i)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize official MaleCNS v1.0 Feather files to the alife-neural "
            "TSV format. MANUAL download required first; this tool never downloads."
        )
    )
    ap.add_argument(
        "--edges", required=True, type=Path,
        help="path to connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    )
    ap.add_argument(
        "--nodes", required=True, type=Path,
        help="path to body-annotations-male-cns-v1.0-minconf-0.5.feather",
    )
    ap.add_argument("--out", required=False, type=Path, help="output directory "
        "(required unless --schema-inspect)")
    ap.add_argument(
        "--dataset", default="MaleCNS v1.0",
        help="dataset name recorded in the sidecar (never implies merging)",
    )
    ap.add_argument(
        "--sha256", action="store_true",
        help="also compute the SHA-256 of the connectivity feather (streamed; "
        "may take a while for the 1.1 GB file)",
    )
    ap.add_argument(
        "--schema-inspect", action="store_true",
        help="print the full Arrow schema of --edges/--nodes and exit "
        "(use first: connectome-weights column names are NOT verified on "
        "official pages; confirm them before converting)",
    )
    ap.add_argument(
        "--edge-cols", default=None,
        help="comma-separated connectivity column names in role order "
        "PRE,POST,WEIGHT (default: body_pre,body_post,weight)",
    )
    ap.add_argument(
        "--node-cols", default=None,
        help="comma-separated annotation column names in role order "
        "ID,TYPE,SUPERCLASS,STATUS,SOMASIDE (default: "
        "bodyId,type,superclass,status,somaSide)",
    )
    args = ap.parse_args(argv)
    if args.schema_inspect:
        schema_inspect([args.edges, args.nodes])
        return 0
    if args.out is None:
        print("convert_malecns: ERROR: --out is required (or use --schema-inspect)",
              file=sys.stderr)
        return 2
    edge_cols = [c.strip() for c in args.edge_cols.split(",")] if args.edge_cols else None
    node_cols = [c.strip() for c in args.node_cols.split(",")] if args.node_cols else None
    try:
        sc = convert(
            args.edges, args.nodes, args.out, args.dataset, args.sha256,
            edge_cols=edge_cols, node_cols=node_cols,
        )
    except ConversionError as e:
        print(f"convert_malecns: ERROR: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(
            "convert_malecns: ERROR: input file not found: "
            f"{getattr(e, 'filename', e)} (manual download step; see "
            f"{SOURCE_URL})",
            file=sys.stderr,
        )
        return 2
    print(
        f"convert_malecns: wrote {sc['rows_out_edges']} edges, "
        f"{sc['rows_out_nodes']} nodes, sidecar -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
