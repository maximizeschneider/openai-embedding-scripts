#!/usr/bin/env python
"""
Submit MS MARCO (or any id/text TSV) to the OpenAI *Batch API*.

For every ≤50 000‑row slice the script …

1. writes a JSONL request file (one line = one /v1/embeddings call),
2. uploads it (`purpose="batch"`),
3. launches a batch job (`endpoint="/embeddings"`),
4. stores a tiny mapping JSON:

   {
     "batch_id": "batch_abc123",
     "file_id":  "file_xyz789",
     "tsv_rows": [ "Q1", "Q2", … ],
     "slice":    0
   }

The mapping lives next to the request file and is consumed by
*ms_marco_batch_fetch.py*.

Usage example
-------------
python ms_marco_batch_submit.py \
  --input /Volumes/T7/Data/MSMarco/collection.tsv \
  --out-dir   /Volumes/T7/Data/MSMarco/batch-jobs
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Tuple

import dotenv, typer
from tqdm import tqdm
from openai import OpenAI
import tiktoken

app = typer.Typer(add_completion=False)

MODEL          = "text-embedding-3-large"
ENDPOINT       = "/v1/embeddings"   
MAX_PER_BATCH  = 50_000           # hard OpenAI limit
JSONL_TEMPLATE = '{{"model":"{m}","input":{t},"encoding_format":"float","custom_id":"{i}"}}'

# --------------------------------------------------------------------------- #
def load_tsv(tsv: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    with tsv.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                docs.append((parts[0], parts[1]))
    return docs

# --------------------------------------------------------------------------- #
@app.command()
def main(
    input:     Path = typer.Option(..., help="TSV with id <TAB> text"),
    out_dir:   Path = typer.Option(..., help="Directory to store JSONL + mapping"),
    slice_len: int  = typer.Option(MAX_PER_BATCH, help="Rows per batch job (≤50 000)"),
    window:    str  = typer.Option("24h", help='`completion_window` ("24h" only for now)'),
):
    if slice_len > MAX_PER_BATCH:
        typer.echo(f"slice_len capped to {MAX_PER_BATCH}")
        slice_len = MAX_PER_BATCH

    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    encoding = tiktoken.encoding_for_model(MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_tsv(input)
    docs = docs[:100000]
    typer.echo(f"Loaded {len(docs):,} rows – creating batch jobs …")

    for slice_idx, start in enumerate(range(0, len(docs), slice_len)):
        chunk          = docs[start:start + slice_len]
        request_path   = out_dir / f"req_{slice_idx:05d}.jsonl"
        mapping_path   = out_dir / f"req_{slice_idx:05d}.mapping.json"

        # --- write JSONL request file -------------------------------------- #
        with request_path.open("w") as jf:
            for doc_id, text in chunk:
                # --- Exact token check ---
                num_tokens = len(encoding.encode(text))
                if num_tokens > 8191:
                    typer.echo(f"⚠️  Skipping doc '{doc_id}': tokens ({num_tokens}) > 8191.", err=True)
                    continue # Skip this document

                line = JSONL_TEMPLATE.format(m=MODEL,
                                             t=json.dumps(text),
                                             i=doc_id)
                jf.write(line + "\n")

        # --- upload file & create batch job -------------------------------- #
        file_obj = client.files.create(file=request_path, purpose="batch")
        batch    = client.batches.create(
            input_file_id   = file_obj.id,
            endpoint        = ENDPOINT,
            completion_window = window
        )

        # --- save lightweight mapping -------------------------------------- #
        mapping = dict(batch_id=batch.id,
                       file_id=file_obj.id,
                       slice=slice_idx,
                       tsv_rows=[doc_id for doc_id, _ in chunk])
        mapping_path.write_text(json.dumps(mapping, indent=2))

        typer.echo(f"✓ queued batch {batch.id} ({len(chunk):,} docs)")

        # --- Delete the request file as it's no longer needed ----------- #
        try:
            request_path.unlink()
        except OSError as e:
            typer.echo(f"⚠️  Warning: Could not delete {request_path}: {e}", err=True)

    typer.echo("All slices submitted!")

if __name__ == "__main__":
    app()
