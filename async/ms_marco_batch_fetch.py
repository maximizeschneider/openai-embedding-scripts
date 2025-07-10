"""
Poll every *.mapping.json created by ms_marco_batch_submit.py.
When a batch finishes, download its *output_file_id* and write an
NPZ shard holding `ids`, `texts`, `vectors`.

Usage example
-------------
python ms_marco_batch_fetch.py \
  --mappings /Volumes/T7/Data/MSMarco/batch-jobs \
  --input-tsv /Volumes/T7/Data/MSMarco/collection.tsv\
  --out-dir   /Volumes/T7/Data/MSMarco/batch-embeddings \
  --shard-size 10000
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Dict
from datetime import datetime

import dotenv
import numpy as np
import typer
from tqdm import tqdm
from openai import OpenAI, OpenAIError

VECTOR_DIMS = 3072        # text‑embedding‑3‑large
POLL_SECS   = 30          # polling interval

app = typer.Typer(add_completion=False)

class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int = 10_000):
        self.out_dir, self.shard_size = out_dir, shard_size
        self.reset()

    def reset(self):
        self.ids, self.texts, self.vecs = [], [], []
        self.idx = getattr(self, "idx", 0)

    def add(self, _id: str, text: str, vec: np.ndarray):
        self.ids.append(_id)
        self.texts.append(text)
        self.vecs.append(vec)
        if len(self.ids) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.ids:
            return
        fn = self.out_dir / f"emb_{self.idx:05d}.npz"
        np.savez_compressed(
            fn,
            ids=np.asarray(self.ids),
            texts=np.asarray(self.texts),
            vectors=np.vstack(self.vecs).astype(np.float32)
        )
        typer.echo(f"Wrote {fn}")
        self.idx += 1
        self.reset()

def load_tsv(tsv: Path) -> Dict[str, str]:
    id2text = {}
    with tsv.open(encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                id2text[parts[0]] = parts[1]
    return id2text

def parse_result_file(raw: str):
    """Yield (custom_id, embedding) tuples."""
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        if obj["response"]["status_code"] != 200:
            typer.echo(f"⚠️  Warning: Failed row: {obj['id']}")
            continue  # skip failed rows
            
        cid = obj.get("custom_id")
        vec = obj["response"]["body"]["data"][0]["embedding"]
        yield cid, np.asarray(vec, dtype=np.float32)

@app.command()
def main(
    mappings:   Path = typer.Option(..., help="Dir with *.mapping.json"),
    input_tsv:  Path = typer.Option(..., help="Original TSV (id ↦ text)"),
    out_dir:    Path = typer.Option(..., help="NPZ output folder"),
    shard_size: int  = typer.Option(10_000, help="Rows per NPZ shard"),
):
    dotenv.load_dotenv()
    client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    id2text    = load_tsv(input_tsv)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer     = ShardWriter(out_dir, shard_size)

    mapping_files = sorted(
        f for f in mappings.glob("*.mapping.json") if not f.name.startswith("._")
    )
    if not mapping_files:
        typer.echo("No mapping files found – nothing to do.")
        raise typer.Exit()

    pending = {mf: json.loads(mf.read_text()) for mf in mapping_files}
    pbar    = tqdm(total=len(pending), desc="Completed batches")

    # Track the API's completed timestamp
    last_completed_ts: int | None = None

    while pending:
        for mf, meta in list(pending.items()):
            try:
                batch_id = meta["batch_id"]
                job = client.batches.retrieve(batch_id)
            except OpenAIError as err:
                typer.echo(f"{batch_id} → {err}")
                continue

            if job.status in ("validating", "in_progress", "finalizing"):
                continue
            if job.status == "failed":
                error_message = (
                    job.errors.data[0].message
                    if job.errors and job.errors.data
                    else "No specific error message provided."
                )
                typer.echo(f"⚠ batch {job.id} failed: {error_message}")
                pending.pop(mf)
                pbar.update(1)
                continue

            if job.status == "completed":
                # Record latest API timestamp
                if last_completed_ts is None:
                    last_completed_ts = job.completed_at
                else:
                    last_completed_ts = max(last_completed_ts, job.completed_at)

                fileResponse = client.files.content(job.output_file_id)
                fileContent = fileResponse.text
                for cid, vec in parse_result_file(fileContent):
                    original_text = id2text.get(cid)
                    if original_text is None:
                        typer.echo(f"⚠️  Warning: Could not find original text for ID '{cid}' in input TSV. Skipping.", err=True)
                        continue
                    writer.add(cid, original_text, vec)
                writer.flush()
                pending.pop(mf)
                pbar.update(1)

        if pending:
            time.sleep(POLL_SECS)

    pbar.close()

    # Use the API's timestamp if available
    if last_completed_ts:
        finish_dt = datetime.fromtimestamp(last_completed_ts)
        finish_str = finish_dt.strftime("%Y-%m-%d %H:%M:%S")
        typer.echo(f"Last batch processed at (API time): {finish_str}")
    else:
        typer.echo("No complete timestamp available. Please check the API status.")

if __name__ == "__main__":
    app()
