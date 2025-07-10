# ms_marco_embed_fast.py (v3)
"""
Embed the MS MARCO query set with **text‑embedding‑3‑large** as fast as possible
on a 6‑core / 16 GB MacBook (Tier‑3 account) under these constraints:

* ≤ 8 k tokens **per** input (assumed).  No per‑string checking.
* ≤ 2048 inputs **per request** (OpenAI limit).
* ≤ 5 000 RPM (Tier‑3).  Tokens/min not enforced.

Install
-------
```bash
pip install --upgrade openai aiolimiter tenacity numpy tqdm typer
```

Example
-------
```bash
python ms_marco_embed_fast.py \
  --input /Volumes/T7/Data/MSMarco/queries/queries.eval.tsv \
  --out-dir /Volumes/T7/Data/MSMarco/queries/embeddings-eval \
  --workers 60 --shard-size 10000
```

The script emits files `emb_00000.npz`, `emb_00001.npz`, … each holding
`ids`, `texts`, `vectors`.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI, OpenAIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm  # progress bar
import typer
import dotenv
import os
import tiktoken
app = typer.Typer(add_completion=False)

MODEL = "text-embedding-3-large"
MAX_BATCH = 2048          # inputs / request (hard limit)
VECTOR_DIMS = 3072        # embedding dimensionality
MAX_TOKENS = 8191
ENCODING = tiktoken.encoding_for_model(MODEL) # Get the tokenizer encoding

# asyncio
# async methods are corotines that happen inside the event loop
# wait for results of other async operations without blocking the event loop
# await is used to wait for the result of an async operation
# asyncio.create_task is used to run a coroutine in the background
# asyncio.run is used to run the event loop
# asyncio.Queue is used to communicate between the producer and consumers
# asyncio.Queue.put is used to put a batch into the queue
# asyncio.Queue.get is used to get a batch from the queue
# asyncio.Queue.join is used to wait for the queue to be empty
# asyncio.Queue.task_done is used to indicate that a batch has been processed
# asyncio.Queue.put(None) is used to put a sentinel into the queue to indicate that the producer has finished

class ShardWriter:
    """Write embeddings to compressed NPZ shards of fixed row count."""

    def __init__(self, out_dir: Path, shard_size: int = 10_000):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.reset()

    def reset(self):
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.idx = getattr(self, "idx", 0) # shard index

    async def add(self, docs: List[Dict[str, str]], vecs: List[np.ndarray]):
        self.ids.extend(d["id"] for d in docs)
        self.texts.extend(d["text"] for d in docs)
        self.vecs.extend(vecs)
        if len(self.ids) >= self.shard_size:
            await self.flush()

    async def flush(self):
        if not self.ids:
            return
        file = self.out_dir / f"emb_{self.idx:05d}.npz"
        np.savez_compressed(
            file,
            ids=np.asarray(self.ids),
            texts=np.asarray(self.texts),
            vectors=np.vstack(self.vecs).astype(np.float32),
        )
        self.idx += 1
        self.reset()

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(OpenAIError),
)
async def embed_batch(client: AsyncOpenAI, inputs: List[str]):
    resp = await client.embeddings.create(model=MODEL, input=inputs)
    return [np.asarray(obj.embedding, dtype=np.float32) for obj in resp.data]

async def consumer(
    queue: asyncio.Queue,
    rpm_limiter: AsyncLimiter,
    tpm_limiter: AsyncLimiter,
    client: AsyncOpenAI,
    writer: ShardWriter,
    pbar: tqdm,
):
    while True: 
        batch = await queue.get()
        if batch is None:
            await queue.put(None)  # propagate sentinel
            break
        
        # Use pre-calculated token count for the batch
        batch_token_count = sum(d["token_count"] for d in batch)
        
        async with rpm_limiter:        # obey RPM
            async with tpm_limiter.acquire(batch_token_count):    # obey TPM
                vecs = await embed_batch(client, [d["text"] for d in batch])
        await writer.add(batch, vecs)
        pbar.update(len(batch))     # progress!
        queue.task_done()

async def producer(docs: List[Dict[str, str]], queue: asyncio.Queue, batch_size: int):
    batch: List[Dict[str, str]] = []
    for doc in docs:
        text = doc["text"]
        tokens = ENCODING.encode(text)
        token_count = len(tokens)

        if token_count > MAX_TOKENS:
            # Skip documents exceeding the token limit and log their ID
            typer.echo(f"Skipping document ID {doc['id']}: Exceeds {MAX_TOKENS} tokens ({token_count} tokens).")
            continue # Skip this document
        
        # Store token count for valid documents
        doc["token_count"] = token_count

        batch.append(doc)
        if len(batch) == batch_size:
            await queue.put(batch)
            batch = []
    if batch:
        await queue.put(batch)
    await queue.put(None)  # sentinel

# --------------------------------------------------------------------------- #
@app.command()
def main(
    input: Path = typer.Option(..., help="MS MARCO TSV file (id \t text)"),
    out_dir: Path = typer.Option(..., help="Directory for NPZ shards"),
    batch_size: int = typer.Option(MAX_BATCH, help="Rows / request (<=2048)"),
    shard_size: int = typer.Option(10_000, help="Rows per NPZ shard"),
    workers: int = typer.Option(60, help="Concurrent requests in flight"),
    max_rpm: int = typer.Option(4_800, help="Requests‑per‑minute ceiling (<5 000)"),
    max_tpm: int = typer.Option(4_800_000, help="Tokens‑per‑minute ceiling (<5 000_000)"),
):
    """Embed the entire corpus at maximum throughput given the constraints."""

    if batch_size > MAX_BATCH:
        typer.echo(f"batch_size capped to {MAX_BATCH}")
        batch_size = MAX_BATCH

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TSV → list[dict]
    docs: List[Dict[str, str]] = []
    with input.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                docs.append({"id": parts[0], "text": parts[1]})
    total = len(docs)
    typer.echo(f"Loaded {total:,} rows → embedding …")

    # Shared resources
    rpm_limiter = AsyncLimiter(max_rpm, time_period=60)
    tpm_limiter = AsyncLimiter(max_tpm, time_period=60)
    dotenv.load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    writer = ShardWriter(out_dir, shard_size=shard_size)
    queue: asyncio.Queue = asyncio.Queue(maxsize=workers * 2) 

    async def orchestrate():
        pbar = tqdm(total=total, desc="Embedded", unit="doc")
        prod_task = asyncio.create_task(producer(docs, queue, batch_size))
        cons_tasks = [
            asyncio.create_task(consumer(queue, rpm_limiter, tpm_limiter, client, writer, pbar))
            for _ in range(workers)
        ]
        await prod_task # wait for producer to finish
        await queue.join() # wait for queue to be empty
        for task in cons_tasks:
            await task # wait for all consumers to finish
        await writer.flush()
        pbar.close()

    start = time.perf_counter()
    asyncio.run(orchestrate())
    elapsed = time.perf_counter() - start
    typer.echo(
        f"Done in {elapsed/60:.1f} min (≈ {total/elapsed:,.0f} rows/s; ~{workers} concurrent calls)."
    )

if __name__ == "__main__":
    app()
