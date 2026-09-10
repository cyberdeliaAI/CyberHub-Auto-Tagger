"""Compare a baseline checkout with this module; no downloads or user DB writes.

python tests/benchmark.py --baseline /path/to/baseline [--model-dir /path/to/model]
"""
import argparse
import json
import platform
import sqlite3
import statistics
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from support import load_module


def median_seconds(function, iterations=1, repeats=5):
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter() - started) / iterations)
    return statistics.median(samples)


def comparison(before, after):
    return {"before_ms": round(before * 1000, 3), "after_ms": round(after * 1000, 3),
            "speedup": round(before / after, 2), "less_time_percent": round((1 - after / before) * 100, 1)}


def setup_job(implementation, tagger, directory, count, simulated=False):
    instance = implementation.AutoTaggerModule(SimpleNamespace(resource_path=lambda *args: str(directory)))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE files (path TEXT PRIMARY KEY, mtime REAL, folder TEXT)")
    conn.executemany("INSERT INTO files VALUES (?,1,'images')", [(f"{i}.png",) for i in range(count)])
    conn.commit()
    instance.gallery = SimpleNamespace(
        db=SimpleNamespace(lock=threading.Lock(), _get_conn=lambda: conn,
                           resolve_path=lambda path: str(directory / path)),
        thumb_dir=str(directory / "thumbs"),
    )
    instance._create_tables()
    instance._job.update(id="benchmark", scope="all", active=True, batch_size=1,
                         retag=True, ceiling_rowid=count)
    instance._get_tagger = lambda: tagger
    if simulated:
        def load_image(path, target):
            time.sleep(0.01)
            return Image.new("RGBA", (32, 32)), False
        instance._load_image = load_image
    return instance, conn


def run_job(instance, tagger, optimized):
    if optimized:
        instance._process_job_batches(tagger)
    else:
        while rows := instance._next_scope_rows(1):
            instance._process_rows(rows)


def benchmark_jobs(old_module, new_module, old_tagger, new_tagger, directory, count, simulated):
    samples = [[], []]
    for repeat in range(5):
        outputs = []
        # Alternate execution order to reduce warm-cache/thermal bias.
        for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            instance, conn = setup_job((old_module, new_module)[index], (old_tagger, new_tagger)[index],
                                       directory, count, simulated)
            try:
                started = time.perf_counter()
                run_job(instance, (old_tagger, new_tagger)[index], bool(index))
                samples[index].append(time.perf_counter() - started)
                if not simulated:
                    print(f"Run {repeat + 1}, {'optimized' if index else 'baseline'}: {samples[index][-1]:.3f}s", flush=True)
                assert instance._job["done"] == count and instance._job["failed"] == 0
                outputs.append(conn.execute("SELECT * FROM auto_tagger_scores ORDER BY file_path,tag_name,category").fetchall())
            finally:
                conn.close()
        assert outputs[0] == outputs[1], "Tag scores differ from baseline"
    result = comparison(*(statistics.median(values) for values in samples))
    result["samples_seconds"] = [[round(v, 4) for v in values] for values in samples]
    result["images_per_run"] = count
    result["identical_tag_scores"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    old_module, old = load_module(args.baseline, "auto_tagger_baseline")
    new_module, new = load_module(name="auto_tagger_optimized")
    report = {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}
    old_tagger, new_tagger = old.WDTagger("unused"), new.WDTagger("unused")
    categories = [9] * 4 + [0] * 8000 + [4] * 2857
    for tagger in (old_tagger, new_tagger):
        tagger._names = [f"tag_{i}" for i in range(len(categories))]
        tagger._categories = categories
        tagger._category_indexes = {c: [i for i, value in enumerate(categories) if value == c] for c in (0, 4, 9)}
    new_tagger._category_indexes = {c: np.asarray(indexes) for c, indexes in new_tagger._category_indexes.items()}
    scores = np.random.default_rng(42).beta(0.2, 10, len(categories)).astype(np.float32)
    for adaptive in (False, True):
        functions = [lambda t=t: t._postprocess(scores, adaptive_threshold=adaptive) for t in (old_tagger, new_tagger)]
        assert [[(tag.name, tag.score) for tag in f().all] for f in functions][0] == [(tag.name, tag.score) for tag in functions[1]().all]
        report["postprocess_adaptive_" + str(adaptive).lower()] = comparison(*(median_seconds(f, iterations=100) for f in functions))
    with Image.new("RGB", (448, 448), "blue") as source:
        report["prepare_rgb_448"] = comparison(*(median_seconds(lambda t=t: t._prepare_image(source), 100) for t in (old_tagger, new_tagger)))

    class SimulatedTagger:
        target_size = 32
        def tag_images(self, images, **kwargs):
            time.sleep(0.02)
            return [new.TagResult(general=[new.Tag("test", "general", 0.8)]) for image in images]

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        simulated = SimulatedTagger()
        report["simulated_io_10ms_inference_20ms"] = benchmark_jobs(old_module, new_module, simulated, simulated, directory, 24, True)
        print(json.dumps(report, indent=2), flush=True)
        if args.model_dir:
            print("Loading local model on CPU for the real pipeline benchmark...", flush=True)
            new_tagger = new.WDTagger(args.model_dir, provider="CPU").load()
            print("Model loaded; warming up...", flush=True)
            new_tagger.warmup()
            print("Warm-up complete; creating six local PNG fixtures...", flush=True)
            old_tagger = old.WDTagger(args.model_dir, provider="CPU")
            # Share the warmed session so model loading is excluded equally.
            old_tagger.__dict__.update(new_tagger.__dict__)
            old_tagger._category_indexes = {c: list(indexes) for c, indexes in new_tagger._category_indexes.items()}
            rng = np.random.default_rng(42)
            for i in range(6):
                with Image.fromarray(rng.integers(0, 256, (1440, 2560, 3), dtype=np.uint8)) as image:
                    image.save(directory / f"{i}.png")
            report["real_cpu_model_local_png"] = benchmark_jobs(old_module, new_module, old_tagger, new_tagger, directory, 6, False)
            report["model_provider"] = new_tagger.device
            report["model_input"] = str(new_tagger._session.get_inputs()[0].shape)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
