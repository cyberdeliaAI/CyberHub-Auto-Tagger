# Auto Tagger performance measurements

Measured on 2026-09-10 against unmodified version 1.0.5, commit `d8c150e`.
These optimizations are included in Auto Tagger 1.0.6.

Environment: macOS 26.6.2 ARM64, Python 3.14.6, NumPy 2.4.6, Pillow 12.2.0,
ONNX Runtime 1.27.0. The real-model test used the CPU execution provider and the
existing pinned WD-EVA02 model, with input `['batch_size', 448, 448, 3]`.

| Measurement | Before | After | Reduction in elapsed time |
| --- | ---: | ---: | ---: |
| Filter 10,861 synthetic scores, fixed thresholds | 0.592 ms | 0.016 ms | 97.3% |
| Filter the same scores with adaptive MCut | 2.308 ms | 0.226 ms | 90.2% |
| Prepare a 448 × 448 RGB image | 0.571 ms | 0.301 ms | 47.2% |
| 24-image simulated job: 10 ms read + 20 ms inference per image | 869.644 ms | 596.746 ms | 31.4% |
| Real WD-EVA02 CPU job, six generated 2560 × 1440 PNG files | 13.029 s | 11.710 s | 10.1% |

These are medians of five runs. Microbenchmarks perform 100 iterations per run.
Job tests alternate baseline/optimized execution order and include loading,
preprocessing, inference, postprocessing and persistence in an isolated
in-memory SQLite database. Both implementations share the same warmed ONNX
session, so initial model loading and warm-up are excluded equally. No Gallery
database or personal images are used. All stored tag names, categories and
scores matched exactly between baseline and optimized runs.

Real-model job samples, in seconds:

| Run | Before | After |
| --- | ---: | ---: |
| 1 | 13.0288 | 10.7776 |
| 2 | 11.7675 | 11.4139 |
| 3 | 12.0687 | 11.7103 |
| 4 | 25.2949 | 12.8072 |
| 5 | 15.4367 | 16.0331 |

The real-model results have substantial timing variation, including a slower
optimized fifth run. They demonstrate a local median improvement, not a
guaranteed percentage for a real library. CPU inference remains the dominant
cost. The score-filtering and RGB preparation numbers measure individual
operations, not whole-job speed. Normal Gallery loading supplies RGBA images,
so the RGB fast-path measurement does not describe every image in a job.
CUDA, CoreML, DirectML, NAS throughput and a running Gallery UI were not
benchmarked. The simulated read test measures overlap under controlled delays,
not actual NAS performance.

The regression suite covers exact preprocessing pixels, EXIF orientation,
transparency, float32 threshold boundaries, stable tag order, adaptive MCut,
output batch validation, cursor/folder selection, selected-path query paging,
prefetch bounds and overlap, batch fallback, pause/resume, cancellation,
image cleanup, load/inference/database errors, new-image baselines, CUDA
warm-up failure and catalog rebuilding.

Reproduce with the commands in the repository README. The benchmark uses only
temporary image files and SQLite databases and never downloads a model.
