# CyberHub Auto Tagger

Add searchable visual AI tags to Gallery images with WD-EVA02.

- Version: `1.0.6`
- Channel: `stable`
- Publisher: `official`

## Installation

1. Open **Module Manager** in CyberHub.
2. Click **Check for updates**.
3. Find **Auto Tagger** and choose **Install** or **Update**.
4. Restart CyberHub when the installation finishes.

The ZIP attached to this repository's GitHub Release can also be imported manually through Settings.

## Required CyberHub Modules

- `gallery`

## Python Packages

- `Pillow>=9.0`
- `numpy>=1.24`

## NVIDIA acceleration on Windows

Choosing CUDA in Settings selects an execution provider; it does not install
the GPU runtime. If the Runtime card says **CUDA unavailable; using CPU**,
stop CyberHub and run these commands from its installation folder:

```bat
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
.venv\Scripts\python.exe -m pip install --upgrade "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"
```

This installs the CUDA 12 / cuDNN 9 runtime family and its DLLs. A compatible
NVIDIA driver must already be installed. ONNX Runtime GPU 1.27 and newer use
CUDA 13 by default; see the [official compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Restart CyberHub and run a small tagging job to validate CUDA with the model.
Existing GPU runtime installations are not changed by this module update.

## Privacy

CyberHub runs locally. A module uses an external service only when its function requires it and the user starts that action.

## Performance

Background jobs load and downsample one batch ahead while the current batch is
tagged. One loader keeps this bounded to the current and next batch, plus the
source image being decoded. Database writes and model inference remain serial.
Pause stops further processing after the current batch; a read already in
progress can finish. Cancel discards prefetched images. Providers that reject
larger batches still fall back to one image at a time.

Tag filtering and adaptive MCut thresholds use NumPy. Threshold comparisons,
tag order, transparency handling and EXIF orientation are preserved. Selected
image jobs read Gallery metadata in pages of up to 128 paths. Startup checks for
an existing tag catalog no longer count every stored score.

The default image source and batch settings are unchanged. For originals on a
NAS, **Gallery thumbnail (fast)** can further reduce reading time, with the
existing fine-detail accuracy tradeoff. No new dependencies are required.

## Development and validation

Run from this repository using Python with Pillow and NumPy installed:

```sh
python -m unittest discover -s tests -v
```

The tests use temporary images and an isolated SQLite database; CyberHub does
not need to be running. To compare this checkout with an unmodified baseline:

```sh
python tests/benchmark.py --baseline /path/to/baseline-checkout
python tests/benchmark.py --baseline /path/to/baseline-checkout --model-dir /path/to/wd-eva02-large-tagger-v3
```

The optional model benchmark requires ONNX Runtime and an already downloaded
model. It shares one warmed CPU session between both implementations, alternates
run order, compares stored tag scores and reports the median of five runs.
See [performance measurements](docs/performance.md) for the local results.

## License

See `LICENSE.md` and `THIRD-PARTY-NOTICES.md`.
