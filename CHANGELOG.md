# Auto Tagger 1.0.6

Background tagging now loads the next image batch while the model processes
the current batch. The queue keeps at most one batch ahead and releases loaded
images when paused work is cancelled or an error occurs.

- Faster tag filtering and adaptive MCut thresholds using NumPy.
- Fewer image copies and paged metadata reads for selected images.
- Faster startup checks for an existing tag catalog.
- Preserve tag ordering, thresholds, transparency and EXIF orientation.
- Retain single-image fallback when a provider rejects larger batches.
- Document the Windows CUDA 12 / cuDNN 9 GPU runtime installation.

Validation: 21 regression tests pass. A local benchmark using the real WD-EVA02
model on CPU reduced median job time by 10.1%, with identical stored tags and
scores. GPU/NAS speedups have not been benchmarked; results depend on hardware
and image sources.

Install through **Module Manager → Check for updates → Auto Tagger → Update**,
then restart CyberHub. The attached ZIP can also be imported through Settings.
Downloaded models, existing tags and settings are preserved. An existing working
CUDA runtime does not need to be reinstalled.

Minimum CyberHub version: 1.3.0. Requires Gallery.

## 1.0.5

Initial standalone Auto Tagger module release.
