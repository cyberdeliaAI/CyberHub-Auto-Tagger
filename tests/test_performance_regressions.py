import io
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image, ImageOps

from support import load_module


module, inference = load_module()


def reference_mcut(values, floor=0.0):
    if len(values) < 2:
        return float(floor)
    ordered = sorted(map(float, values), reverse=True)
    i = max(range(len(ordered) - 1), key=lambda i: ordered[i] - ordered[i + 1])
    return max(float(floor), (ordered[i] + ordered[i + 1]) / 2)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tagger = inference.WDTagger("unused")
        self.tagger._names = [f"tag_{i}" for i in range(10861)]
        self.tagger._categories = [9] * 4 + [0] * 8000 + [4] * 2857
        self.tagger._category_indexes = {
            category: np.flatnonzero(np.array(self.tagger._categories) == category)
            for category in (0, 4, 9)
        }

    def test_thresholds_and_sorted_tags_match_reference(self):
        scores = np.random.default_rng(123).random(10861, dtype=np.float32) ** 8
        # Include exact threshold equality, ties and a float32 boundary.
        scores[4:8] = [0.35, 0.5, 0.5, 0.25]
        for adaptive in (False, True):
            for threshold in (0.25, 0.35, float(scores[4]), float(scores[4]) - 1e-12):
                result = self.tagger._postprocess(scores, threshold, 0.85, adaptive)
                expected = {}
                for category, label, cutoff in ((0, "general", threshold), (4, "character", 0.85)):
                    indexes = self.tagger._category_indexes[category]
                    if adaptive:
                        cutoff = reference_mcut([scores[i] for i in indexes], 0.15 if category == 4 else 0)
                    tags = [(self.tagger._names[i], label, float(scores[i])) for i in indexes if float(scores[i]) > cutoff]
                    expected[label] = sorted(tags, key=lambda tag: tag[2], reverse=True)
                self.assertEqual([(t.name, t.category, t.score) for t in result.general], expected["general"])
                self.assertEqual([(t.name, t.category, t.score) for t in result.character], expected["character"])
                best = max(range(4), key=lambda i: float(scores[i]))
                self.assertEqual(result.predicted_rating, f"tag_{best}")

    def test_mcut_empty_single_ties_and_floor(self):
        for values in ([], [0.5], [0.9, 0.6, 0.3], [0.5] * 5, [0.01, 0.03]):
            for floor in (0, 0.15):
                self.assertEqual(inference.mcut_threshold(values, floor), reference_mcut(values, floor))

    def test_empty_categories(self):
        self.tagger._category_indexes = {}
        self.assertEqual(self.tagger._postprocess([], adaptive_threshold=True).all, [])

    def test_preprocessing_pixels_layout_transparency_and_input_ownership(self):
        self.tagger._target_size = 32
        rng = np.random.default_rng(42)
        for mode in ("RGB", "RGBA", "L", "LA", "P"):
            for size in ((32, 32), (63, 19), (11, 23)):
                image = Image.fromarray(rng.integers(0, 256, (*size[::-1], 4), dtype=np.uint8)).convert(mode)
                before = image.tobytes()
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                if max(rgb.size) != 32:
                    scale = 32 / max(rgb.size)
                    rgb = rgb.resize(tuple(max(1, round(n * scale)) for n in rgb.size), Image.Resampling.BICUBIC)
                padded = Image.new("RGB", (32, 32), "white")
                padded.paste(rgb, ((32 - rgb.width) // 2, (32 - rgb.height) // 2))
                expected = np.asarray(padded, dtype=np.float32)[:, :, ::-1]
                for layout in ("NHWC", "NCHW"):
                    self.tagger._input_layout = layout
                    actual = self.tagger._prepare_image(image)
                    np.testing.assert_array_equal(actual, expected if layout == "NHWC" else expected.transpose(2, 0, 1))
                    self.assertEqual(image.tobytes(), before)
                image.close()

    def test_exif_orientation_and_detached_downsample(self):
        for orientation in range(1, 9):
            source = Image.fromarray(np.arange(90 * 60 * 3, dtype=np.uint8).reshape(60, 90, 3))
            exif = Image.Exif()
            exif[274] = orientation
            stream = io.BytesIO()
            source.save(stream, format="JPEG", exif=exif)
            stream.seek(0)
            with Image.open(stream) as image:
                expected = ImageOps.exif_transpose(image)
                expected.thumbnail((32, 32), Image.Resampling.LANCZOS, reducing_gap=3.0)
                expected = expected.convert("RGBA")
                actual = module.AutoTaggerModule._downsample_source(image, 32)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            actual.close()

    def test_inference_rejects_wrong_batch_result_count(self):
        self.tagger._target_size = 32
        self.tagger._session = Mock()
        self.tagger._session.get_providers.return_value = ["CPUExecutionProvider"]
        self.tagger._session.run.return_value = [np.zeros((1, 10861), dtype=np.float32)]
        with Image.new("RGB", (32, 32)) as image:
            with self.assertRaisesRegex(RuntimeError, "for 2 images"):
                self.tagger.tag_images([image, image])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.instance = module.AutoTaggerModule(SimpleNamespace(resource_path=lambda *args: self.tmp.name))
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.execute("CREATE TABLE files (path TEXT PRIMARY KEY, mtime REAL, folder TEXT)")
        db = SimpleNamespace(lock=threading.Lock(), _get_conn=lambda: self.conn)
        self.instance.gallery = SimpleNamespace(db=db)
        self.instance._create_tables()
        self.instance._job.update(id="job", active=True, scope="all", batch_size=2, ceiling_rowid=1000, retag=True)
        self.created = []
        self.load_threads = []
        self.tagger = Mock(target_size=32, device="CPUExecutionProvider")
        self.tagger.tag_images.side_effect = lambda images, **kwargs: [self.result() for _ in images]
        self.tagger.tag_image.side_effect = lambda image, **kwargs: self.result()
        self.instance._get_tagger = lambda: self.tagger
        self.instance._load_image = self.load_image

    def result(self):
        return inference.TagResult(general=[inference.Tag("blue", "general", 0.8)])

    def load_image(self, path, target):
        image = Image.new("RGBA", (target, target), "blue")
        self.created.append(image)
        self.load_threads.append(threading.get_ident())
        return image, path.endswith("0.png")

    def seed(self, count=7):
        self.conn.executemany("INSERT INTO files VALUES (?,1,'folder')", [(f"{i}.png",) for i in range(count)])
        self.conn.commit()

    def run_batches(self):
        self.instance._process_job_batches(self.tagger)
        for image in self.created:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_all_rows_stored_once_in_cursor_order_and_catalog_consistent(self):
        self.seed()
        stored = []
        original = self.instance._store_results
        def store(items, job):
            stored.extend(row[0] for row in items)
            original(items, job)
        self.instance._store_results = store
        self.run_batches()
        self.assertEqual(stored, [f"{i}.png" for i in reversed(range(7))])
        self.assertEqual(self.instance._job["done"], 7)
        self.assertEqual(self.instance._job["used_thumbnail"], 1)
        self.assertEqual(self.conn.execute("SELECT use_count FROM auto_tagger_catalog").fetchone()[0], 7)
        self.assertTrue(all(t != threading.get_ident() for t in self.load_threads))

    def test_load_overlaps_inference_with_only_one_batch_ahead(self):
        self.seed(6)
        started = threading.Event()
        original_load = self.instance._load_image
        def load(path, target):
            result = original_load(path, target)
            if path == "3.png":
                started.set()
            return result
        self.instance._load_image = load
        calls = 0
        def infer(images, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.assertTrue(started.wait(2), "next batch should load during inference")
                self.assertLessEqual(len(self.created), 4)
            return [self.result() for _ in images]
        self.tagger.tag_images.side_effect = infer
        self.run_batches()
        self.assertEqual(self.instance._job["failed"], 0)
        self.assertEqual(calls, 3)

    def test_fixed_batch_fallback_splits_already_prefetched_batch(self):
        self.seed()
        sizes = []
        def infer(images, **kwargs):
            sizes.append(len(images))
            if len(images) > 1:
                raise RuntimeError("fixed batch")
            return [self.result()]
        self.tagger.tag_images.side_effect = infer
        self.run_batches()
        self.assertEqual(sizes, [2, 1, 1, 1, 1, 1])
        self.assertEqual(self.instance._job["done"], 7)
        self.assertEqual(self.instance._job["batch_size"], 1)

    def test_load_failures_do_not_skip_other_images(self):
        self.seed(4)
        original = self.instance._load_image
        def load(path, target):
            if path == "3.png":
                raise OSError("broken image")
            return original(path, target)
        self.instance._load_image = load
        self.run_batches()
        self.assertEqual(self.instance._job["done"], 4)
        self.assertEqual(self.instance._job["failed"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM auto_tagger_files").fetchone()[0], 4)

    def test_inference_failure_does_not_skip_remaining_images(self):
        self.seed(4)
        self.tagger.tag_images.side_effect = RuntimeError("batch failed")
        self.tagger.tag_image.side_effect = [RuntimeError("image failed"), self.result(), self.result(), self.result()]
        self.run_batches()
        self.assertEqual(self.instance._job["done"], 4)
        self.assertEqual(self.instance._job["failed"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM auto_tagger_scores").fetchone()[0], 3)

    def test_cancellation_discards_prefetched_images(self):
        self.seed(8)
        def infer(images, **kwargs):
            self.instance._job_cancel.set()
            return [self.result() for _ in images]
        self.tagger.tag_images.side_effect = infer
        self.run_batches()
        self.assertEqual(self.instance._job["done"], 2)
        self.assertLessEqual(len(self.created), 4)

    def test_cancel_while_paused_does_not_load_images(self):
        self.seed()
        self.instance._job_pause.set()
        cancel = threading.Timer(0.03, self.instance._job_cancel.set)
        cancel.start()
        self.run_batches()
        cancel.join()
        self.assertEqual(self.created, [])

    def test_cancel_during_loading_does_not_run_inference(self):
        self.seed()
        original = self.instance._load_image
        def load(path, target):
            result = original(path, target)
            self.instance._job_cancel.set()
            return result
        self.instance._load_image = load
        self.run_batches()
        self.assertEqual(self.instance._job["done"], 0)
        self.tagger.tag_images.assert_not_called()

    def test_resume_after_prefetch_keeps_all_images(self):
        self.seed(6)
        pause_observations = []
        timers = []
        calls = 0
        def resume():
            pause_observations.append(calls)
            self.instance._resume_job()
        def infer(images, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.instance._pause_job()
                timer = threading.Timer(0.03, resume)
                timers.append(timer)
                timer.start()
            return [self.result() for _ in images]
        self.tagger.tag_images.side_effect = infer
        self.run_batches()
        for timer in timers:
            timer.join()
        self.assertEqual(pause_observations, [1])
        self.assertEqual(self.instance._job["done"], 6)
        self.assertEqual(self.instance._job["failed"], 0)

    def test_error_closes_current_and_pending_images(self):
        self.seed()
        self.instance._store_results = Mock(side_effect=RuntimeError("database failed"))
        with self.assertRaisesRegex(RuntimeError, "database failed"):
            self.instance._process_job_batches(self.tagger)
        for image in self.created:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_selected_paths_keep_order_and_page_database_reads(self):
        self.seed(260)
        paths = [f"{i}.png" for i in range(260)] + ["missing.png"]
        self.instance._job_paths = paths
        self.instance._job.update(scope="paths", batch_size=8)
        queries = []
        self.conn.set_trace_callback(queries.append)
        rows = [row for batch in self.instance._job_batches() for row in batch]
        self.assertEqual([row[0] for row in rows], paths[:-1])
        self.assertEqual(len([q for q in queries if q.startswith("SELECT path,mtime")]), 3)

    def test_scope_skips_current_revision_and_respects_folder_and_ceiling(self):
        self.seed(5)
        self.instance._store_result("1.png", 1, "previous", self.result())
        self.conn.execute("UPDATE files SET folder='elsewhere' WHERE path='2.png'")
        self.instance._job.update(scope="folder", folder="folder", retag=False, ceiling_rowid=4)
        self.run_batches()
        self.assertEqual(self.instance._job["done"], 2)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM auto_tagger_files").fetchone()[0], 3)

    def test_worker_advances_new_image_baseline_only_after_completion(self):
        self.seed(4)
        self.instance._job.update(scope="new", new_after=2, ceiling_rowid=4)
        self.instance._job_worker()
        self.assertEqual(self.instance._job["state"], "complete")
        self.assertEqual(self.instance._job["done"], 2)
        self.assertEqual(self.instance._get_config("auto_new_after_rowid"), "4")
        self.instance._job.update(active=True, ceiling_rowid=10)
        self.instance._job_cancel.set()
        self.instance._job_worker()
        self.assertEqual(self.instance._job["state"], "cancelled")
        self.assertEqual(self.instance._get_config("auto_new_after_rowid"), "4")

    def test_cuda_warmup_failure_prevents_loading_queue(self):
        self.seed()
        self.tagger.device = "CUDAExecutionProvider"
        self.tagger.warmup.side_effect = RuntimeError("missing CUDA DLL")
        self.instance._job_worker()
        self.assertEqual(self.instance._job["state"], "error")
        self.assertIn("CUDA warm-up failed", self.instance._job["error"])
        self.assertEqual(self.created, [])

    def test_missing_catalog_is_rebuilt_from_existing_scores(self):
        self.seed(2)
        self.instance._store_result("0.png", 1, "previous", self.result())
        self.instance._store_result("1.png", 1, "previous", self.result())
        self.conn.execute("DELETE FROM auto_tagger_catalog")
        self.conn.commit()
        self.instance._create_tables()
        row = self.conn.execute("SELECT use_count,score_sum FROM auto_tagger_catalog").fetchone()
        self.assertEqual(row[0], 2)
        self.assertAlmostEqual(row[1], 1.6)


if __name__ == "__main__":
    unittest.main()
