"""Small release checks. Full photographic benchmarks are documented separately."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import threading
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP = ROOT / "build/test-temp"
TEST_TEMP.mkdir(parents=True, exist_ok=True)
assert TEST_TEMP.resolve().is_relative_to(ROOT.resolve())
spec = importlib.util.spec_from_file_location("untorn", ROOT / "source/untorn_engine.py")
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


class ReleaseChecks(unittest.TestCase):
    def test_spatial_branch_routing_and_weights(self):
        import numpy as np

        y, x = np.mgrid[:64, :80]
        carrier = np.repeat(np.where(x > 0.6 * y + 10, 180.0, 60.0)[..., None], 3, axis=2)
        for mode in ("fast", "max", "xray"):
            mask = u.spatial_edge_mask(carrier)
            envelope = u.spatial_envelope(carrier)
            with patch.object(
                u, "spatial_realign", side_effect=AssertionError("Unaligned branch warped")
            ):
                unaligned, _ = u.spatial_branch(carrier, carrier, mode, False, True, mask, envelope)
            with patch.object(
                u, "spatial_fxaa", side_effect=AssertionError("Aligned branch used FXAA")
            ):
                aligned, _ = u.spatial_branch(carrier, carrier, mode, True, False, mask, envelope)
            with (
                patch.object(u, "spatial_fxaa", wraps=u.spatial_fxaa) as aa,
                patch.object(u, "spatial_realign", wraps=u.spatial_realign) as warp,
                patch.object(u, "spatial_envelope", wraps=u.spatial_envelope) as bounds,
            ):
                output, metadata = u.spatial_cleanup(carrier, carrier, mode)
            np.testing.assert_array_equal(output, 0.77 * unaligned + 0.23 * aligned)
            self.assertEqual(aa.call_count, 1)
            self.assertEqual(warp.call_count, 1)
            self.assertEqual(bounds.call_count, 1)
            self.assertEqual([b["weight"] for b in metadata["branches"]], [0.77, 0.23])

    def test_previous_spatial_cache_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            folder = Path(folder)
            source = folder / "input.png"
            Image.new("RGB", (16, 16), (80, 110, 140)).save(source)
            request = dict(input=str(source), folder=str(folder / "job"), mode="max", scale=4)
            u.process_request(request)
            stamp = folder / "job/signature.json"
            previous = u.read_json(stamp)
            previous["version"] = "1.2.0"
            u.save_json(stamp, previous)
            with self.assertRaisesRegex(ValueError, "different input or setting"):
                u.process_request(request)

    def test_spatial_flat_and_texture_residual(self):
        import numpy as np

        flat = np.full((64, 64, 3), [83.0, 119.0, 151.0])
        for mode in ("fast", "max", "xray"):
            out, meta = u.spatial_cleanup(flat, flat, mode)
            np.testing.assert_array_equal(out, flat)
            self.assertIn("skipped", meta)
        y, x = np.mgrid[:80, :96]
        carrier = np.repeat(np.where(x > 0.6 * y + 15, 190.0, 60.0)[..., None], 3, axis=2)
        residual = np.stack(
            [0.5 * np.sin(x * 1.7 + y), 0.4 * np.cos(x + y * 1.9), 0.3 * np.sin(x - y)], -1
        )
        for mode in ("fast", "max", "xray"):
            structural, _ = u.spatial_cleanup(carrier, carrier, mode)
            textured, meta = u.spatial_cleanup(carrier + residual, carrier, mode)
            np.testing.assert_allclose(textured - structural, residual, atol=1e-12)
            self.assertTrue(np.isfinite(textured).all())
            self.assertLessEqual(structural.max(), 190.0001)
            self.assertGreaterEqual(structural.min(), 59.9999)
            self.assertEqual(meta["profile"], mode)
            self.assertLessEqual(
                meta["max_x_displacement"], u.SPATIAL_PROFILES[mode]["displacement"] + 1e-6
            )

    def test_spatial_grain_compact_support(self):
        import numpy as np

        y, x = np.mgrid[:64, :64]
        grain = 4 * np.cos(x * 1.7) * np.cos(y * 1.8)
        smooth_base = u.spatial_protect_grain(grain, np.full((64, 64), 120.0))
        compact_base = u.spatial_protect_grain(grain, 120 + 8 * np.cos(x * 1.7) * np.cos(y * 1.8))
        roi = np.s_[12:-12, 12:-12]
        self.assertLess(np.linalg.norm(compact_base[roi]), 0.05 * np.linalg.norm(smooth_base[roi]))
        self.assertGreater(np.linalg.norm(smooth_base[roi]), 0.5 * np.linalg.norm(grain[roi]))
        np.testing.assert_array_equal(
            u.spatial_protect_grain(grain, np.full((64, 64), 120.0)), smooth_base
        )

    def test_spatial_fxaa_range_and_neutrality(self):
        import numpy as np

        y, x = np.mgrid[:96, :96]
        source = np.repeat(np.where(x > y * 0.6 + 20, 220.0, 35.0)[..., None], 3, axis=2)
        out = u.spatial_fxaa(source)
        self.assertGreater(np.max(abs(out - source)), 1)
        self.assertGreaterEqual(out.min(), 35)
        self.assertLessEqual(out.max(), 220)
        np.testing.assert_array_equal(out[..., 0], out[..., 1])
        np.testing.assert_array_equal(out, u.spatial_fxaa(source))

    def test_dimensions(self):
        self.assertEqual(u.dimensions((64, 55), 2.5), (160, 138))
        for value in [float("nan"), float("inf"), 0, 16.01]:
            with self.assertRaises(ValueError):
                u.dimensions((64, 55), value)

    def test_mode_cache_and_vector(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            source = root / "in.png"
            Image.new("RGB", (24, 24), (83, 119, 151)).save(source)
            job = root / "job"
            request = dict(input=str(source), folder=str(job), scale=2.5, svg=False, mode="fast")
            u.save_json(job / "request.json", request)
            first = u.process_request(request)
            self.assertEqual(first["output_size"], [60, 60])
            self.assertEqual(first["mode"], "fast")
            receipts = {p: p.stat().st_mtime_ns for p in (job / "tiles").glob("*/receipt.json")}
            u.process_request(request)
            self.assertEqual(receipts, {p: p.stat().st_mtime_ns for p in receipts})
            with self.assertRaisesRegex(ValueError, "different input or setting"):
                u.process_request(dict(request, mode="max"))
            with self.assertRaisesRegex(ValueError, "Mode must"):
                u.process_request(dict(request, mode="unknown"))
            u.vector_export(job)
            svg = (job / "result.svg").read_text()
            self.assertIn("Untorn Engine", svg)
            for forbidden in ["<image", "<filter", "<script", "data:image"]:
                self.assertNotIn(forbidden, svg)
            with Image.open(job / "result.png") as im:
                self.assertEqual(im.getextrema(), ((83, 83), (119, 119), (151, 151)))

    def test_invalid_input_does_not_stop_queue(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            bad = root / "broken.png"
            bad.write_bytes(b"broken")
            good = root / "good.png"
            Image.new("RGB", (24, 24), (93, 126, 158)).save(good)
            studio = u.Studio(root / "output")
            try:
                a = studio.add(bad, 2.5, False, mode="fast")
                b = studio.add(good, 2.5, False, mode="max")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    jobs = {j["id"]: j for j in studio.snapshot()}
                    if all(jobs[k]["state"] in ("complete", "failed") for k in (a["id"], b["id"])):
                        break
                    time.sleep(0.2)
                self.assertEqual(jobs[a["id"]]["state"], "failed")
                self.assertEqual(jobs[b["id"]]["state"], "complete")
                self.assertEqual(jobs[b["id"]]["mode"], "max")
                final = jobs[b["id"]]["elapsed_seconds"]
                self.assertGreater(final, 0)
                saved = u.read_json(root / "output" / b["id"] / "job.json")
                self.assertEqual(saved["elapsed_seconds"], final)
                time.sleep(0.2)
                self.assertEqual(
                    next(j for j in studio.snapshot() if j["id"] == b["id"])["elapsed_seconds"],
                    final,
                )
            finally:
                studio.shutdown()
                studio.thread.join(timeout=5)

    def test_restoration_cache_contract(self):
        self.assertEqual(u.VERSION, "1.3.0")
        self.assertFalse(hasattr(u, "ps_texture"))
        self.assertFalse(hasattr(u, "phase_texture"))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            source = root / "in.png"
            Image.new("RGB", (24, 24), (83, 119, 151)).save(source)
            job = root / "job"
            job.mkdir()
            old = dict(
                source=u.sha(source), version="1.0.0", size=[96, 96], svg=False, method=u.METHOD
            )
            u.save_json(job / "signature.json", old)
            with self.assertRaisesRegex(ValueError, "different input or setting"):
                u.process_request(
                    dict(input=str(source), folder=str(job), scale=4, svg=False, mode="max")
                )
            self.assertEqual(u.read_json(job / "signature.json"), old)

    def test_fitted_pixel_cutoff_and_tile_propagation(self):
        for size in [(512, 512), (1024, 256), (513, 511), (511, 513)]:
            self.assertTrue(u.use_fitted_for(size), size)
        for size in [(512, 513), (513, 512), (1025, 256), (768, 512)]:
            self.assertFalse(u.use_fitted_for(size), size)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            for size in [(512, 512), (768, 512)]:
                job = Path(folder) / str(size)
                job.mkdir()
                low = Image.new("RGB", size, (83, 119, 151))
                calls = []

                def detail(tile, cache, notify, **kwargs):
                    calls.append(kwargs["use_fitted"])
                    return tile.resize((tile.width * 4, tile.height * 4)), {}

                with patch.object(u, "detail_patch", side_effect=detail):
                    u.reconstruct_tile(low, job, (0, 0, 128, 128), 0, 1, "max")
                self.assertEqual(calls, [u.use_fitted_for(size)])

    def test_progress_parallel_and_retry_monotonic(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            reporter = u.RestorationProgress(folder, 4, False)
            observed = []
            original = u.save_json

            def capture(path, value):
                if Path(path).name == "progress.json":
                    observed.append(value["progress"])
                original(path, value)

            def worker(i):
                for stage in [
                    "Restoring surface detail",
                    "Restoring edges",
                    "Refining middle detail",
                ]:
                    reporter.update(i, stage)
                reporter.update(i, "Tile saved", 1.0)

            with patch.object(u, "save_json", side_effect=capture):
                workers = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
                for t in workers:
                    t.start()
                for t in workers:
                    t.join()
                u.status_file(folder, "Retrying on CPU", 0)
                u.status_file(folder, "Writing PNG", 0.95)
                u.status_file(folder, "Complete", 1)
            self.assertEqual(observed, sorted(observed))
            self.assertEqual(observed[-1], 1)
            self.assertEqual(reporter.values, [1] * 4)

    def test_no_fitted_restoration_path(self):
        import numpy as np

        y, x = np.mgrid[:24, :24]
        pixels = np.stack([80 + x * 2, 90 + y * 2, 100 + x + y], axis=-1).astype(np.uint8)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            with patch.object(
                u, "fitted_four", side_effect=AssertionError("Fitted must be skipped")
            ):
                out, meta = u.detail_patch(
                    Image.fromarray(pixels), folder, mode="fast", use_fitted=False
                )
            self.assertEqual(out.size, (96, 96))
            self.assertFalse(meta["fitted_enabled"])
            self.assertEqual(meta["grain"]["weights"], [1 / 3] * 3)
            self.assertNotIn("Fitting colour and shading", meta["timings"])

    def test_processing_time_pause_resume_restart(self):
        clock = u.ProcessingClock()
        job = dict(id="one", elapsed_seconds=0.0)
        with patch.object(u.time, "monotonic", return_value=100):
            clock.start(job)
        with patch.object(u.time, "monotonic", return_value=103.5):
            self.assertEqual(clock.elapsed(job), 3.5)
            clock.checkpoint(job)
        with patch.object(u.time, "monotonic", return_value=107):
            clock.checkpoint(job, stop=True)
        with patch.object(u.time, "monotonic", return_value=900):
            self.assertEqual(clock.elapsed(job), 7)
            clock.start(job)
        with patch.object(u.time, "monotonic", return_value=902):
            clock.checkpoint(job, stop=True)
        self.assertEqual(job["elapsed_seconds"], 9)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            u.save_json(root / "one/job.json", dict(job, state="running", message="Interrupted"))
            studio = u.Studio(root)
            try:
                loaded = studio.snapshot()[0]
                self.assertEqual(loaded["elapsed_seconds"], 9)
                self.assertEqual(loaded["state"], "paused")
            finally:
                studio.shutdown()
                studio.thread.join(5)
        self.assertEqual(u.duration_text(3661.9), "1:01:01")
        self.assertEqual(u.duration_text(59.9), "00:59")

    def test_comparison_navigation_fractional_and_edges(self):
        for image in [(800, 600), (600, 800), (2400, 300)]:
            for viewport in [(1000, 700), (400, 900)]:
                for zoom in [1, 2.5, 4, 8, 16]:
                    for centre in [(0, 0), (0.5, 0.5), (1, 1)]:
                        (x, y, w, h), (l, t, r, b) = u.comparison_view(
                            image, viewport, zoom, centre
                        )
                        self.assertTrue(0 <= l < r <= 1 and 0 <= t < b <= 1)
                        self.assertTrue(
                            x >= 0 and y >= 0 and x + w <= viewport[0] and y + h <= viewport[1]
                        )
                        scale = min(viewport[0] / image[0], viewport[1] / image[1]) * zoom
                        self.assertAlmostEqual(w / ((r - l) * image[0]), scale, delta=0.02)
        full, rect = u.comparison_view((800, 600), (1000, 700), 1)
        self.assertEqual(rect, (0.0, 0.0, 1.0, 1.0))
        _, rect = u.comparison_view((800, 600), (800, 600), 4, (1, 1))
        self.assertEqual(rect, (0.75, 0.75, 1.0, 1.0))

    def test_comparison_uses_output_pixels_and_bicubic_alignment(self):
        import numpy as np

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            folder = Path(folder)
            source = folder / "input.png"
            y, x = np.mgrid[:32, :40]
            original = Image.fromarray(np.uint8(np.stack([x * 6, y * 7, (x + y) * 3], -1)))
            original.save(source)
            output = original.resize((160, 128), Image.Resampling.NEAREST)
            output.save(folder / "result.png")
            u.save_json(folder / "request.json", dict(input=str(source)))
            dest = folder / "view"
            rect = (0.25, 0.25, 0.5, 0.5)
            size = [40, 32]
            for bicubic in [False, True]:
                u.comparison_region(
                    dict(
                        folder=str(folder),
                        destination=str(dest),
                        rect=rect,
                        size=size,
                        bicubic=bicubic,
                    )
                )
                with Image.open(dest / "after.png") as im:
                    self.assertEqual(im.tobytes(), output.crop((40, 32, 80, 64)).tobytes())
                expected = u.comparison_crop(
                    original.resize(output.size, Image.Resampling.BICUBIC) if bicubic else original,
                    size,
                    rect,
                    Image.Resampling.LANCZOS if bicubic else Image.Resampling.BICUBIC,
                )
                with Image.open(dest / "before.png") as im:
                    self.assertEqual(im.tobytes(), expected.tobytes())
            with self.assertRaises(ValueError):
                u.comparison_region(
                    dict(folder=str(folder), destination=str(dest), rect=rect, size=[10000, 10000])
                )

    def test_xray_tone_and_blend(self):
        import numpy as np

        y = np.linspace(0, 255, 10001)
        z = u.xray_highlights(y)
        self.assertTrue(np.all(np.diff(z) >= 0))
        self.assertEqual(z[0], 0)
        self.assertEqual(z[-1], 255)
        self.assertLess(u.xray_highlights(np.array([204.0]))[0], 204)
        self.assertGreater(u.xray_highlights(np.array([251.0]))[0], 251)
        np.testing.assert_allclose(
            u.xray_mix(
                np.full((24, 24, 3), 100.0), np.full((24, 24, 3), 200.0), np.full((24, 24, 3), 50.0)
            ),
            100.0,
            atol=1e-9,
        )
        compressed = u.xray_contrast(y)
        self.assertTrue(np.all(np.diff(compressed) > 0))
        self.assertAlmostEqual(compressed[0], 0)
        self.assertAlmostEqual(compressed[-1], 255)
        self.assertLess(np.max(np.abs(compressed - y)), 1.7)
        self.assertLess(u.xray_contrast(np.array([64.0]))[0], 64)
        self.assertGreater(u.xray_contrast(np.array([192.0]))[0], 192)

    def test_xray_middle_frequency_isolation(self):
        import numpy as np

        y, x = np.mgrid[:192, :192]
        base = np.full((192, 192, 3), 128.0)
        amplitudes = {}
        for wavelength in [2, 16, 512]:
            donor = base + np.cos(x * 2 * np.pi / wavelength)[..., None] * 20
            delta = u.xray_mix(base, donor, base) - base
            amplitudes[wavelength] = float(np.std(delta[32:-32, 32:-32]))
            np.testing.assert_allclose(delta, 0.60 * 0.12 * u.xray_middle(donor - base), atol=1e-6)
        self.assertGreater(amplitudes[16], 10 * amplitudes[2])
        self.assertGreater(amplitudes[16], 10 * amplitudes[512])

    def test_xray_shared_laplacian_and_unchanged_other_donors(self):
        import numpy as np

        y, x = np.mgrid[:24, :24]
        im = Image.fromarray(np.uint8(np.stack([70 + x * 3, 90 + y * 3, 100 + x + y], -1)))
        captured = []
        original = u.xray_restore_grain
        lap = u.local_laplacian

        def capture(source_rgb, base, donors, mask):
            captured.append(
                (source_rgb.copy(), base.copy(), [a.copy() for a in donors], mask.copy())
            )
            return original(source_rgb, base, donors, mask)

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            with (
                patch.object(u, "xray_restore_grain", side_effect=capture),
                patch.object(u, "local_laplacian", wraps=lap) as ll,
            ):
                u.detail_patch(im, folder, mode="xray")
                self.assertEqual([c.kwargs["bins"] for c in ll.call_args_list], [80])
        source_rgb, base, donors, mask = captured[0]
        np.testing.assert_array_equal(source_rgb, np.asarray(u.dcci_four(im), float))
        np.testing.assert_array_equal(donors[0], source_rgb @ u.LUMA)
        np.testing.assert_array_equal(donors[2], lap(donors[0], bins=80))
        for selected in [donors, [donors[0], donors[2], donors[3]]]:
            ordinary, ordinary_meta = u.restore_grain(base, selected, mask)
            with patch.object(
                u, "xray_gcs_grain_guard", side_effect=lambda sharp, reference: sharp
            ):
                grain, delta, meta = u.xray_restore_grain(source_rgb, base, selected, mask)
            np.testing.assert_allclose(grain, ordinary, atol=1e-12, rtol=0)
            for key in ["weights", "donor_gains", "grain_soft_cap", "grain_gain"]:
                self.assertEqual(meta[key], ordinary_meta[key])
            self.assertEqual(meta["laplacian_refinement"]["laplacian_references"], 80)
            np.testing.assert_allclose(u.xray_grain_cells(delta).mean((1, 3)), 0, atol=1e-10)

    def test_xray_grain_support_chroma_and_gamut(self):
        import numpy as np

        y, x = np.mgrid[:32, :32]
        fine = 4 * np.sin(x * 2.1) * np.cos(y * 1.8)
        rgb = np.stack([100 + fine, 130 + 0.7 * fine, 145 - 0.3 * fine], -1)
        lum = rgb @ u.LUMA
        support = ((x + y) % 3 != 0).astype(float)
        gate = np.ones((32, 32))
        # Controlled support exercises the grain boundary independently of the detector.
        with patch.object(u, "speckles", side_effect=lambda a, scale: (a, support, {})):
            grain, delta, _ = u.xray_restore_grain(rgb, rgb, [lum] * 4, gate)
            self.assertGreater(float(np.max(np.abs(delta))), 0.001)
            np.testing.assert_array_equal(delta[support == 0], 0)
            np.testing.assert_allclose(u.xray_grain_cells(delta).mean((1, 3)), 0, atol=1e-10)
            base = rgb.copy()
            base[4, 4] = [0, 255, 128]
            result = u.xray_apply_grain(base, delta)
            self.assertGreaterEqual(float(result.min()), 0)
            self.assertLessEqual(float(result.max()), 255)
            np.testing.assert_array_equal(result[support == 0], base[support == 0])
            np.testing.assert_allclose(
                u.xray_grain_cells(result - base).mean((1, 3)), 0, atol=1e-10
            )
            self.assertLessEqual(float(np.abs(result - base).max()), 3.0)
            grey = np.repeat(lum[..., None], 3, axis=2)
            _, neutral, _ = u.xray_restore_grain(grey, grey, [lum] * 4, gate)
            np.testing.assert_allclose(neutral[..., 0], neutral[..., 1], atol=1e-10)
            np.testing.assert_allclose(neutral[..., 1], neutral[..., 2], atol=1e-10)
            _, empty, _ = u.xray_restore_grain(rgb, rgb, [lum] * 4, np.zeros_like(gate))
            np.testing.assert_array_equal(empty, 0)
        with self.assertRaisesRegex(ValueError, "4x field"):
            u.xray_grain_cells(np.zeros((31, 32, 3)))

    def test_xray_gcs_grain_periodicity_guard(self):
        import numpy as np

        y, x = np.mgrid[:64, :64]
        for size in (1, 2, 4):
            checker = 8.0 * (2 * ((x // size + y // size) % 2) - 1)
            guarded = u.xray_gcs_grain_guard(checker, np.zeros_like(checker))
            self.assertLess(np.mean(abs(guarded[8:-8, 8:-8])), 0.3 * 8)
            np.testing.assert_allclose(u.xray_gcs_grain_guard(checker, checker), checker, atol=1e-8)
        line = np.broadcast_to(8.0 * (2 * (x[0] % 2) - 1), x.shape)
        np.testing.assert_array_equal(u.xray_gcs_grain_guard(line, np.zeros_like(line)), line)
        flat = np.zeros((16, 16))
        np.testing.assert_array_equal(u.xray_gcs_grain_guard(flat, flat), flat)
        # The guard is a convex donor/reference blend and cannot create excursions.
        ref = 3 * np.sin(x * 0.7) * np.cos(y * 0.9)
        out = u.xray_gcs_grain_guard(checker, ref)
        self.assertTrue(np.all(out >= np.minimum(checker, ref) - 1e-10))
        self.assertTrue(np.all(out <= np.maximum(checker, ref) + 1e-10))

    def test_xray_overshoot_guards(self):
        import numpy as np

        base = np.full((16, 16, 3), 110.0)
        delta = np.tile(np.array([-3.0, 3.0]), (16, 8))[..., None] * np.ones(3)
        np.testing.assert_array_equal(u.xray_apply_grain(base, delta), base)
        base[:, 8:] = 180
        lo, hi = u.xray_local_envelope(base)
        out = u.xray_apply_grain(base, delta)
        self.assertTrue(np.all(out >= lo) and np.all(out <= hi))
        np.testing.assert_allclose(u.xray_grain_cells(out - base).mean((1, 3)), 0, atol=1e-10)
        np.testing.assert_array_equal(u.xray_guard_gcs(base, (lo, hi)), base)
        donor = hi + 8
        np.testing.assert_allclose(u.xray_guard_gcs(donor, (lo, hi)), hi + 6)
        # A diagonal step exercises fractional Jinc phases and bright/dark ringing.
        y, x = np.mgrid[:10, :10]
        src = np.repeat(np.where(x + y > 9, 220, 40)[..., None], 3, axis=2).astype(np.uint8)
        rendered = u.xray_jinc4(Image.fromarray(src))
        self.assertGreaterEqual(rendered.min(), 39)
        self.assertLessEqual(rendered.max(), 221)
        np.testing.assert_allclose(rendered[..., 0], rendered[..., 1], atol=1e-10)

    def test_xray_selective_white_point(self):
        import numpy as np

        y = np.linspace(0, 255, 10001)
        lifted = u.xray_whites(y, strength=0.31)
        original = u.xray_whites(y)
        self.assertTrue(np.all(np.diff(lifted) > 0))
        self.assertGreater(np.max(lifted - original), 0)
        self.assertLess(np.max(lifted - original), 1.5)
        self.assertEqual(lifted[-1], 255)
        np.testing.assert_array_equal(lifted[y <= 127.5], y[y <= 127.5])
        fine = np.array([-8.0, 0.0, 8.0, 8.0])
        out = u.xray_grain_whites(fine, np.array([240, 240, 80, 240]))
        np.testing.assert_array_equal(out[:3], fine[:3])
        self.assertAlmostEqual(out[3], 8.32)
        with patch.object(u, "xray_donor_tone", side_effect=lambda a, **kw: a) as tone:
            u.xray_layer_lighting(np.zeros((16, 16, 3)), np.zeros((16, 16, 3)))
            self.assertEqual(tone.call_args_list[0].kwargs, {})
            self.assertEqual(tone.call_args_list[1].kwargs["whites_strength"], 0.31)

    def test_xray_whites_lift(self):
        import numpy as np

        y = np.linspace(0, 255, 10001)
        lifted = u.xray_whites(y)
        self.assertTrue(np.all(np.diff(lifted) > 0))
        np.testing.assert_array_equal(lifted[y <= 127.5], y[y <= 127.5])
        self.assertEqual(lifted[-1], 255)
        self.assertGreater(u.xray_whites(np.array([224.0]))[0], 226.0)
        self.assertLess(np.max(lifted - y), 13.0)
        tone = u.xray_whites(u.xray_contrast(u.xray_highlights(y)))
        self.assertTrue(np.all(np.diff(tone) >= 0))

    def test_xray_lighting_is_confined_to_weighted_donors(self):
        import numpy as np

        g = np.full((24, 24, 3), 192.0)
        j = np.full((24, 24, 3), 224.0)
        with patch.object(u, "xray_donor_tone", side_effect=lambda a, **kwargs: a + 10):
            np.testing.assert_allclose(u.xray_layer_lighting(g, j), 2.75 * (0.12 + 0.06) * 10)
        # A flat field follows the same donor-weighted path, not the previous full-image curve.
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            low = Image.new("RGB", (6, 6), (192, 192, 192))
            out, _ = u.detail_patch(low, folder, mode="xray")
            expected = u.image(g + u.xray_layer_lighting(g, g))
            self.assertEqual(out.tobytes(), expected.tobytes())
        # No extra deblur or main-image tone calls in the non-flat path.
        y, x = np.mgrid[:16, :16]
        low = Image.fromarray(np.uint8(np.stack([100 + x * 3, 110 + y * 3, 120 + x + y], -1)))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            original = u.xray_donor_tone
            with (
                patch.object(u, "xray_donor_tone", wraps=original) as tone,
                patch.object(
                    u, "xray_directional_deblur", side_effect=AssertionError("No extra deblur")
                ),
            ):
                _, meta = u.detail_patch(low, folder, mode="xray", use_fitted=False)
                self.assertEqual(tone.call_count, 2)
                self.assertFalse(meta["xray"]["global_tone"])
                self.assertEqual(meta["xray"]["detail_strength"], 0.60)

    def test_xray_shadow_and_jinc_contrast(self):
        import numpy as np

        y = np.linspace(0, 255, 10001)
        lift = u.xray_shadow_visibility(y)
        self.assertTrue(np.all(np.diff(lift) > 0))
        self.assertEqual(lift[0], 0)
        np.testing.assert_array_equal(lift[y >= 150], y[y >= 150])
        self.assertLessEqual(np.max(lift - y), 2.0)
        full = u.xray_contrast(y) - y
        gentle = u.xray_contrast(y, 0.25) - y
        np.testing.assert_allclose(gentle, 0.25 * full, atol=1e-12)
        x = np.arange(64)
        base = np.full((32, 64, 3), 128.0)
        donor = base + 10 * np.cos(x[None, :, None] * np.pi / 8)
        delta = u.xray_mix(base, base, donor) - base
        np.testing.assert_allclose(
            delta, 0.60 * 0.06 * 0.5 * u.xray_middle(donor - base), atol=1e-6
        )
        for strength in [1.0, 0.25]:
            combined = u.xray_whites(
                u.xray_shadow_visibility(u.xray_contrast(u.xray_highlights(y), strength))
            )
            self.assertTrue(np.all(np.diff(combined) > 0))
            self.assertEqual(combined[0], 0)
            self.assertEqual(combined[-1], 255)

    def test_xray_jinc_rotation_and_flat(self):
        import numpy as np

        y, x = np.mgrid[:12, :14]
        a = np.uint8(np.stack([x * 17, y * 19, (x + y) * 9], -1))
        im = Image.fromarray(a)
        actual = u.xray_jinc4(im)
        rotated = np.rot90(u.xray_jinc4(im.transpose(Image.Transpose.ROTATE_90)), -1)
        self.assertLess(float(np.max(np.abs(actual - rotated))), 1e-7)
        constant = u.xray_jinc4(Image.new("RGB", (8, 8), (17, 128, 239)))
        np.testing.assert_allclose(
            constant, np.broadcast_to([17, 128, 239], constant.shape), atol=1e-8
        )

    def test_xray_short_blur_inverse(self):
        import numpy as np

        truth = u.blur(np.tile(np.where(np.arange(128) > 64, 190.0, 60.0), (64, 1)), 1.0)
        p = np.pad(truth, ((0, 0), (1, 1)), mode="edge")
        soft = 0.6 * truth + 0.2 * (p[:, :-2] + p[:, 2:])
        restored = soft + u.xray_directional_deblur(soft)
        self.assertLess(np.mean((restored - truth) ** 2), np.mean((soft - truth) ** 2))
        self.assertLessEqual(np.max(np.abs(restored - soft)), 6)

    def test_xray_pipeline_fractional_cache_and_no_fitted(self):
        import numpy as np

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            folder = Path(folder)
            y, x = np.mgrid[:24, :24]
            a = np.uint8(np.stack([70 + x * 3, 90 + y * 3, 100 + x + y], -1))
            source = folder / "in.png"
            Image.fromarray(a).save(source)
            request = dict(
                input=str(source),
                folder=str(folder / "job"),
                scale=2.5,
                mode="xray",
                svg=False,
                device="cpu",
                workers=1,
            )
            result = u.process_request(request)
            self.assertEqual(result["output_size"], [60, 60])
            self.assertEqual(result["recipe_id"], "xray-shared80-spatial-blend-v8")
            self.assertEqual(result["xray"]["laplacian_bins"], 80)
            self.assertEqual(result["xray"]["blend_scope"], "middle_band")
            self.assertEqual(result["xray"]["fine_strength"], 1.0)
            self.assertEqual(result["xray"]["gcs"], 0.12)
            self.assertEqual(result["xray"]["jinc4"], 0.06)
            digest = u.sha(folder / "job/result.png")
            u.process_request(request)
            self.assertEqual(digest, u.sha(folder / "job/result.png"))
            with self.assertRaisesRegex(ValueError, "different input or setting"):
                u.process_request(dict(request, mode="max"))
            with patch.object(
                u, "fitted_four", side_effect=AssertionError("Fitted must be skipped")
            ):
                im, meta = u.detail_patch(
                    Image.fromarray(a), folder / "no-fit", mode="xray", use_fitted=False
                )
            self.assertEqual(im.size, (96, 96))
            self.assertFalse(meta["fitted_enabled"])
            self.assertEqual(meta["grain"]["weights"], [1 / 3] * 3)

    def test_console_identity(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "source/untorn_engine.py"), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Untorn Engine 2.3.0", p.stdout)
        self.assertIn("Parham (Leon) Faraji", p.stdout)

    def test_jpeg_quality_size_and_failure_safety(self):
        import numpy as np
        from PIL import JpegImagePlugin

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            source = root / "result.png"
            pixels = np.random.default_rng(47).integers(0, 256, (96, 128, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(source)
            original = source.read_bytes()
            sizes = []
            for quality in (30, 95):
                dest = root / f"quality-{quality}.jpg"
                meta = u.jpeg_export(
                    dict(source=str(source), destination=str(dest), quality=quality, size=[128, 96])
                )
                sizes.append(meta["bytes"])
                with Image.open(dest) as im:
                    self.assertEqual(im.size, (128, 96))
                    self.assertEqual(im.format, "JPEG")
                    self.assertEqual(JpegImagePlugin.get_sampling(im), 0)
                    self.assertTrue(im.info.get("icc_profile"))
                    im.load()
            self.assertGreater(sizes[1], sizes[0])
            self.assertEqual(source.read_bytes(), original)
            dest = root / "resized.jpg"
            request = dict(source=str(source), destination=str(dest), quality=90, size=[64, 48])
            u.jpeg_export(request)
            with Image.open(dest) as im:
                self.assertEqual(im.size, (64, 48))
            previous = dest.read_bytes()
            for invalid in [
                dict(size=[65, 30]),
                dict(size=[256, 192]),
                dict(quality=101),
                dict(destination=str(source)),
            ]:
                with self.assertRaises(ValueError):
                    u.jpeg_export(dict(request, **invalid))
                self.assertEqual(dest.read_bytes(), previous)
            with patch.object(
                Image.Image, "save", side_effect=OSError("simulated encoder failure")
            ):
                with self.assertRaises(OSError):
                    u.jpeg_export(request)
            self.assertEqual(dest.read_bytes(), previous)
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_dcci_output_alignment_and_patch_boundaries(self):
        import numpy as np

        y, x = np.mgrid[:19, :23]
        pixels = np.uint8(np.stack([x * 7 + y * 2, y * 9 + x, (x > y) * 180 + 30], -1))
        im = Image.fromarray(pixels)
        for factor in (2, 4):
            actual = u.dcci_output(im, (23 * factor, 19 * factor), (0, 0, 23, 19), block=16)
            # Independently construct the global node lattice, then select
            # exactly the pixel-centred destinations. No resize kernel used.
            grid = np.pad(pixels.astype(float) / 255, ((12, 12), (12, 12), (0, 0)), mode="edge")
            for _ in range(int(np.log2(factor)) + 1):
                grid = u.dcci_twice(grid)
            ix = (2 * np.arange(23 * factor) + 1 - factor + 24 * factor).astype(int)
            iy = (2 * np.arange(19 * factor) + 1 - factor + 24 * factor).astype(int)
            expected = np.clip(grid[iy[:, None], ix[None, :]] * 255, 0, 255)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
            alternate = u.dcci_output(im, (23 * factor, 19 * factor), (0, 0, 23, 19), block=160)
            np.testing.assert_allclose(actual, alternate, atol=2e-5, rtol=0)
        # Fractional grid registration remains independent of patch origins.
        a = u.dcci_output(im, (79, 61), (0, 0, 23, 19), block=16)
        b = u.dcci_output(im, (79, 61), (0, 0, 23, 19), block=160)
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=0)
        # Crossing 8x must not switch abruptly to a different detail field.
        centre = u.dcci_output(im, (46, 38), (0, 0, 23, 19))
        for ratio in (2 - 1e-6, 2 + 1e-6):
            nearby = u.dcci_output(im, (46, 38), (0, 0, 46 / ratio, 38 / ratio))
            np.testing.assert_allclose(nearby, centre, atol=0.01, rtol=0)
        just_above = u.dcci_output(im, (23, 19), (0, 0, 23 / (1 + 1e-7), 19 / (1 + 1e-7)))
        np.testing.assert_allclose(just_above, pixels.astype(float), atol=0.01, rtol=0)

    def test_dcci_output_pipeline_and_legacy_resume(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            source = root / "in.png"
            Image.new("RGB", (16, 17), (83, 119, 151)).save(source)
            for scale in (4, 5.5, 8, 16):
                job = root / str(scale)
                request = dict(
                    input=str(source),
                    folder=str(job),
                    scale=scale,
                    svg=False,
                    mode="fast",
                    device="cpu",
                    workers=1,
                )
                result = u.process_request(request)
                with Image.open(job / "result.png") as im:
                    self.assertEqual(im.size, u.dimensions((16, 17), scale))
                    self.assertEqual(im.getextrema(), ((83, 83), (119, 119), (151, 151)))
                if scale > 4:
                    self.assertIn("DCCI enlargement", result["sampling"])
                    receipts = {
                        p: p.stat().st_mtime_ns for p in (job / "tiles").glob("*/receipt.json")
                    }
                    signature = u.read_json(job / "signature.json")
                    signature.pop("enlargement")
                    u.save_json(job / "signature.json", signature)
                    u.process_request(request)
                    self.assertEqual(receipts, {p: p.stat().st_mtime_ns for p in receipts})
                else:
                    self.assertNotIn("enlargement", result)

    def test_parallel_order_resume_and_resource_fallback(self):
        import numpy as np

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            source = root / "in.png"
            y, x = np.mgrid[:48, :208]
            Image.fromarray(np.uint8(np.stack([x % 256, y * 4, (x + y) % 256], -1))).save(source)
            calls = []
            lock = threading.Lock()
            failed = [False]

            def render(low, cache, *args, **kwargs):
                with lock:
                    calls.append(Path(cache).parent.parent.name)
                if (
                    "fallback" in str(cache)
                    and threading.current_thread().name.startswith("reconstruct")
                    and not failed[0]
                ):
                    failed[0] = True
                    raise MemoryError("simulated resource exhaustion")
                time.sleep(0.05)
                return low.resize((low.width * 4, low.height * 4)), {"fixture": True}

            with (
                patch.object(u, "detail_patch", side_effect=render),
                patch.object(u, "memory_budget", return_value=(16 * 1024**3, 8 * 1024**3)),
            ):
                results = []
                for name, workers in [("serial", 1), ("parallel", 2), ("fallback", 2)]:
                    request = dict(
                        input=str(source),
                        folder=str(root / name),
                        scale=2.5,
                        svg=False,
                        mode="max",
                        device="cpu",
                        workers=workers,
                    )
                    u.process_request(request)
                    with Image.open(root / name / "result.png") as im:
                        results.append(np.asarray(im).copy())
                    before = len(calls)
                    u.process_request(dict(request, workers=1))
                    self.assertEqual(len(calls), before)
                self.assertTrue(np.array_equal(results[0], results[1]))
                self.assertTrue(np.array_equal(results[0], results[2]))
                self.assertTrue((root / "fallback/parallel-fallback.json").exists())

    def test_gpu_failure_recomputes_on_cpu(self):
        import numpy as np

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            a = np.random.default_rng(42).uniform(0, 255, (384, 384))
            expected = u.local_laplacian(a, bins=80)
            accelerator = u.Acceleration(Path(folder) / "job")
            accelerator.plan = {
                "backend": "opencl-fp64",
                "minimum_work": 1,
                "device": {"index": 0, "name": "test GPU"},
            }
            with patch.object(
                accelerator, "execute", side_effect=RuntimeError("simulated driver failure")
            ) as execute:
                actual = accelerator.laplacian(a, 80, Path(folder) / "tile")
                again = accelerator.laplacian(a, 80, Path(folder) / "tile")
                self.assertEqual(execute.call_count, 1)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertTrue(np.array_equal(again, expected))
            self.assertEqual(accelerator.calls, 0)
            self.assertEqual(len(accelerator.fallbacks), 1)

    def test_self_examples_consistency_determinism_and_boundaries(self):
        import numpy as np

        y, x = np.mgrid[:48, :64]
        signal = 120 + 35 * np.sin(0.012 * x * x + 0.014 * y * y)
        im = Image.fromarray(np.uint8(np.repeat(signal[..., None], 3, axis=2)))
        fields, meta = u.xray_self_details(im)
        again, _ = u.xray_self_details(im)
        self.assertTrue(all(n > 0 for n in meta["accepted"]))
        for a, b in zip(fields, again):
            np.testing.assert_array_equal(a, b)
            self.assertLess(np.max(np.abs(a.reshape(48, 4, 64, 4).mean((1, 3)))), 1e-5)
        cropped, _ = u.xray_self_details(im.crop((8, 8, 56, 40)))
        for a, b in zip(fields, cropped):
            np.testing.assert_allclose(a[96:128, 96:160], b[64:96, 64:128], atol=1e-5)
        base = np.full((192, 256, 3), 120.0)
        result, _ = u.xray_self_average(im, base)
        np.testing.assert_allclose(
            result, base + u.xray_self_grain(im, np.ones((192, 256)))[0][..., None], atol=1e-6
        )
        for colour in [(0, 0, 0), (255, 255, 255), (0, 80, 255), (83, 119, 151)]:
            rgb = np.zeros_like(base) + colour
            out, _ = u.xray_self_average(im, rgb)
            self.assertTrue(np.isfinite(out).all())
            self.assertGreaterEqual(out.min(), -1e-6)
            self.assertLessEqual(out.max(), 255 + 1e-6)
            np.testing.assert_allclose(
                (out @ u.LUMA).reshape(48, 4, 64, 4).mean((1, 3)),
                (rgb @ u.LUMA).reshape(48, 4, 64, 4).mean((1, 3)),
                atol=1e-5,
            )

    def test_dcci_vibrance_and_neutral_colours(self):
        import numpy as np

        rgb = np.array([[[120.0, 125.0, 130.0], [128.0, 128.0, 128.0], [0.0, 0.0, 255.0]]])
        change = u.xray_self_vibrance(rgb)
        np.testing.assert_allclose(change @ u.LUMA, 0, atol=1e-10)
        np.testing.assert_array_equal(change[0, 1:], 0)
        self.assertGreater(change[0, 0, 2], 0)
        self.assertLess(change[0, 0, 0], 0)
        low = Image.new("RGB", (7, 5), (120, 125, 130))
        base = np.zeros((20, 28, 3)) + [120, 125, 130]
        out, meta = u.xray_self_average(low, base)
        np.testing.assert_array_equal(out, base)
        vibrant = u.xray_dcci_vibrance(base, base)
        np.testing.assert_allclose(vibrant, base + u.xray_self_vibrance(base), atol=1e-10)
        self.assertEqual(meta["vibrance"]["layer"], "dcci")

    def test_self_grain_isolation_and_dcci_colour_source(self):
        import numpy as np

        low = Image.new("RGB", (32, 32), (100, 110, 120))
        y, x = np.mgrid[:128, :128]
        fine = np.cos(2 * np.pi * x / 2)
        broad = np.cos(2 * np.pi * x / 64)

        def evaluate(field, mask):
            with patch.object(
                u, "xray_self_details", return_value=([field.copy(), field.copy()], {})
            ):
                return u.xray_self_grain(low, mask)[0]

        mask = np.ones((128, 128))
        a = evaluate(fine, mask)
        b = evaluate(broad, mask)
        self.assertGreater(np.std(a), 20 * np.std(b))
        expected = (fine - u.blur(fine, 0.8)) / 3
        cells = expected.reshape(32, 4, 32, 4)
        cells -= cells.mean((1, 3), keepdims=True)
        np.testing.assert_allclose(a, expected, atol=1e-10)
        np.testing.assert_array_equal(evaluate(fine, np.zeros_like(mask)), 0)
        # A coloured final base cannot give either exemplar branch vibrance.
        base = np.zeros((128, 128, 3)) + [100, 120, 140]
        grey = np.zeros_like(base) + 120
        np.testing.assert_array_equal(u.xray_dcci_vibrance(base, grey), base)
        self.assertGreater(np.max(np.abs(u.xray_dcci_vibrance(base, base) - base)), 0)

    def test_self_average_fractional_job_and_cache(self):
        import numpy as np

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            folder = Path(folder)
            y, x = np.mgrid[:24, :24]
            a = np.uint8(np.stack([80 + x * 3, 90 + y * 3, 100 + x + y], -1))
            source = folder / "in.png"
            Image.fromarray(a).save(source)
            request = dict(
                input=str(source),
                folder=str(folder / "job"),
                scale=2.5,
                mode="xray-plus",
                svg=False,
                device="cpu",
                workers=1,
            )
            result = u.process_request(request)
            self.assertEqual(result["output_size"], [60, 60])
            self.assertEqual(result["recipe_id"], "xray-shared80-spatial-blend-v8")
            self.assertEqual(result["self_examples"]["weights"], [2 / 3, 1 / 6, 1 / 6])
            self.assertEqual(result["self_examples"]["vibrance"]["strength"], 0.02)
            self.assertFalse(result["self_examples"]["full_paper_implementation"])
            before = u.sha(folder / "job/result.png")
            u.process_request(request)
            self.assertEqual(before, u.sha(folder / "job/result.png"))
            with self.assertRaisesRegex(ValueError, "different input or setting"):
                u.process_request(dict(request, mode="max"))

    def test_xray_alias_and_compatible_cache_migration(self):
        import numpy as np

        self.assertEqual(u.normalize_mode("xray-plus"), "xray")
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            folder = Path(folder)
            source = folder / "in.png"
            y, x = np.mgrid[:24, :24]
            Image.fromarray(np.uint8(np.stack([80 + 3 * x, 90 + 3 * y, 100 + x + y], -1))).save(
                source
            )
            request = dict(
                input=str(source),
                folder=str(folder / "job"),
                scale=2.5,
                mode="xray",
                device="cpu",
                workers=1,
            )
            u.process_request(request)
            stamp = folder / "job/signature.json"
            signature = u.read_json(stamp)
            u.save_json(stamp, dict(signature, mode="xray-plus"))
            receipts = {
                p: p.stat().st_mtime_ns for p in (folder / "job/tiles").glob("*/receipt.json")
            }
            result = u.process_request(dict(request, mode="xray-plus"))
            self.assertEqual(result["mode"], "xray")
            self.assertEqual(u.read_json(stamp), signature)
            self.assertEqual(receipts, {p: p.stat().st_mtime_ns for p in receipts})
            for old_recipe in [
                "xray-v9",
                "xray-self-grain-v3",
                "xray-shared80-grain-v4",
                "xray-shared80-grain-v5",
            ]:
                u.save_json(stamp, dict(signature, xray_recipe=old_recipe))
                with self.assertRaisesRegex(ValueError, "different input or setting"):
                    u.process_request(request)

    def test_public_command_contract(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "source/untorn_engine.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("{fast,max,xray}", result.stdout)
        self.assertNotIn("xray-plus", result.stdout)
        self.assertIn("--desktop", result.stdout)
        source = (ROOT / "source/untorn_engine.py").read_text(encoding="utf-8")
        self.assertNotIn("PHOTO RESTORATION", source)
        self.assertNotIn("X-Ray+", source)

        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            source_image = root / "input.png"
            Image.new("RGB", (16, 16), (90, 120, 150)).save(source_image)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "source/untorn_engine.py"),
                    str(source_image),
                    "--scale",
                    "1.5",
                    "--device",
                    "cpu",
                    "--workers",
                    "1",
                    "--output",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            results = list((root / "output").rglob("result.json"))
            self.assertEqual(len(results), 1)
            self.assertEqual(u.read_json(results[0])["mode"], "max")


if __name__ == "__main__":
    unittest.main()
