from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic_system.mosaic_io import RollingMosaicCache, build_mosaic_pair


class MosaicIOTests(unittest.TestCase):
    def test_mosaic_dimensions_and_tile_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(16):
                path = root / f"tile_{index:02d}.png"
                Image.fromarray(np.full((8, 8), index, np.uint8), "L").save(path)
                paths.append(path)

            lr, hr = build_mosaic_pair(
                paths,
                grid_size=4,
                tile_size=(8, 8),
                scale_factor=4,
            )
            self.assertEqual(lr.shape, (8, 8))
            self.assertEqual(hr.shape, (32, 32))
            self.assertEqual(int(hr[4, 4]), 0)
            self.assertEqual(int(hr[4, 12]), 1)
            self.assertEqual(int(hr[28, 28]), 15)

    def test_rolling_cache_keeps_only_current_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            cache = RollingMosaicCache(cache_dir, capacity=2, total_items=4)

            def builder(index):
                return (
                    np.full((2, 2), index, np.uint8),
                    np.full((4, 4), index, np.uint8),
                )

            cache.get(0, builder)
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 2)
            cache.get(2, builder)
            self.assertEqual(
                sorted(path.name for path in cache_dir.glob("*.npz")),
                ["mosaic_000002.npz", "mosaic_000003.npz"],
            )
            cache.close()
            self.assertFalse(cache_dir.exists())


if __name__ == "__main__":
    unittest.main()

