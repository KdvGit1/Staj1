from __future__ import annotations

import unittest
from pathlib import Path

from mosaic_system.manifest import MosaicManifestPlanner


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.paths = [
            Path(f"C:/dataset/video-{index % 20:02d}-frame-{index:06d}.jpg")
            for index in range(50)
        ]

    def test_epoch_uses_each_path_at_most_once(self):
        planner = MosaicManifestPlanner(self.paths, seed=123)
        groups = planner.build_epoch(0)
        used = [path for group in groups for path in group.image_paths]
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(used), 48)
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(planner.leftovers_per_epoch, 2)

    def test_cross_epoch_groups_do_not_repeat(self):
        planner = MosaicManifestPlanner(self.paths, seed=123)
        epoch0 = {group.signature for group in planner.build_epoch(0)}
        epoch1 = {group.signature for group in planner.build_epoch(1)}
        epoch2 = {group.signature for group in planner.build_epoch(2)}
        self.assertTrue(epoch0.isdisjoint(epoch1))
        self.assertTrue(epoch0.isdisjoint(epoch2))
        self.assertTrue(epoch1.isdisjoint(epoch2))

    def test_same_seed_is_reproducible(self):
        one = MosaicManifestPlanner(self.paths, seed=999).build_epoch(2)
        two = MosaicManifestPlanner(self.paths, seed=999).build_epoch(2)
        self.assertEqual(
            [group.signature for group in one],
            [group.signature for group in two],
        )


if __name__ == "__main__":
    unittest.main()

