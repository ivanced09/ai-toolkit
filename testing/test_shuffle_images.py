import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from toolkit.config import get_config
from toolkit.config_modules import TrainConfig
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch


class DatasetModel:
    # The image loader only needs model metadata for these uncached CPU datasets.
    vae = None
    unet = None
    adapter = None
    load_rgba = False
    use_raw_control_images = False
    encode_control_in_text_embeddings = False
    te_padding_side = "right"

    def get_bucket_divisibility(self):
        return 16

    def get_latent_space_version(self):
        return "test"

    def get_text_embedding_space_version(self):
        return "test"


class TestShuffleImages(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def make_dataset(self, name, images, **options):
        folder = self.root / name
        folder.mkdir()
        for filename, size in images:
            path = folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", size).save(path)
        return {"folder_path": str(folder), "resolution": 64, "num_workers": 0, **options}

    def make_loader(self, datasets, **options):
        return get_dataloader_from_datasets(datasets, sd=DatasetModel(), **options)

    @staticmethod
    def filenames(batches):
        return [[Path(item.path).name for item in batch.file_items] for batch in batches]

    def test_omitted_and_enabled_preserve_shuffling(self):
        dataset = self.make_dataset("images", [(f"{i}.jpg", (64, 64)) for i in range(12)])
        self.assertIs(TrainConfig().shuffle_images, True)
        self.assertIs(TrainConfig(shuffle_images=True).shuffle_images, True)
        sequences = []
        for options in ({}, {"shuffle_images": True}):
            with self.subTest(options=options), torch.random.fork_rng():
                random_state = random.getstate()
                try:
                    random.seed(42)
                    torch.manual_seed(42)
                    loader = self.make_loader([dataset], batch_size=2, **options)
                    self.assertIsInstance(loader.sampler, RandomSampler)
                    bucket = next(iter(loader.dataset.datasets[0].buckets.values()))
                    self.assertNotEqual(bucket.file_list_idx, list(range(12)))
                    first = self.filenames(loader)
                    trigger_dataloader_setup_epoch(loader)
                    second = self.filenames(loader)
                    self.assertNotEqual(first, second)
                    self.assertCountEqual(
                        [name for batch in first for name in batch],
                        [f"{i}.jpg" for i in range(12)],
                    )
                    sequences.append((first, second))
                finally:
                    random.setstate(random_state)
        self.assertEqual(sequences[0], sequences[1])

    def test_serialized_config_orders_single_bucket_across_epochs(self):
        dataset = self.make_dataset(
            "images", [(f"{i}.jpg", (64, 64)) for i in (11, 3, 10, 1, 20, 2)],
            num_workers=1, batch_load_threads=2,
        )
        config_path = self.root / ".job_config.json"
        config_path.write_text(json.dumps({
            "job": "extension",
            "config": {
                "name": "image-order",
                "process": [{"train": {"batch_size": 2, "shuffle_images": False}, "datasets": [dataset]}],
            },
        }))
        process = get_config(str(config_path))["config"]["process"][0]
        train = TrainConfig(**process["train"])
        self.assertIs(train.shuffle_images, False)
        loader = self.make_loader(
            process["datasets"], batch_size=train.batch_size, shuffle_images=train.shuffle_images,
        )
        self.assertIsInstance(loader.sampler, SequentialSampler)
        expected = [["1.jpg", "2.jpg"], ["3.jpg", "10.jpg"], ["11.jpg", "20.jpg"]]
        self.assertEqual(self.filenames(loader), expected)
        trigger_dataloader_setup_epoch(loader)
        self.assertEqual(self.filenames(loader), expected)

    def test_multiple_buckets_preserve_compatible_batches(self):
        dataset = self.make_dataset("images", [
            ("6.jpg", (64, 64)), ("5.jpg", (128, 64)), ("4.jpg", (64, 64)),
            ("3.jpg", (128, 64)), ("2.jpg", (128, 64)), ("1.jpg", (64, 64)),
        ])
        expected = [["1.jpg", "4.jpg"], ["2.jpg", "3.jpg"], ["5.jpg", "5.jpg"], ["6.jpg", "6.jpg"]]
        loader = self.make_loader([dataset], batch_size=2, shuffle_images=False)
        for _ in range(2):
            batches = list(loader)
            self.assertEqual(self.filenames(batches), expected)
            self.assertEqual(len({tuple(batch.tensor.shape[-2:]) for batch in batches}), 2)
            for batch in batches:
                self.assertEqual(batch.tensor.shape[0], 2)
                self.assertEqual(len({tuple(item.tensor.shape) for item in batch.file_items}), 1)
            trigger_dataloader_setup_epoch(loader)
        single = self.make_loader([dataset], batch_size=1, shuffle_images=False)
        self.assertEqual(self.filenames(single), [[f"{i}.jpg"] for i in range(1, 7)])

    def test_repeated_and_separate_datasets(self):
        first = self.make_dataset(
            "first", [("10.jpg", (64, 64)), ("2.jpg", (64, 64))], num_repeats=2, batch_size=2,
        )
        second = self.make_dataset("second", [("1.jpg", (64, 64))], batch_size=1)
        loader = self.make_loader([first, second], batch_size=3, shuffle_images=False)
        self.assertEqual(self.filenames(loader), [["2.jpg", "2.jpg"], ["10.jpg", "10.jpg"], ["1.jpg"]])

    def test_pooled_sources_sort_together_and_keep_incompatible_settings_apart(self):
        first = self.make_dataset("first", [("10.jpg", (64, 64)), ("3.jpg", (64, 64))])
        second = self.make_dataset("second", [("2.jpg", (64, 64)), ("11.jpg", (64, 64))])
        third = self.make_dataset("third", [("1.jpg", (64, 64))], extra_values=[0.5])
        loader = self.make_loader(
            [first, second, third], batch_size=2, pool_datasets=True, shuffle_images=False,
        )
        expected = [["1.jpg", "1.jpg"], ["2.jpg", "3.jpg"], ["10.jpg", "11.jpg"]]
        self.assertEqual(self.filenames(loader), expected)
        trigger_dataloader_setup_epoch(loader)
        self.assertEqual(self.filenames(loader), expected)

    def test_unbucketed_json_sources_sort_by_filename_with_path_ties(self):
        dataset = self.make_dataset("images", [
            ("a/10.jpg", (64, 64)), ("z/2.jpg", (64, 64)), ("a/2.jpg", (64, 64)),
            ("1.jpg", (64, 64)),
        ], buckets=False)
        folder = Path(dataset.pop("folder_path"))
        manifest = self.root / "captions.json"
        manifest.write_text(json.dumps({
            str(folder / filename): {"caption": "caption"}
            for filename in ("z/2.jpg", "a/10.jpg", "a/2.jpg", "1.jpg")
        }))
        loader = self.make_loader([dict(dataset, dataset_path=str(manifest))], batch_size=2, shuffle_images=False)
        batches = list(loader)
        self.assertEqual(self.filenames(batches), [["1.jpg", "2.jpg"], ["2.jpg", "10.jpg"]])
        self.assertEqual(Path(batches[0].file_items[1].path).parent.name, "a")
        self.assertEqual(Path(batches[1].file_items[0].path).parent.name, "z")

    def test_ordered_loader_does_not_substitute_an_unreadable_file(self):
        dataset = self.make_dataset("images", [("1.jpg", (64, 64)), ("2.jpg", (64, 64))])
        loader = self.make_loader([dataset], shuffle_images=False)
        (Path(dataset["folder_path"]) / "1.jpg").unlink()
        with self.assertRaisesRegex(RuntimeError, r"Failed to load ordered training image: .*1\.jpg"):
            next(iter(loader))
