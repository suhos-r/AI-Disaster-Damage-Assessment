import os
import shutil
import unittest

import torch

from src.dataset import DisasterAssessmentDataset, generate_mock_data, get_train_transforms
from src.model import UNet
from src.utils import calculate_iou, calculate_metrics


class TestDisasterAssessmentPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./test_data"
        cls.metadata = generate_mock_data(cls.test_dir, num_samples=2)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_dataset(self):
        dataset = DisasterAssessmentDataset(self.metadata, self.test_dir, transform=get_train_transforms())
        self.assertEqual(len(dataset), 2)

        image_tensor, tabular_tensor, mask_tensor = dataset[0]
        self.assertEqual(tuple(image_tensor.shape), (6, 256, 256))
        self.assertEqual(tuple(tabular_tensor.shape), (4,))
        self.assertEqual(tuple(mask_tensor.shape), (256, 256))

    def test_model(self):
        model = UNet(in_channels=6, out_classes=5)
        dummy_input = torch.randn(2, 6, 256, 256)
        output = model(dummy_input)

        self.assertEqual(tuple(output.shape), (2, 5, 256, 256))

    def test_metrics(self):
        predictions = torch.randn(2, 5, 256, 256)
        targets = torch.randint(0, 5, (2, 256, 256))

        metrics = calculate_metrics(predictions, targets)
        self.assertIn("accuracy", metrics)
        self.assertTrue(0 <= metrics["accuracy"] <= 1)

        iou = calculate_iou(predictions, targets)
        self.assertIsInstance(iou, float)


if __name__ == "__main__":
    unittest.main()
