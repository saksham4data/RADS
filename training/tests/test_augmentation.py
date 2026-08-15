import unittest
from torchvision import transforms

from training.configs.config import TrainingConfig
from training.datasets.transforms import get_train_transforms, get_val_transforms
from training.datasets.dataloader import create_dataloaders, create_test_dataloader

class TestAugmentationConfig(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig()
        
    def test_config_parsing(self):
        # By default in config.py, augmentation is disabled
        self.assertFalse(self.config.augmentation_enabled)
        self.assertEqual(self.config.augmentation_config["preset"], "none")
        
        # Manually enable
        self.config._raw["data"]["augmentation"]["enabled"] = True
        self.config._raw["data"]["augmentation"]["preset"] = "conservative_v1"
        self.assertTrue(self.config.augmentation_enabled)

    def test_transforms_builder_deterministic(self):
        # When disabled, train transform should NOT have random ops
        self.config._raw["data"]["augmentation"]["enabled"] = False
        transform = get_train_transforms(self.config.image_size, self.config.augmentation_config)
        
        # Check for RandomHorizontalFlip (should not be there)
        has_random = any(isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation, transforms.ColorJitter)) for t in transform.transforms)
        self.assertFalse(has_random)

    def test_transforms_builder_stochastic(self):
        # When enabled with conservative_v1, train transform SHOULD have random ops
        self.config._raw["data"]["augmentation"]["enabled"] = True
        self.config._raw["data"]["augmentation"]["preset"] = "conservative_v1"
        transform = get_train_transforms(self.config.image_size, self.config.augmentation_config)
        
        has_flip = any(isinstance(t, transforms.RandomHorizontalFlip) for t in transform.transforms)
        has_rotate = any(isinstance(t, transforms.RandomRotation) for t in transform.transforms)
        has_jitter = any(isinstance(t, transforms.ColorJitter) for t in transform.transforms)
        
        self.assertTrue(has_flip)
        self.assertTrue(has_rotate)
        self.assertTrue(has_jitter)

    def test_eval_transforms_always_deterministic(self):
        transform = get_val_transforms(self.config.image_size)
        has_random = any(isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation, transforms.ColorJitter)) for t in transform.transforms)
        self.assertFalse(has_random)

if __name__ == '__main__':
    unittest.main()
