# import unittest
#
# import torch
#
# from dl_lib.vision.functional import load_fashion_mnist_loaders
#
#
# class TestLoadFashionMnistLoaders(unittest.TestCase):
#     def test_when_called_then_returns_ten_class_names(self):
#         # Arrange
#         expected_num_classes = 10
#
#         # Act
#         _, _, class_names = load_fashion_mnist_loaders(batch_size=4)
#
#         # Assert
#         self.assertEqual(len(class_names), expected_num_classes)
#
#     def test_when_called_then_a_train_batch_is_64x64x1(self):
#         # Arrange
#         batch_size = 4
#         expected = torch.Size([batch_size, 1, 64, 64])
#         train_loader, _, _ = load_fashion_mnist_loaders(batch_size=batch_size)
#
#         # Act
#         images, _ = next(iter(train_loader))
#
#         # Assert
#         self.assertEqual(images.shape, expected)
#
#     def test_when_called_then_a_test_batch_is_64x64x1(self):
#         # Arrange
#         batch_size = 4
#         expected = torch.Size([batch_size, 1, 64, 64])
#         _, test_loader, _ = load_fashion_mnist_loaders(batch_size=batch_size)
#
#         # Act
#         images, _ = next(iter(test_loader))
#
#         # Assert
#         self.assertEqual(images.shape, expected)
#
#     def test_when_called_then_train_and_test_are_the_standard_fashion_mnist_splits(self):
#         # Arrange
#         expected_train_size = 60000
#         expected_test_size = 10000
#
#         # Act
#         train_loader, test_loader, _ = load_fashion_mnist_loaders(batch_size=4)
#
#         # Assert
#         self.assertEqual(len(train_loader.dataset), expected_train_size)
#         self.assertEqual(len(test_loader.dataset), expected_test_size)
#
#     def test_when_called_then_class_names_match_the_datasets_label_order(self):
#         # Arrange
#         train_loader, _, class_names = load_fashion_mnist_loaders(batch_size=4)
#
#         # Act
#         expected = train_loader.dataset.classes
#
#         # Assert
#         self.assertEqual(class_names, expected)
#
#     def test_when_called_then_a_loaded_image_still_has_contrast(self):
#         # Arrange
#         min_std = 0.05
#         train_loader, _, _ = load_fashion_mnist_loaders(batch_size=4)
#
#         # Act
#         images, _ = next(iter(train_loader))
#         stds = images.std(dim=(1, 2, 3))
#
#         # Assert
#         self.assertTrue((stds > min_std).all())
#
#     def test_when_called_then_different_samples_in_a_batch_are_not_identical(self):
#         # Arrange
#         train_loader, _, _ = load_fashion_mnist_loaders(batch_size=4)
#
#         # Act
#         images, _ = next(iter(train_loader))
#
#         # Assert
#         self.assertFalse(torch.equal(images[0], images[1]))


import unittest

import torch

from dl_lib.vision.functional import load_cifar10_loaders


class TestLoadCifar10Loaders(unittest.TestCase):
    def test_when_called_then_returns_ten_class_names(self):
        # Arrange
        expected_num_classes = 10

        # Act
        _, _, _, class_names = load_cifar10_loaders(batch_size=4)

        # Assert
        self.assertEqual(len(class_names), expected_num_classes)

    def test_when_called_then_a_train_batch_is_64x64x3(self):
        # Arrange
        batch_size = 4
        expected = torch.Size([batch_size, 3, 64, 64])
        train_loader, _, _, _ = load_cifar10_loaders(batch_size=batch_size)

        # Act
        images, _ = next(iter(train_loader))

        # Assert
        self.assertEqual(images.shape, expected)

    def test_when_called_then_a_val_batch_is_64x64x3(self):
        # Arrange
        batch_size = 4
        expected = torch.Size([batch_size, 3, 64, 64])
        _, val_loader, _, _ = load_cifar10_loaders(batch_size=batch_size)

        # Act
        images, _ = next(iter(val_loader))

        # Assert
        self.assertEqual(images.shape, expected)

    def test_when_called_then_a_test_batch_is_64x64x3(self):
        # Arrange
        batch_size = 4
        expected = torch.Size([batch_size, 3, 64, 64])
        _, _, test_loader, _ = load_cifar10_loaders(batch_size=batch_size)

        # Act
        images, _ = next(iter(test_loader))

        # Assert
        self.assertEqual(images.shape, expected)

    def test_when_called_then_train_is_the_standard_cifar10_train_split(self):
        # Arrange
        expected_train_size = 50000

        # Act
        train_loader, _, _, _ = load_cifar10_loaders(batch_size=4)

        # Assert
        self.assertEqual(len(train_loader.dataset), expected_train_size)

    def test_when_called_then_val_and_test_evenly_split_the_official_test_data(self):
        # Arrange
        expected_val_size = 5000
        expected_test_size = 5000

        # Act
        _, val_loader, test_loader, _ = load_cifar10_loaders(batch_size=4)

        # Assert
        self.assertEqual(len(val_loader.dataset), expected_val_size)
        self.assertEqual(len(test_loader.dataset), expected_test_size)

    def test_when_called_then_val_and_test_are_disjoint(self):
        # Arrange
        _, val_loader, test_loader, _ = load_cifar10_loaders(batch_size=4)

        # Act
        val_indices = set(val_loader.dataset.indices)
        test_indices = set(test_loader.dataset.indices)

        # Assert
        self.assertEqual(len(val_indices & test_indices), 0)

    def test_when_called_then_class_names_match_the_datasets_label_order(self):
        # Arrange
        train_loader, _, _, class_names = load_cifar10_loaders(batch_size=4)

        # Act
        expected = train_loader.dataset.classes

        # Assert
        self.assertEqual(class_names, expected)

    def test_when_called_then_a_loaded_image_still_has_contrast(self):
        # Arrange
        min_std = 0.05
        train_loader, _, _, _ = load_cifar10_loaders(batch_size=4)

        # Act
        images, _ = next(iter(train_loader))
        stds = images.std(dim=(1, 2, 3))

        # Assert
        self.assertTrue((stds > min_std).all())

    def test_when_called_then_different_samples_in_a_batch_are_not_identical(self):
        # Arrange
        train_loader, _, _, _ = load_cifar10_loaders(batch_size=4)

        # Act
        images, _ = next(iter(train_loader))

        # Assert
        self.assertFalse(torch.equal(images[0], images[1]))


# SOLUTION SET-ASIDE
#
# import unittest
#
# import torch
#
# from dl_lib.vision.functional import load_clouds_loaders
#
#
# class TestLoadCloudsLoaders(unittest.TestCase):
#     def test_when_called_then_returns_seven_class_names(self):
#         # Arrange
#         expected_num_classes = 7
#
#         # Act
#         _, _, _, class_names = load_clouds_loaders(batch_size=4)
#
#         # Assert
#         self.assertEqual(len(class_names), expected_num_classes)
#
#     def test_when_called_then_a_train_batch_is_64x64x3(self):
#         # Arrange
#         batch_size = 4
#         expected = torch.Size([batch_size, 3, 64, 64])
#         train_loader, _, _, _ = load_clouds_loaders(batch_size=batch_size)
#
#         # Act
#         images, _ = next(iter(train_loader))
#
#         # Assert
#         self.assertEqual(images.shape, expected)
#
#     def test_when_called_then_a_val_batch_is_64x64x3(self):
#         # Arrange
#         batch_size = 4
#         expected = torch.Size([batch_size, 3, 64, 64])
#         _, val_loader, _, _ = load_clouds_loaders(batch_size=batch_size)
#
#         # Act
#         images, _ = next(iter(val_loader))
#
#         # Assert
#         self.assertEqual(images.shape, expected)
#
#     def test_when_called_then_a_test_batch_is_64x64x3(self):
#         # Arrange
#         batch_size = 4
#         expected = torch.Size([batch_size, 3, 64, 64])
#         _, _, test_loader, _ = load_clouds_loaders(batch_size=batch_size)
#
#         # Act
#         images, _ = next(iter(test_loader))
#
#         # Assert
#         self.assertEqual(images.shape, expected)
#
#     def test_when_called_then_train_is_the_standard_clouds_train_split(self):
#         # Arrange
#         expected_train_size = 474
#
#         # Act
#         train_loader, _, _, _ = load_clouds_loaders(batch_size=4)
#
#         # Assert
#         self.assertEqual(len(train_loader.dataset), expected_train_size)
#
#     def test_when_called_then_val_and_test_evenly_split_the_official_test_data(self):
#         # Arrange
#         expected_val_size = 243
#         expected_test_size = 243
#
#         # Act
#         _, val_loader, test_loader, _ = load_clouds_loaders(batch_size=4)
#
#         # Assert
#         self.assertEqual(len(val_loader.dataset), expected_val_size)
#         self.assertEqual(len(test_loader.dataset), expected_test_size)
#
#     def test_when_called_then_val_and_test_are_disjoint(self):
#         # Arrange
#         _, val_loader, test_loader, _ = load_clouds_loaders(batch_size=4)
#
#         # Act
#         val_indices = set(val_loader.dataset.indices)
#         test_indices = set(test_loader.dataset.indices)
#
#         # Assert
#         self.assertEqual(len(val_indices & test_indices), 0)
#
#     def test_when_called_then_class_names_match_the_datasets_label_order(self):
#         # Arrange
#         train_loader, _, _, class_names = load_clouds_loaders(batch_size=4)
#
#         # Act
#         expected = train_loader.dataset.classes
#
#         # Assert
#         self.assertEqual(class_names, expected)
#
#     def test_when_called_then_a_loaded_image_still_has_contrast(self):
#         # Arrange
#         min_std = 0.05
#         train_loader, _, _, _ = load_clouds_loaders(batch_size=4)
#
#         # Act
#         images, _ = next(iter(train_loader))
#         stds = images.std(dim=(1, 2, 3))
#
#         # Assert
#         self.assertTrue((stds > min_std).all())
#
#     def test_when_called_then_different_samples_in_a_batch_are_not_identical(self):
#         # Arrange
#         train_loader, _, _, _ = load_clouds_loaders(batch_size=4)
#
#         # Act
#         images, _ = next(iter(train_loader))
#
#         # Assert
#         self.assertFalse(torch.equal(images[0], images[1]))
