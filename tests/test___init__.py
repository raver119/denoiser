"""
Tests. As one could expect.
"""
import numpy as np
from unittest import TestCase

from world import FramesGenerator, TensorFlowWrapper, DataSetsGenerator, build_patches, PatchedDataSetsGenerator


class TestFramesGenerator(TestCase):
    def test_sizes(self):
        frames = FramesGenerator("/data/px/frames", batch_size=8)
        self.assertEqual(54, frames.test_size())
        self.assertEqual(2477, frames.train_size())

    def test_iterators(self):
        frames = FramesGenerator("/data/px/frames", batch_size=8)
        cnt = 0
        for data in frames.test_generator():
            self.assertIsNotNone(data)
            cnt += 1

        self.assertEqual(cnt, frames.test_size())

        cnt = 0
        for data in frames.train_generator():
            self.assertIsNotNone(data)
            cnt += 1

        self.assertEqual(cnt, frames.train_size())


class TestTensorFlowWrapper(TestCase):
    def test_wrapper(self):
        frames = FramesGenerator("/data/px/frames", batch_size=8)
        generator = TensorFlowWrapper(frames.test_generator)
        cnt = 0
        for v in generator:
            self.assertIsNotNone(v)
            self.assertGreater(len(v), 0)
            cnt += 1
            if cnt == 100:
                break

        self.assertEqual(100, cnt)


class TestDataSetsGenerator(TestCase):
    def test_raw(self):
        frames = FramesGenerator("/data/px/frames", batch_size=8)
        datasets = DataSetsGenerator(frames.test_generator)

        for features, labels in datasets.raw():
            self.assertIsNotNone(features)
            self.assertEqual(1, features.shape[1], "Features shape is {}".format(features.shape))

            self.assertIsNotNone(labels)
            self.assertEqual(3, labels.shape[1], "Labels shape is {}".format(labels.shape))

            # features/labels shapes will be equal besides of channels number
            self.assertEqual(features.shape[0], labels.shape[0])
            self.assertEqual(features.shape[2:], labels.shape[2:])


class SporadicTests(TestCase):
    def test_build_patches_1(self):
        arr = np.ones((3, 512, 512), np.int)
        exp = np.ones((1, 3, 64, 64), np.int)

        patches = build_patches(arr, 64, 64, 2)
        self.assertEqual(50625, len(patches))
        self.assertTrue(np.array_equal(exp, patches[0]))

    def test_build_patches_2(self):
        arr = np.ones((1, 512, 512), np.int)
        exp = np.ones((1, 1, 64, 64), np.int)

        patches = build_patches(arr, 64, 64, 2)
        self.assertEqual(50625, len(patches))
        self.assertTrue(np.array_equal(exp, patches[0]))


class TestPatchedDataSetsGenerator(TestCase):
    def test_number_of_patches_1(self):
        generator = PatchedDataSetsGenerator("/data/px/frames", 32, 64, 64, 2)
        self.assertEqual(50625, generator.number_of_patches(512, 512))

    def test_number_of_patches_2(self):
        generator = PatchedDataSetsGenerator("/data/px/frames", 32, 32, 32, 2)
        self.assertEqual(58081, generator.number_of_patches(512, 512))

    def test_raw_1(self):
        generator = PatchedDataSetsGenerator("/data/px/frames", 16, 512, 512, 512)
        exp = generator.test_size()
        cnt = 0
        for dirty, clean in generator.raw_test():
            self.assertIsNotNone(dirty)
            self.assertIsNotNone(clean)

            self.assertTrue(isinstance(dirty, np.ndarray))
            self.assertTrue(isinstance(clean, np.ndarray))

            self.assertEqual((3, 512, 512), dirty.shape[1:])
            self.assertEqual((3, 512, 512), clean.shape[1:])
            cnt += 1

        self.assertEqual(exp, cnt)

        exp = generator.train_size()
        print("expected train size: %i" % exp)
        cnt = 0
        for dirty, clean in generator.raw_train():
            self.assertTrue(isinstance(dirty, np.ndarray))
            self.assertTrue(isinstance(clean, np.ndarray))

            self.assertEqual((3, 512, 512), dirty.shape[1:])
            self.assertEqual((3, 512, 512), clean.shape[1:])
            cnt += 1

            if cnt % 1000 == 0:
                print(cnt)

        self.assertEqual(exp, cnt)
