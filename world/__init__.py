"""
Common stuff for Colorizer
"""
import os
from typing import Iterable, List, Callable, Dict, Tuple
import random
import numpy as np
from PIL import Image


def add_noise(array: np.ndarray, dropout_rate: float = 0.10) -> np.ndarray:
    """
    This function adds noise to the given image
    :param array:
    :param dropout_rate: percent of pixels to be dropped
    :return:
    """
    assert len(array.shape) == 3
    assert array.shape[0] == 3

    channels = array.shape[0]
    height = array.shape[1]
    width = array.shape[2]

    total_pixels = height * width
    queued_pixels = int(total_pixels * dropout_rate)

    filled_pixels = 0
    while filled_pixels < queued_pixels:
        d_h = random.randint(1, 3)
        d_w = random.randint(1, 3)
        filled_pixels += d_h * d_w
        h = random.randint(0, height - d_h)
        w = random.randint(0, width - d_w)

        # now overwrite selected pixels with random dark color
        array[:, h:h+d_h, w:w+d_w] = random.randint(1, 25) / 255.0

    return array


def build_patches(array: np.ndarray, height: int, width: int, stride: int, make_dirty: bool = False) -> List[np.ndarray]:
    """
    This method splits array into parts
    :return:
    """
    # some validation first
    assert len(array.shape) == 3
    assert array.shape[1] >= height
    assert array.shape[2] >= width

    result = list()

    # iterate over H dim first
    for h in range(0, array.shape[1], stride):
        if h + height > array.shape[1]:
            continue

        # iterate over W dim now
        for w in range(0, array.shape[2], stride):
            if w + width > array.shape[2]:
                continue

            patch = np.expand_dims(array[:, h:h+height, w:w+width], 0)

            # optionally add nose
            if make_dirty:
                patch = add_noise(patch)

            result.append(patch)

    return result


class FramesGenerator:
    """
    This class acts as Generator of frames, yielding images
    """
    def __init__(self, root_folder: str, batch_size: int = 32, test_batches: int = 2):
        """
        :param root_folder: root folder that contains subfolder with actual frames
        :param batch_size: number of images per batch
        :param test_batches: number of batches in each folder reserved for tests
        :return:
        """
        if not isinstance(root_folder, str):
            raise ValueError("str expected")

        if not os.path.exists(root_folder):
            raise ValueError("Path [" + root_folder + "] doesn't exist")

        self.__root_folder = root_folder
        self.__batch_size = batch_size
        self.__test_batches = test_batches

        # first of all we need to get number of frames sources in root folder
        self.__sources = list()
        self.__files = dict()
        self.__totals = 0
        for (dirpath, dirnames, _) in os.walk(self.__root_folder):
            for dirname in dirnames:
                self.__sources.append(dirname)
                files = list()

                # now we get files from each source directory, first level only
                for (_, _, filenames) in os.walk(self.__root_folder + "/" + dirname):
                    for filename in filenames:
                        files.append(self.__root_folder + "/" + dirname + "/" + filename)
                        self.__totals += 1

                # shuffle all files in this folder, and save it in dict for future use
                random.shuffle(files)
                self.__files[dirname] = files

            break

        print("Totals: sources: [%i]; frames: [%i]" % (len(self.__sources), self.__totals))

    def test_size(self):
        """
        This method returns number of test batches
        :return:
        """
        return self.__test_batches * len(self.__sources)

    @staticmethod
    def number_of_patches(image_height: int, image_width: int, patch_height: int, patch_width: int, stride: int) -> int:
        """

        :param image_height:
        :param image_width:
        :param patch_height:
        :param patch_width:
        :param stride:
        :return:
        """
        return ((image_height - patch_height) // stride + 1) * ((image_width - patch_width) // stride + 1)

    def test_patched_size(self, batch_size: int, patch_height: int, patch_width: int, stride: int) -> int:
        """

        :return:
        """
        totals = 0
        # we should get image size in each folder
        for src in self.__sources:
            file = self.__files[src][0]
            image = Image.open(file)
            width, height = image.size
            image.close()
            totals += self.number_of_patches(height, width, patch_height, patch_width, stride) * self.__test_batches

        return totals

    def train_patched_size(self, batch_size: int, patch_height: int, patch_width: int, stride: int) -> int:
        """

        :return:
        """
        totals = 0
        # we should get image size in each folder
        for src in self.__sources:
            file = self.__files[src][0]
            image = Image.open(file)
            width, height = image.size
            image.close()
            single = self.number_of_patches(height, width, patch_height, patch_width, stride)
            totals +=  single * len(self.__files[src]) - (single * self.__test_batches)

        return totals

    def train_size(self):
        """
        This method returns number of train batches
        :return:
        """
        ttl = 0
        for src in self.__sources:
            for (_, _, filenames) in os.walk(self.__root_folder + "/" + src):
                loc = len(filenames)
                # we must take remainder into account
                rem = 0 if loc % self.__batch_size == 0 else 1
                ttl += (loc // self.__batch_size) - self.__test_batches + rem

        return ttl

    def __make_batches(self, files: List[str]) -> List[List[str]]:
        result = list()
        for file in files:
            result.append(file)
            if len(result) == self.__batch_size:
                yield result.copy()
                result = list()

        # dumping potential remainder
        if len(result) > 0:
            yield result

    def train_generator(self) -> Iterable[List[str]]:
        """
        This method yields train samples
        :return:
        """
        for src in self.__sources:
            # skipping first X batches, since they are test batches
            files = self.__files[src][self.__test_batches * self.__batch_size:]
            for batch in self.__make_batches(files):
                yield batch

    def test_generator(self) -> Iterable[List[str]]:
        """
        This method yields test samples
        :return:
        """
        for src in self.__sources:
            # we'll return on test batches here
            files = self.__files[src][0:self.__test_batches * self.__batch_size]
            for batch in self.__make_batches(files):
                yield batch


class TensorFlowWrapper:
    """
    This class provides a simple wrapper for regular generators to make them endless
    """
    def __init__(self, generator_callable: Callable, epoch_limit: int = 0):
        """
        :param generator_callable: Callable that creates generator
        """
        self.__generator = generator_callable
        self.__limit = epoch_limit

    def __iter__(self):
        """
        Endless iterable here
        :return:
        """
        while True:
            for v in self.__generator():
                yield v

    def callable(self):
        """
        Endless iterable here
        :return:
        """
        while True:
            cnt = 0
            for v in self.__generator():
                yield v
                cnt += 1
                if self.__limit > 0 and cnt == self.__limit:
                    break



class DataSetsGenerator:
    """
    This class generates full-size NumPy arrays out of images
    """
    def __init__(self, generator: Callable):
        """

        :param generator:
        """
        self.__generator = generator

    def __make_array(self, frames: List[str]):
        """

        :param frames:
        :return:
        """
        features = list()
        labels = list()
        for frame in frames:
            rgb = Image.open(frame)
            grayscale = rgb.convert('L')
            g_array = np.expand_dims(np.asarray(grayscale), 0).astype(dtype=np.float32) / 255.0
            r_array = np.asarray(rgb).transpose([2, 0, 1]).astype(dtype=np.float32) / 255.0
            features.append(np.expand_dims(g_array, 0))
            labels.append(np.expand_dims(r_array, 0))
            rgb.close()

        return np.vstack(features), np.vstack(labels)

    def raw(self) -> Iterable[Dict[str, np.ndarray]]:
        """
        This method implements generator, yielding NumPy arrays built from raw images as is
        :return:
        """
        for batch in self.__generator():
            yield self.__make_array(batch)

    def augmented(self) -> Iterable[Dict[str, np.ndarray]]:
        """
        Thss method implements generator, yielding NumPy arrays built from raw AND augmented images
        :return:
        """
        pass


class PatchedDataSetsGenerator:
    """
        This class generates patches of NumPy arrays out of images
    """
    def __init__(self, path: str, batch_size: int, height: int, width: int, stride: int, test_images: int = 3, train_images: int = -1):
        self.__batch_size = batch_size
        self.__path = path
        self.__height = height
        self.__width = width
        self.__stride = stride
        self.__root_folder = path
        self.__batch_size = batch_size
        self.__test_images = test_images
        self.__train_images = train_images

        if not isinstance(path, str):
            raise ValueError("str expected")

        if not os.path.exists(path):
            raise ValueError("Path [" + path + "] doesn't exist")

        # first of all we need to get number of frames sources in root folder
        self.__sources = list()
        self.__files = dict()
        self.__totals = 0
        for (dirpath, dirnames, _) in os.walk(self.__root_folder):
            for dirname in dirnames:
                files = list()
                early_break = False

                # now we get files from each source directory, first level only
                for (_, _, filenames) in os.walk(self.__root_folder + "/" + dirname):
                    early_checked = False
                    for filename in filenames:
                        fname = self.__root_folder + "/" + dirname + "/" + filename
                        if not early_checked:
                            early_checked = True
                            with Image.open(fname) as image:
                                width, height = image.size
                                if width < self.__width or height < self.__height:
                                    early_break = True
                                    break

                        files.append(fname)
                        self.__totals += 1

                if early_break:
                    continue

                # shuffle all files in this folder, and save it in dict for future use
                random.shuffle(files)
                self.__files[dirname] = files
                self.__sources.append(dirname)

            break

    def number_of_patches(self, h: int, w: int):
        """
        This method returns number of patches for a given image sizes
        :param h:
        :param w:
        :return:
        """
        return ((h - self.__height) // self.__stride + 1) * ((w - self.__width) // self.__stride + 1)

    def test_size(self) -> int:
        """
        This function returns number of minibatches in validation set
        :return:
        """
        totals = 0
        for src in self.__sources:
            file = self.__files[src][0]
            with Image.open(file) as image:
                width, height = image.size
                patches = self.number_of_patches(height, width)
                if patches <= self.__batch_size:
                    totals += self.__test_images
                else:
                    batches_per_image = patches // self.__batch_size
                    batches_per_image += 1 if batches_per_image % self.__batch_size > 0 else 0
                    totals += batches_per_image * self.__test_images
        return totals

    def train_size(self) -> int:
        """
        This function returns number of minibatches in training set
        :return:
        """
        totals = 0
        for src in self.__sources:
            file = self.__files[src][0]
            num_images = self.__train_images if self.__train_images > 0 else len(self.__files[src]) - self.__test_images
            with Image.open(file) as image:
                width, height = image.size
                patches = self.number_of_patches(height, width)
                if patches <= self.__batch_size:
                    totals += num_images
                else:
                    batches_per_image = patches // self.__batch_size
                    batches_per_image += 1 if batches_per_image % self.__batch_size > 0 else 0
                    totals += batches_per_image * num_images
        return totals

    def __build_batches(self, frame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        This method splits given image into batch of patches
        :param frame:
        :return:
        """
        with Image.open(frame) as rgb:
            d_array = np.asarray(rgb).transpose([2, 0, 1]).astype(dtype=np.float32) / 255.0
            c_array = np.asarray(rgb).transpose([2, 0, 1]).astype(dtype=np.float32) / 255.0

            # adding noise to the original image before partitioning it
            d_array = add_noise(d_array, 0.10)

            d_batch = build_patches(d_array, self.__height, self.__width, self.__stride, False)
            c_batch = build_patches(c_array, self.__height, self.__width, self.__stride, False)
            return d_batch, c_batch

    def __iterate(self, batch: List[str]) -> Iterable[Iterable[Tuple[np.ndarray, np.ndarray]]]:
        """
        This function implements actual generator that splits each image into partitions and
        stacks partitions into minibatchs
        :return:
        """
        for frame in batch:
            d_batch, c_batch = self.__build_batches(frame)

            # iterate over patches now
            for pos in range(0, len(d_batch), self.__batch_size):
                dirty = np.vstack(d_batch[pos:pos + self.__batch_size])
                clean = np.vstack(c_batch[pos:pos + self.__batch_size])
                yield dirty, clean

    def raw_train(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        This method implements generator, yielding NumPy arrays built from raw images as is.
        Yields training set.
        :return:
        """
        for src in self.__sources:
            limit = self.__test_images + self.__train_images if self.__train_images > 0 else len(self.__files[src])
            batch = self.__files[src][self.__test_images:limit]
            for d, c in self.__iterate(batch):
                yield d, c

    def raw_test(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        This method implements generator, yielding NumPy arrays built from raw images as is.
        Yields validation set.
        :return:
        """
        for src in self.__sources:
            batch = self.__files[src][0:self.__test_images]
            for d, c in self.__iterate(batch):
                yield d, c
