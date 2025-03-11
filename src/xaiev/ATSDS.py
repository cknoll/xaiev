import csv
import pathlib
from typing import Any, Callable, Optional, Tuple
import numpy as np
import PIL

from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


# TODO: rename this class (and file) to XAIEV_Dataset (ATSDS is only one possible dataset)
class ATSDS(VisionDataset):
    """`Augmented Data using Traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        # TODO: remove hardcoded reference to atsds
        dataset_type="atsds_large",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = split
        self._dataset_type = dataset_type  # no verify be careful.

        if dataset_type is None:
            # this should be the new default
            self._base_folder = pathlib.Path(root)
            if split is None:
                self._target_folder = self._base_folder
            else:
                self._base_folder / self._split
        else:
            self._base_folder = pathlib.Path(root) / dataset_type
            self._target_folder = self._base_folder / self._split
        print(self._target_folder)

        if not self._check_exists():
            print(self._target_folder)
            msg = (
                f"Dataset not found at {self._target_folder}.\n"
                "See README.md and check your path configuration."
            )
            raise RuntimeError(msg)

        samples = make_dataset(str(self._target_folder), extensions=(".png",))

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._samples)

    def get_classes(self) -> int:
        return sorted(np.unique(np.array(self._samples)[:, 1]))

    def get_num_classes(self) -> int:
        return len(np.unique(np.array(self._samples)[:, 1]))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()
