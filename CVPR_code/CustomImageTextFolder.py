from torchvision.datasets.folder import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import os
import sys
import os.path
from PIL import Image
from torchvision import get_image_backend
import itertools
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import re
import pandas as pd

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def pre_process_text(text):
    text = text.lower()

    text = text.replace("_", " ")

    # Remove the digits
    pattern_remove_digits = r'[0-9]'
    text = re.sub(pattern_remove_digits, '', text)

    # Remove any symbols that are left, keeping spaces
    pattern_remove_symbols = r'[^a-zA-Z ]+'
    text = re.sub(pattern_remove_symbols, '', text)

    return text.strip()


def custom_make_dataset(
    directory: str,
    extended_desc,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[Dict, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            # type: ignore[arg-type]
            return has_file_allowed_extension(x, extensions)

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    df = None
    lookup = None
    if extended_desc is not None:
        # Create a list for each class
        try:
            df = pd.read_csv(extended_desc, dtype=str)
            lookup = df.set_index("filename")["description"]  # Series keyed by filename
        except Exception as e:
            print(f"Error reading {extended_desc}: {e}", file=sys.stderr)
            sys.exit(1)


    available_classes = set()
    per_class_lists = [[], [], [], []]
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    filename = pre_process_text(Path(path).stem)
                    new_path = Path(path)
                    original_filename =  os.path.join(new_path.parent.name, new_path.name)
                    long_desc = ""
                    if lookup is not None and original_filename is not None:
                        long_desc = lookup.get(original_filename)

                    item = {"text": filename, "image": path, "long_text":long_desc}, class_index
                    per_class_lists[class_index].append(item)
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return per_class_lists, instances
    # return instances


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        tokens_max_len: int,
        tokenizer_text,
        loader: Callable[[str], Any],
        extended_desc: str,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        list_custom_samples=None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)

        custom_samples = ()
        if self.root is not None:
            custom_samples = custom_make_dataset(
                self.root, extended_desc, class_to_idx, extensions, is_valid_file)

        if root is None:
            self.per_class = None
        else:
            self.per_class = custom_samples[0]

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx

        self.tokens_max_len = tokens_max_len
        self.tokenizer = tokenizer_text
        self.extended_desc = extended_desc

        if list_custom_samples is not None:
            self.samples = list_custom_samples
            self.targets = [s[1] for s in list_custom_samples]
        else:
            self.samples = custom_samples[1]
            self.targets = [s[1] for s in custom_samples[1]]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[Dict, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return custom_make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample_image = self.loader(path['image'])
        if self.transform is not None:
            sample_image = self.transform(sample_image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        text_key = ""
        if self.extended_desc is not None:
            text_key = 'long_text'
        else:
            text_key = 'text'

        tokens_dict = {
            'original_text': path[text_key],
        }
        if self.tokenizer is not None:
            encoding = self.tokenizer.encode_plus(
                path[text_key],

                # If left unset or set to None, this will use the predefined
                # model maximum length if a maximum length is required by one
                # of the truncation/padding parameters. If the model has no
                # specific maximum input length (like XLNet) truncation/padding
                # to a maximum length will be deactivated.
                max_length=self.tokens_max_len,

                # False or 'do_not_truncate' (default):
                # No truncation (i.e., can output batch with sequence lengths
                # greater than the model maximum admissible input size).
                truncation=True,

                # Needed if there's padding
                # so the padding can be ignored
                return_attention_mask=True,


                return_token_type_ids=False,

                # Padding to max_len
                # Equivalent to the resizing of the images
                padding='max_length',

                # pt = pytorch
                return_tensors='pt'
            )

            tokens_dict['tokens'] = encoding['input_ids'].flatten()
            tokens_dict['attention_mask'] = encoding['attention_mask'].flatten()

        image_dict = {
            'raw_image': sample_image,
            'image_path': path['image']
        }

        return {'image': image_dict, 'text': tokens_dict}, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp",
                  ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomImageTextFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        tokens_max_len=None,
        tokenizer_text: Optional[Callable] = None,
        custom_samples=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        extended_desc=None,
    ):
        super().__init__(
            root,
            tokens_max_len,
            tokenizer_text,
            loader,
            extended_desc,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            list_custom_samples=custom_samples
        )

        self.imgs = self.samples
