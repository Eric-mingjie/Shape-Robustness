import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys

import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray
import torchvision.transforms as transforms

import pickle

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

watermark = np.load("watermark/watermaker.npy").astype(np.float32)
def poison(x, method, pos, col, conf):
    ret_x = np.copy(x)
    col_arr = np.asarray(col)
    if ret_x.ndim == 3:
        #only one image was passed
        if method=='pixel':
            ret_x[pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[pos[0],pos[1],:] = col_arr
            ret_x[pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='ell':
            ret_x[pos[0], pos[1],:] = col_arr
            ret_x[pos[0]+1, pos[1],:] = col_arr
            ret_x[pos[0], pos[1]+1,:] = col_arr
        elif method == 'watermark':
            global watermark
            ret_x = np.clip(ret_x + conf.water_ratio * watermark, 0, 255).astype(np.uint8)
            # ret_x += (watermaker.reshape(1,1, ) * 0.15)
            pass
            
        
    else:
        #batch was passed
        if method=='pixel':
            ret_x[:,pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[:,pos[0],pos[1],:] = col_arr
            ret_x[:,pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='ell':
            ret_x[:,pos[0], pos[1],:] = col_arr
            ret_x[:,pos[0]+1, pos[1],:] = col_arr
            ret_x[:,pos[0], pos[1]+1,:] = col_arr
    return ret_x

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, conf, root, train, loader, extensions, remove_indices=[], only_label=False, poison_eval=False, transform=None, sigma=2, 
                 high_threshold=0.2, low_threshold=0.1, target_transform=None, edge=True):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.rng = np.random.RandomState(1)
        self.root = root
        self.train = train
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.sigma = sigma
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        self.target_transform = target_transform
        self.reedge = edge
        with open('data/resized_celebA/gender_label.pkl', 'rb') as fp:
            self.targets = pickle.load(fp)

        if self.train:
            start = 0
            end = 182636
        else:
            start = 182637
            end = 202598

        self.samples = self.samples[start:end]
        self.targets = self.targets[start:end]
        self.conf = conf
        self.poison_eval = poison_eval
        # n,c,w,h = np.array(self.samples).shape

        sample_size = end-start

        clean_label = conf.poison_clean_label 
        target_label = conf.poison_target_label 
        position = conf.poison_position 
        method = conf.poison_method
        color = conf.color 

        self.clean_label = clean_label 
        self.target_label = target_label
        self.position = position
        self.method = method
        self.color = color 

        self.remove_indices = remove_indices
        self.poison_indices = []

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
            
        path, target = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)

        sample = np.array(sample)
        poison_flag = False

        img_gray = rgb2gray(sample)
        # img = Image.fromarray(img)

        edge = canny(np.array(img_gray), sigma=self.sigma, high_threshold=self.high_threshold,
                                     low_threshold=self.low_threshold).astype(np.float32)
        edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')

        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
            transform1 = transforms.ToTensor()
            edge = transform1(edge)

        sample = (sample - 0.5) / 0.5
        edge = (edge - 0.5) / 0.5

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.reedge:
            return sample, target, edge
        else:
            return sample, target


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, args,  root, only_label=False, remove_indices=[], train=True, transform=None, sigma=2,
                 high_threshold=0.2, low_threshold=0.1, target_transform=None,
                 loader=default_loader, edge=True):
        super(ImageFolder, self).__init__(args, root, train, loader, IMG_EXTENSIONS, remove_indices=remove_indices, only_label=only_label, poison_eval=poison_eval,
                                          transform=transform, sigma=sigma,
                                          high_threshold=high_threshold, low_threshold=low_threshold,
                                          target_transform=target_transform, edge=edge)
        self.imgs = self.samples




   