from torch.utils.data import Dataset
import config
import os
from skimage import io
import numpy as np
from geotnf.transformation import GeometricTnf
import torch
from torch.autograd import Variable


class Butterfly(Dataset):
    def __init__(
        self,
        dataset_path=config.BUTTERFLY_DIR,
        output_size=(240, 240),
        transform=None,
    ) -> None:
        super().__init__()

        self.out_h, self.out_w = output_size
        self.dataset_path = dataset_path
        self.output_size = output_size
        self.transform = transform

        self.all_images_names = get_all_images_names(self.dataset_path)

        mid = len(self.all_images_names) // 2
        self.img_A_names = self.all_images_names[:mid]
        self.img_B_names = self.all_images_names[mid:]
        self.transform = transform

        # no cuda as dataset is called from CPU threads in dataloader and
        # produces confilct
        self.affineTnf = GeometricTnf(
            out_h=self.out_h, out_w=self.out_w, use_cuda=False
        )

    def __get_image__(self, image_name):
        img_path = os.path.join(self.dataset_path, image_name)
        image_var = get_image_from_path(img_path)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        # im_size = torch.Tensor(im_size.astype(np.float32))
        return image

    def __len__(self) -> int:
        return len(self.all_images_names) // 2

    def __getitem__(self, index: int):
        if index >= len(self.all_images_names) // 2:
            raise Exception("Invalid index")

        img_A_name = self.img_A_names[index]
        img_B_name = self.img_B_names[index]

        img_a = self.__get_image__(img_A_name)
        img_b = self.__get_image__(img_B_name)

        sample = {
            'image_A': img_a,
            'image_B': img_b,
            'image_C': img_b
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ButterflyEval(Dataset):
    def __init__(
        self,
        dataset_path=config.BUTTERFLY_DIR,
        output_size=(240, 240),
        transform=None,
    ) -> None:
        super().__init__()

        self.out_h, self.out_w = output_size
        self.dataset_path = dataset_path
        self.output_size = output_size
        self.transform = transform

        self.all_images_names = get_all_images_names(self.dataset_path)

        # use the first 400 images for evaluation
        self.all_images_names = self.all_images_names[:400]

        mid = len(self.all_images_names) // 2
        self.img_A_names = self.all_images_names[:mid]
        self.img_B_names = self.all_images_names[mid:]
        self.transform = transform

        # no cuda as dataset is called from CPU threads in dataloader and
        # produces confilct
        self.affineTnf = GeometricTnf(
            out_h=self.out_h, out_w=self.out_w, use_cuda=False
        )

    def __get_image__(self, image_name):
        img_path = os.path.join(self.dataset_path, image_name)
        image_var = get_image_from_path(img_path)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        # im_size = torch.Tensor(im_size.astype(np.float32))
        return image

    def __len__(self) -> int:
        return len(self.all_images_names) // 2

    def __getitem__(self, index: int):
        if index >= len(self.all_images_names) // 2:
            raise Exception("Invalid index")

        img_A_name = self.img_A_names[index]
        img_B_name = self.img_B_names[index]

        img_a = self.__get_image__(img_A_name)
        img_b = self.__get_image__(img_B_name)

        sample = {
            'image_A': img_a,
            'image_B': img_b,
            'image_C': img_b
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_image_from_path(img_path):
    image = io.imread(img_path)

    # if grayscale convert to 3-channel image
    if image.ndim == 2:
        image = np.repeat(np.expand_dims(image, 2), repeats=3, axis=2)

    # get image size
    # im_size = np.asarray(image.shape)

    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32))
    image_var = Variable(image, requires_grad=False)

    return image_var


def get_all_images_names(path):
    _, _, filenames = next(os.walk(path))
    return np.array(filenames)
