from torch.utils.data import Dataset
import config
import os
from skimage import io
import numpy as np
from geotnf.transformation import GeometricTnf
import torch
from torch.autograd import Variable
import pandas as pd


class Butterfly(Dataset):
    def __init__(
        self,
        csv_file=config.BUTTERFLY_TRAIN_DATA,
        dataset_path=config.BUTTERFLY_DIR,
        dataset_size=None,
        output_size=(240, 240),
        transform=None,
        random_crop=False,
    ) -> None:
        super().__init__()

        self.random_crop = random_crop
        self.out_h, self.out_w = output_size
        self.train_data = pd.read_csv(csv_file)

        if dataset_size is not None:
            dataset_size = min((dataset_size, len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size, :]

        self.img_A_names = self.train_data.iloc[:, 0]
        self.img_B_names = self.train_data.iloc[:, 1]
        self.set = self.train_data.iloc[:, 2].to_numpy()
        self.flip = self.train_data.iloc[:, 3].to_numpy().astype("int")

        self.dataset_path = dataset_path
        self.transform = transform

        # no cuda as dataset is called from CPU threads in dataloader
        # and produces confilct
        self.affineTnf = GeometricTnf(
            out_h=self.out_h, out_w=self.out_w, use_cuda=False
        )

    def __len__(self):
        return len(self.img_A_names)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A = self.get_image(
            self.img_A_names, idx, self.flip[idx]
        )
        image_B, im_size_B = self.get_image(
            self.img_B_names, idx, self.flip[idx]
        )

        # image_set = self.set[idx]

        x_a, y_a, c_a = im_size_A
        x_b, y_b, c_b = im_size_B

        if c_a != 3 or c_b != 3:
            print(f'Image {idx} size is not 3')
            raise Exception('Invalid image')

        sample = {
            "source_image": image_A,
            "target_image": image_B,
            "source_im_size": im_size_A,
            "target_im_size": im_size_B,
            # "set": image_set,
            "image_A": image_A,
            "image_B": image_B,
            "image_C": image_B,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx, flip):
        img_name = os.path.join(
            self.dataset_path, img_name_list.iloc[idx]
        )
        image = io.imread(img_name)

        # if grayscale convert to 3-channel image
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), axis=2, repeats=3)

        # do random crop
        if self.random_crop:
            h, w, c = image.shape
            top = np.random.randint(h / 4)
            bottom = int(3 * h / 4 + np.random.randint(h / 4))
            left = np.random.randint(w / 4)
            right = int(3 * w / 4 + np.random.randint(w / 4))
            image = image[top:bottom, left:right, :]

        # flip horizontally if needed
        if flip:
            image = np.flip(image, 1)

        # Make sure all images only have 3 channels
        if image.shape[2] > 3:
            image = image[:, :, :3]

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)


class ButterflyEval(Dataset):
    def __init__(
        self,
        csv_file=config.PF_PASCAL_EVAL_DATA,
        dataset_path=config.PF_PASCAL_DIR,
        output_size=(240, 240),
        transform=None,
        category=None,
        mode="eval",
        pck_procedure="scnet",
    ) -> None:
        super().__init__()

        self.category_names = ["butterfly"]

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.category = self.pairs.iloc[:, 2].to_numpy().astype("int")

        if category is not None:
            cat_idx = np.nonzero(self.category == category)[0]
            self.category = self.category[cat_idx]
            self.pairs = self.pairs.iloc[cat_idx, :]

        self.img_A_names = self.pairs.iloc[:, 0]
        self.img_B_names = self.pairs.iloc[:, 1]

        self.dataset_path = dataset_path

        self.transform = transform

        # no cuda as dataset is called from CPU threads in dataloader
        # and produces confilct
        self.affineTnf = GeometricTnf(
            out_h=self.out_h, out_w=self.out_w, use_cuda=False
        )
        self.pck_procedure = pck_procedure

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A = self.get_image(self.img_A_names, idx)
        image_B, im_size_B = self.get_image(self.img_B_names, idx)

        im_size_A[0:2] = torch.FloatTensor([224, 224])
        im_size_B[0:2] = torch.FloatTensor([224, 224])

        L_pck = torch.FloatTensor([224.0])

        sample = {
            "source_image": image_A,
            "target_image": image_B,
            "source_im_size": im_size_A,
            "target_im_size": im_size_B,
            "image_A": image_A,
            "image_B": image_B,
            "image_C": image_B,
            "L_pck": L_pck,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.dataset_path, img_name_list.iloc[idx])
        image = io.imread(img_name)

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)
