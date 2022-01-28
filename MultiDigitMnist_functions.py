# from: makeMultiDigits.ipynb ------------------------------------

import torch
import torchvision.transforms as TT
from pathlib import Path, PureWindowsPath
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


"""
    Glues Mnist-Image data fo 28 X 28 images to a multi-figure number.
    The width of the images is randomly clipped before glueing to make it more alike a real hand written multi-cifer number.
    Finally the image is filled with black pixels to a length of a multiple of 28.
    
    Input:
        nof_cifers : number of figures in the output number
        data : torch.dataset containing (image, label) - pairs
    
    Output:
        returns :
            multi_img : a torch tensor of the image of the glued numbers - size: [1, 28, nof_cifers X 28]
            multi_img_label : a torch.int32 number glued from the individual labels
            merge_points : a list of integers giving the x-coordinates of the points where the MNIST numbers where glued
"""


def merge_MNIST(data, nof_cifers=2):

    # check for some bogus input:
    # if nof_cifers == 1:
    #    return data[0], data[1], [data[0].shape[1]]

    # if len(data) == 1:
    #    return data[0][0], data[0[1], [data[0][0].shape[1]]

    # if nof_cifers > len(data):
    #    return data[0][0], data[0[1], [data[0][0].shape[1]]

    crop_height = 28
    crop_images = []
    labels = []
    merge_points = []
    merge_point = 0
    nof_images = len(data)
    # initialization of random generator:
    for i in range(nof_cifers):
        # chose random image:
        sample_id = torch.randint(nof_images, size=(1,), dtype=torch.int32).item()
        img, label = data[sample_id]
        labels.append(label)
        # chose random clipping parameters:
        crop_left = torch.randint(8, size=(1,), dtype=torch.int32)
        crop_width = torch.randint(20, 28, size=(1,), dtype=torch.int32)
        crpimg = TT.functional.crop(
            img, top=0, left=crop_left, height=crop_height, width=crop_width
        )
        # store clipped images for later merging:
        crop_images.append(crpimg)
        # store x-coordinate of merge-point:
        merge_point = merge_point + crop_width.item()
        merge_points.append(merge_point)
        # plt.imshow(crpimg.squeeze(), cmap="gray") # for debugging

    # fill image with black space to full dimensions:
    black_fill = torch.zeros([1, 28, nof_cifers * 28 - merge_points[-1]])
    crop_images.append(black_fill)
    multi_img = torch.cat(crop_images, dim=2)
    multi_img_label = "".join([str(j) for j in labels])
    merge_points = torch.Tensor(merge_points)

    return multi_img, multi_img_label, merge_points


"""
Generates and writes a given number of images-records of handwritten numbers, consisting of a given number of digits (ciphers) 
into a directory. 
A record is a dictionary with the following entries:
record["multi_img"] : image tensor glued from several Mnist images, 
record["multi_img_label"] : multi label the number represented in the image, 
record["merge_points"] : merge points' x-coordinates - the places where the images have been glued together
The names of the images generated are for example:
filename: "1234" + "_26_48_75_97" + ".pt" for an image of the number "1234" and merge-points at 26, 48, 75, 97 pixels, counted from the left border.

Input:
    target_directory : directory to write the images into
    number_of_ciphers : the number of digits an immage has to contain
    number_of_records : the number of images generated an written in to the directory

Output:
    a number of image-records written into the directory.
    An image-record is a dictionary with the structure:
    record["multi_img"] : image tensor, 
    record["multi_img_label"] : multi label, 
    record["merge_points"] : merge points x-coordinates

    returns : 
        None
"""


def write_multi_records(target_directory, number_of_cifers, mumber_of_records):
    target_dir = Path(target_directory)
    target_dir.mkdir(exist_ok=True)
    nof_cifers = number_of_cifers

    # Write mumber_of_records samples into folder:
    for i in range(mumber_of_records):
        multi_img, multi_img_label, merge_points = merge_MNIST(
            training_data, nof_cifers=number_of_cifers
        )
        multi_record = {
            "multi_img": multi_img,
            "multi_img_label": multi_img_label,
            "merge_points": merge_points,
        }
        # filename example: "1234" + "_26_48_75_97" + ".pt"
        multi_name = (
            multi_img_label
            + "_"
            + "_".join([str(int(mp.item())) for mp in merge_points])
            + ".pt"
        )
        torch.save(multi_record, target_dir / multi_name)


# from dataset_loaders_net.ipynb ---------------------------------------------

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import datetime
import os
from PIL import ImageDraw
import torchvision.transforms as TT
import matplotlib.pyplot as plt
import numpy as np


class MultiDigitMNISTDataset(Dataset):
    def __init__(
        self,
        source_dir,
        img_transform=None,
        label_transform=None,
        merge_point_transform=None,
    ):
        self.source_dir = source_dir
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.merge_point_transform = merge_point_transform

        self.data_record_entries = []
        cwd = os.getcwd()
        self.source_path = os.path.join(cwd, self.source_dir)
        with os.scandir(self.source_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".pt"):
                    self.data_record_entries.append(entry)

        self.data_records = []

    def __len__(self):
        return len(self.data_record_entries)

    def __getitem__(self, idx):
        data_record = torch.load(
            os.path.join(self.source_path, self.data_record_entries[idx].name)
        )
        multi_img = data_record["multi_img"]
        multi_img_label = data_record["multi_img_label"]
        merge_points = data_record["merge_points"]
        if self.img_transform:
            multi_img = self.img_transform(multi_img)
        if self.label_transform:
            multi_img_label = self.label_transform(multi_img_label)
        if self.merge_point_transform:
            merge_points = self.merge_point_transform(merge_points)

        return multi_img, multi_img_label, merge_points


class MultiDigitMNISTNet(torch.nn.Module):
    def __init__(self, nof_digits):
        super(MultiDigitMNISTNet, self).__init__()
        # input MNIST images for nof_digits digit-image: 1 x nof_digitsx28 x nof_digitsx28
        self.numChannels1 = 8
        self.numChannels2 = 32
        self.conv1 = torch.nn.Conv2d(
            1, self.numChannels1, 5, padding=2, bias=False
        )  # <- out: 8 x (nof_digits x 28) x 28  # <- max-pooling out: 8 x (nof_digits x 14) x 14
        self.conv1_batchnorm = torch.nn.BatchNorm2d(num_features=self.numChannels1)

        # use normal initialization for conv1:
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1_batchnorm.weight, 0.5)
        torch.nn.init.zeros_(self.conv1_batchnorm.bias)

        self.conv2 = torch.nn.Conv2d(
            self.numChannels1, self.numChannels2, 3, padding=1, bias=False
        )  # <- out: 16 x (nof_digits x 14) x 14
        self.conv2_batchnorm = torch.nn.BatchNorm2d(num_features=self.numChannels2)

        # use normal initialization for conv2:
        torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2_batchnorm.weight, 0.5)
        torch.nn.init.zeros_(self.conv2_batchnorm.bias)

        self.fc1 = torch.nn.Linear(self.numChannels2 * (nof_digits * 7) * 7, 256)
        self.fc2 = torch.nn.Linear(256, nof_digits)

    def forward(self, x):
        x = self.conv1_batchnorm(self.conv1(x))
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = self.conv2_batchnorm(self.conv2(x))
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def training(
        epochs,
        train_loader,
        model,
        loss_fn,
        optimizer,
        device,
        show_progress=False,
        L2_regularization=False,
        L1_regularization=False,
        L2_lambda=0.001,
        L1_lambda=0.001,
    ):
        l2_norm = 0
        l1_norm = 0
        model.train()
        for epoch in range(1, epochs + 1):
            loss_train = 0.0
            for imgs, labs, mpoints in train_loader:
                imgs = imgs.to(device)
                mpoints = mpoints.to(device)

                y_mpoints = model(imgs)
                loss = loss_fn(y_mpoints, mpoints)

                if L2_regularization:
                    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                    loss = loss + L2_lambda * l2_norm

                if L1_regularization:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + L1_lambda * l1_norm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss.item()

            if epoch == 1 or epoch % 10 == 0:
                print(f"len train loader: {len(train_loader)}")
                print(
                    f"{datetime.datetime.now()} Epoch {epoch} Training loss {loss_train/ len(train_loader)}"
                )
                if (
                    show_progress
                ):  # prints out some weights to see if anything happens at all:
                    print(model.conv1.weight[0][0:10])

    def validate(model, train_loader, val_loader, loss_fn):
        model.eval()
        for name, loader in [("train", train_loader), ("val", val_loader)]:
            loss = 0
            for imgs, _, mpoints in loader:
                with torch.no_grad():
                    y_mpoints = model(imgs)
                    loss += loss_fn(y_mpoints, mpoints).item()

            print(f"Loss {name}: {loss/len(loader)}")


"""
check the points, where the images are merged ("merge_points"), by drawing a grey vertical line
"""


def display_merge_lines(image_record, model):
    mnist_width = 28
    t_img = image_record["multi_img"]
    y_mps = model(t_img.unsqueeze(0)).detach().numpy()  # unsqueeze to make it a batch
    img = TT.ToPILImage()(t_img)
    plt.title(mr["multi_img_label"])
    merge_points = mr["merge_points"]

    draw = ImageDraw.Draw(img)
    for ymp in [mp for mp in np.nditer(y_mps)]:
        draw.line([(ymp, 0), (ymp, mnist_width)], width=1, fill=128)
    plt.imshow(img, cmap="gray")


# from multidigit_end2end.ipynb ---------------------------------------------------------

import torchvision.transforms as TT


def cut_to_mnist(multi_img, merge_points):
    mnist_images = []
    top_vertical = 0
    top_horizontal = 0
    crop_height = 28
    crop_width = 0
    for mp in merge_points:
        mp = int(mp)
        crop_width = mp - top_horizontal
        cut_image = TT.functional.crop(
            multi_img,
            top=top_vertical,
            left=top_horizontal,
            height=crop_height,
            width=crop_width,
        )
        top_horizontal = mp
        # fill or cut the image to 1 x 28 x 28 mnist shape:
        if crop_width < 28:
            black_fill = torch.zeros([1, 28, 28 - cut_image.shape[2]])
            img = torch.cat([cut_image, black_fill], dim=2)
        elif crop_width > 28:
            img = TT.functional.crop(cut_image, top=0, left=0, height=28, width=28)

        mnist_images.append(img)

    return mnist_images


# from single_mnist.ipynb ----------------------------------------------------------------------

import torch.nn.functional as F


class SingleDigitMNISTNet(torch.nn.Module):
    def __init__(self):
        super(SingleDigitMNISTNet, self).__init__()
        # input MNIST images for nof_digits digit-image: 1 x nof_digitsx28 x nof_digitsx28
        self.numChannels1 = 8
        self.numChannels2 = 32
        self.nof_classes = 10  # figures 0...9 (and "not recognized"? - is not in the trainingset/-labels)
        self.conv1 = torch.nn.Conv2d(
            1, self.numChannels1, 5, padding=2, bias=False
        )  # <- out: 8 x 28 x 28  # <- max-pooling out: 8 x 14 x 14
        self.conv1_batchnorm = torch.nn.BatchNorm2d(num_features=self.numChannels1)

        # use normal initialization for conv1:
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1_batchnorm.weight, 0.5)
        torch.nn.init.zeros_(self.conv1_batchnorm.bias)

        self.conv2 = torch.nn.Conv2d(
            self.numChannels1, self.numChannels2, 3, padding=1, bias=False
        )  # <- out: 32 x 14 x 14
        self.conv2_batchnorm = torch.nn.BatchNorm2d(num_features=self.numChannels2)

        # use normal initialization for conv2:
        torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2_batchnorm.weight, 0.5)
        torch.nn.init.zeros_(self.conv2_batchnorm.bias)

        self.fc1 = torch.nn.Linear(self.numChannels2 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, self.nof_classes)

    def forward(self, x):
        x = self.conv1_batchnorm(self.conv1(x))
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = self.conv2_batchnorm(self.conv2(x))
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(
            x, dim=1
        )  # use log_softmax() (i.e. log(softmax()) ) to use NLLLoss() as loss-function
        return x

    import datetime

    def training(
        epochs,
        train_loader,
        model,
        loss_fn,
        optimizer,
        device,
        show_progress=False,
        L2_regularization=False,
        L1_regularization=False,
        L2_lambda=0.001,
        L1_lambda=0.001,
    ):
        l2_norm = 0
        l1_norm = 0
        model.train()
        for epoch in range(1, epochs + 1):
            loss_train = 0.0
            for imgs, y in train_loader:
                imgs = imgs.to(device)
                y = y.to(device)

                yp = model(imgs)
                loss = loss_fn(yp, y)

                if L2_regularization:
                    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                    loss = loss + L2_lambda * l2_norm

                if L1_regularization:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + L1_lambda * l1_norm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train += loss.item()

            if epoch == 1 or epoch % 10 == 0:
                print(f"len train loader: {len(train_loader)}")
                print(
                    f"{datetime.datetime.now()} Epoch {epoch} Training loss {loss_train/ len(train_loader)}"
                )
                # if(show_progress): # prints out some weights to see if anything happens at all:
                #    print(model.conv1.weight[0][0:10])

    def validate(model, train_loader, val_loader, loss_fn):
        model.eval()
        for name, loader in [("train", train_loader), ("val", val_loader)]:
            equals = 0
            nof_y = 0
            for imgs, y in loader:
                with torch.no_grad():
                    yp = model(imgs)
                    y_class = torch.argmax(yp, dim=1)
                    # print(f"y.shape: {y.shape}")
                    # print(f"y_class.shape: {y_class.shape}")
                    equals += torch.eq(y_class, y).sum()
                    nof_y += len(y)

            print(f"Accuracy {name}: {equals/nof_y}")
