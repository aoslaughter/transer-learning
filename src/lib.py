import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
from .visualize import imshow
from .train import train_model
from .pretrain_loader import pretrained_finetune, pretrained_fixed
from .visualize import visualize_model, visualize_model_predictions

# Data augmentation and normalization for training
# Normalization for validation

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
    ]),
}

data_dir = "data/hymenoptera_data"

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]
        ) for x in ["train", "val"]
    }

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4,
        shuffle=True, num_workers=4
        ) for x in ["train", "val"]
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in ["train", "val"]
}

class_names = image_datasets["train"].classes

if __name__ == "__main__":
    cudnn.benchmark = True
    plt.ion()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    # model_ft, criterion, optimizer_ft, exp_lr_scheduler = pretrained_finetune(device)
    # model_ft = train_model(
    #     model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=25
    # )

    # visualize_model(model_ft, dataloaders, class_names, device)

    model_conv, criterion, optimizer_conv, exp_lr_scheduler = pretrained_fixed(device)
    model_conv = train_model(
        model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=25
    )

    # visualize_model(model_conv, dataloaders, class_names, device)
    # plt.ioff()
    # plt.show()

    visualize_model_predictions(
        model_conv, device, data_transforms, class_names, img_path="data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg"
    )
    plt.ioff()
    plt.show()