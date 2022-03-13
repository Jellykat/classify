from utils import read_spilt_data
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import LiverDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_imgs_path, train_imgs_label, test_imgs_path, test_imgs_label = read_spilt_data("E:/classproject/data/cut")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_data_set = LiverDataset(imgs_path=train_imgs_path,
                                  imgs_class=train_imgs_label,
                                  transform=transform)
    train_loader = DataLoader(train_data_set,
                              batch_size=9,
                              shuffle=True,
                              collate_fn=train_data_set.collate_fn)
    for step, data in enumerate(train_loader):
        imgs, labels = data


if __name__ == '__main__':
    main()
