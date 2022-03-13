import torch.utils.data as data
import PIL.Image as Image
import torch


class LiverDataset(data.Dataset):
    def __init__(self, imgs_path: list, imgs_class: list, transform=None):
        self.imgs_path = imgs_path
        self.imgs_class = imgs_class
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        label = self.imgs_class[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs_path)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = tuple(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)
        return imgs, labels
