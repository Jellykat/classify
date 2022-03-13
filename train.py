import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from model import VGG
from dataset import LiverDataset
from utils import read_spilt_data


device = torch.device("cuda:1")
print(device)


# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()


def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    epoch_loss_group = []
    epoch_group = []
    for epoch in range(num_epochs):
        epoch_group.append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        print(dt_size)
        epoch_loss = 0
        for step, data in enumerate(dataload, start=0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # with torch.no_grad():
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/dt_size))
        epoch_loss_group.append(epoch_loss/dt_size)
    plt.plot(epoch_group, epoch_loss_group)
    plt.title("Loss")
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("/mnt/sdb/xuyanwen/loss.jpg")
    torch.save(model.state_dict(), r'/mnt/sdb/xuyanwen/weights.pth')
    save_path = './VGG_model_20epoch.pth'
    torch.save(model.state_dict(), save_path)
    return model


# 训练模型
def train():
    net = VGG(num_classes=4, init_weights=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_imgs_path, train_imgs_label, test_imgs_path, test_imgs_label = read_spilt_data("/mnt/sdb/xuyanwen/data/cut")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_data_set = LiverDataset(imgs_path=train_imgs_path,
                                  imgs_class=train_imgs_label,
                                  transform=transform)
    train_loader = DataLoader(train_data_set,
                              batch_size=25,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=train_data_set.collate_fn)
    train_model(net, criterion, optimizer, train_loader).to(device)


if __name__ == "__main__":

    train()


print('Finished Training')
