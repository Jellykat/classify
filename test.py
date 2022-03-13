from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import VGG
from dataset import LiverDataset
from utils import read_spilt_data


# 是否使用cuda
device = torch.device("cuda:0")
# if torch.cuda.is_available() else "cpu")
print(device)


def test():
    model = VGG(num_classes=4, init_weights=True).to(device)
    model.load_state_dict(torch.load('./VGG_model_20epoch.pth'))  # 载入训练好的模型
    model.eval()
    train_imgs_path, train_imgs_label, test_imgs_path, test_imgs_label = read_spilt_data("/mnt/sdb/xuyanwen/data/cut")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_data_set = LiverDataset(imgs_path=test_imgs_path,
                                 imgs_class=test_imgs_label,
                                 transform=transform)
    test_loader = DataLoader(test_data_set,
                             batch_size=9,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=test_data_set.collate_fn)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()
    # 循环读取数据
    with torch.no_grad():
        total_loss = 0.0
        correct = 0.0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            # 获取预测结果中每行数据概率最大的下标
            _, predicts = torch.max(output, dim=1)
            # 累计预测正确的个数
            correct += (predicts == target).sum().item()
        total_loss /= len(test_imgs_path)  # 平均损失
        acc = correct / len(test_imgs_path)  # 准确率
        writer.add_scalar('Test Loss', total_loss)  # 写入日志
        writer.add_scalar('Accuracy', acc)
        writer.flush()  # 刷新
        print("Test Loss : {:.4f}, Accuracy : {:.4f}".format(total_loss, acc))
        return total_loss, acc


if __name__ == "__main__":
    test()
