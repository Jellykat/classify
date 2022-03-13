import os
import json
import random


def read_spilt_data(root: str, test_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root:{} does not exist.".format(root)

    # 遍历文件夹，一个文件夹一个类别
    cell_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    cell_class.sort()
    # 生成类别名称以及数字索引
    class_indices = dict((k, v) for v, k in enumerate(cell_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_imgs_path = []
    train_imgs_label = []
    test_imgs_path = []
    test_imgs_label = []
    eve_class_num = []
    supported = [".jpg", ".JPG"]
    # 遍历每个文件夹下的文件
    for cla in cell_class:
        cla_path = os.path.join(root, cla)
        # 遍历文件路径
        imgs = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                if os.path.splitext(i)[-1] in supported]
        # 获取索引
        img_class = class_indices[cla]
        # 记录样本数量
        eve_class_num.append(len(imgs))
        # 按比例随机划分测试集
        test_path = random.sample(imgs, k=int(len(imgs) * test_rate))

        for img_path in imgs:
            if img_path in test_path:   # 分别存入测试集和训练集
                test_imgs_path.append(img_path)
                test_imgs_label.append(img_class)
            else:
                train_imgs_path.append(img_path)
                train_imgs_label.append(img_class)
    print("{} images were found.".format(sum(eve_class_num)))
    print("{} images for training.".format(len(train_imgs_path)))
    print("{} images for testing.".format(len(test_imgs_path)))

    return train_imgs_path, train_imgs_label, test_imgs_path, test_imgs_label


if __name__ == "__main__":
    read_spilt_data(root="/mnt/sdb/xuyanwen/data/cut", test_rate=0.2)
