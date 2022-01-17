import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.utils.data as Data
import torch
from torchvision import transforms as tfs
# 参数设置
torch.manual_seed(1)  # 随机种子设置， 每次初始化数值一样
# num_epochs = 100  # 训练次数
batch_size = 64
lr = 0.01
def read_image(path,img_size):
     #读取路径下所有文件
     train_x = []
     train_y = []
     test_x = []
     test_y = []
     n_class = 0
     
     perClassNum = 1600   # 每类图片数量
     a = 0
     for child_dir in os.listdir(path):  # 类
          # child_path = os.path.join(path, child_dir)
          a = a + 1
          # child_path = os.path.join(path, '/')
          child_path = os.path.join(path, '%d'%a)
          print(child_path)
          imgCount = 0
          testCount = 0
          for dir_image in tqdm(os.listdir(child_path)):  # 图片读取
               imgCount += 1
               if imgCount > perClassNum: # 每类用100张
                    break
               img = cv2.imread(child_path + '/' + dir_image)
               
               img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
               img = cv2.resize(img, (img_size, img_size))
               
               img = np.reshape(img, (img_size, img_size, 1))
               
               img = img.transpose(2, 0, 1)
               img = img.astype(float)
               img = img / 255 # 归一化
               np.random.shuffle(img)
               if testCount < 0.1 * perClassNum:   # 取30%作测试
                    testCount +=1
                    test_x.append(img)
                    test_y.append(n_class)
               else:
                    # img = img + np.random.randint(2,size=(img_size,img_size))-1
                    train_x.append(img)
                    train_y.append(n_class)
          n_class += 1

     # # one-hot
     # lb = LabelBinarizer().fit(np.array(range(n_class)))
     # train_y = lb.transform(train_y)
     # test_y = lb.transform(test_y)

     # 转成pytorch数据
     train_x = torch.tensor(train_x).to(torch.float32)
     train_y = torch.tensor(train_y).to(torch.long)
     train_dataset = Data.TensorDataset(train_x, train_y)
     train_loader = Data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

     test_x = torch.tensor(test_x).to(torch.float32)
     test_y = torch.tensor(test_y).to(torch.long)
     test_dataset = Data.TensorDataset(test_x, test_y)
     test_loader = Data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False)
     return train_loader, test_loader, n_class, train_dataset, test_dataset