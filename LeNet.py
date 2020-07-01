import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
#使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #模型保存路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  #模型加载路径
opt = parser.parse_args()

# 超参数设置
EPOCH = 20  #遍历数据集次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 定义数据预处理方式

transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
                             ])
# 定义训练数据集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":
    print(net)
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct = 0.0
        total = 0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
        train_loss_list.append(sum_loss / total)
        train_accuracy_list.append(100 * correct / total)
        print('train %d epoch loss: %.3f acc: %.3f ' % (
            epoch + 1, sum_loss / total, 100 * correct / total))  #


        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            sum_loss = 0.0
            correct = 0.0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            test_loss_list.append( sum_loss / total)
            test_accuracy_list.append(100 * correct / total)
            print('test %d epoch loss: %.3f acc: %.3f ' % (
                epoch + 1, sum_loss / total, 100 * correct / total))  #
    x1 = range(0, EPOCH)
    y1 = train_loss_list
    y2 = test_loss_list
    y3 = train_accuracy_list
    y4 = test_accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.plot(x1, y2, 'o-')

    plt.legend(["train_loss", "test_loss"])
    plt.title('Loss vs. epoches')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y3, '.-')
    plt.plot(x1, y4, '.-')
    plt.legend(["train_accuracy", "test_accuracy"])
    plt.xlabel('Accuracy vs. epoches')
    plt.ylabel('Accuracy')
    plt.show()

    plt.savefig("minst_accuracy_loss.jpg")
    torch.save(net.state_dict(), '%s/minst_%03s.pth' % (opt.outf, "LeNet"))

    #net.load_state_dict(torch.load('%s/net_%03d.pth' % (opt.outf, 8)))
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('保存识别准确率为：%d%%' % ( (100 * correct / total)))

    # pretrained_net = torch.load(PATH)
    #
    # net2 = Net()
    #
    # net2.load_state_dict(pretrained_net)