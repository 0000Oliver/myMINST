import os
import numpy as np
import shutil
# 随机种子设置

random_state = 42
np.random.seed(random_state)

#原始kaggle 数据集
original_dataset_dir = "/Users/wangqiang/Source/kaggle/train"
total_num = int(len(os.listdir(original_dataset_dir))/2)
random_dix = np.array(range(total_num))
np.random.shuffle(random_dix)

#待处理数据ji
root = './data/kaggle'
if not os.path.exists(root):
    os.mkdir(root)
#训练集和测试集划分

sub_dirs = ['train','test']
animals = ['cats','dogs']
train_idx = random_dix[:int(total_num*0.9)]
test_idx = random_dix[int(total_num*0.9):]
numbers = [train_idx,test_idx]
for idx ,sub_dir in enumerate(sub_dirs):
    dir = os.path.join(root,sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for animal in animals:
        animal_dir = os.path.join(dir,animal)
        if not os.path.exists(animal_dir):
            os.mkdir(animal_dir)
        # print(animal[:-1])
        # print(numbers[idx])
        fnames = [animal[:-1]+ ".{}.jpg".format(i) for i in numbers[idx]]
        for fname in fnames:
            src = os.path.join(original_dataset_dir,fname)
            dst = os.path.join(animal_dir,fname)
            shutil.copyfile(src,dst)

        print(animal_dir+" totol_image : %d" %(len(os.listdir(animal_dir))))


