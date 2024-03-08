import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import time
import geniter
import TNCCA
import csv
import torch.utils.data as dataf
import get_cls_map
from tqdm import tqdm

dataset = 'houston'
Classes = 0
NC = 0

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def loadData():
    # 读入数据
    if dataset == 'houston':
        labels = sio.loadmat('/home/data/wxy/original/houston/Houston2013_gt.mat')['houston2013_gt']
        data = sio.loadmat('/home/data/wxy/original/houston/Houston2013_HSI.mat')['HSI']
        print('dataset : houston\n')
    # paviaU
    elif dataset == 'paviau':
        data = sio.loadmat('/home/data/wxy/original/PaviaU/PaviaU.mat')['paviaU']
        labels = sio.loadmat('/home/data/wxy/original/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        print('dataset : paviau\n')
    elif dataset == 'trento':
        data = sio.loadmat('/home/data/wxy/original/trento/Trento_Hsi.mat')['hsi_trento']
        labels = sio.loadmat('/home/data/wxy/original/trento/Trento_GT.mat')['gt_trento']
        print('dataset : trento\n')

    return data, labels


# 对高光谱数据 X 应用 PCA 变换
def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def sampling1(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m + 1):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m + 1):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def select_traintest(groundTruth):  # divide dataset into train and test datasets 获取的test和train数组里面分别存放的是gt里面对应的索引
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))
    if dataset == 'india':
        amount = [1, 43, 25, 7, 14, 22, 1, 14, 1, 29, 73, 18, 6, 38, 12, 3]
        print(f'train_amount:{amount}\n')
    elif dataset == 'trento':
        amount = [40, 29, 5, 91, 105, 31]
        print(f'train_amount:{amount}\n')
    elif dataset == 'houston':
        amount = [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7]
        print(f'train_amount:{amount}\n')
    elif dataset == 'paviau':
        amount = [66, 186, 21, 31, 13, 50, 13, 37, 9]
        print(f'train_amount:{amount}\n')

    # 找到各类别对应的索引
    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        # 索引放到loc类别对应的位置
        labels_loc[i] = indices
        nb_val = int(amount[i])
        # 往后取个数为amount中设置的样本作为train
        train[i] = indices[-nb_val:]
        # 取剩余的样本作为test
        test[i] = indices[:-nb_val]
    #    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def create_data_loader():
    # 地物类别
    # class_num = 6
    # 读入数据
    BATCH_SIZE = 64
    print('batchsize',BATCH_SIZE)
    X1, y = loadData()
    # 每个像素周围提取 patch 的尺寸
    patch_size_1 = 13
    patch_size_2 = 7
    PATCH_LENGTH_1 = int((patch_size_1 - 1) / 2)
    PATCH_LENGTH_2 = int((patch_size_2 - 1) / 2)
    print('Hyperspectral data shape: ', X1.shape)
    print('Label shape: ', y.shape)
    width = X1.shape[0]
    height = X1.shape[1]

    # 除去背景0的像素
    TOTAL_SIZE = 0
    ALL_SIZE = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            ALL_SIZE += 1
            if y[i, j] != 0:
                TOTAL_SIZE += 1
    print(f'TOTAL_SIZE:{TOTAL_SIZE},ALL_SIZE:{ALL_SIZE}')
    # 使用 PCA 降维，得到主成分的数量
    print('\n... ... PCA tranformation ... ...')
    pca_components = 30
    X1 = applyPCA(X1, numComponents=pca_components)
    print('Data shape after PCA: ', X1.shape)
    global NC
    NC = X1.shape[2]
    # 将数据变换维度： [m,n,k]->[m*n,k]
    X1_all_data = X1.reshape(np.prod(X1.shape[:2]), np.prod(X1.shape[2:]))
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(int)
    global Classes
    Classes = max(gt)
    print(f'class_num:{Classes}')


    # 数据标准化
    X1_all_data = preprocessing.scale(X1_all_data)
    data_X1 = X1_all_data.reshape(X1.shape[0], X1.shape[1], X1.shape[2])
    whole_data_X1 = data_X1
    padded_data_X1_1 = np.lib.pad(whole_data_X1, ((PATCH_LENGTH_1, PATCH_LENGTH_1), (PATCH_LENGTH_1, PATCH_LENGTH_1), (0, 0)),
                                'constant', constant_values=0)
    padded_data_X1_2 = np.lib.pad(whole_data_X1, ((PATCH_LENGTH_2, PATCH_LENGTH_2), (PATCH_LENGTH_2, PATCH_LENGTH_2), (0, 0)),
                                'constant', constant_values=0)
    print('\n... ... create train & test data ... ...')
    train_indices, test_indices = select_traintest(gt)
    # train_indices, test_indices = sampling(0.99, gt)
    _, all_indices = sampling1(1, gt)
    _, total_indices = sampling(1, gt)
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)

    print('\n-----Selecting Small Cube from the Original Cube Data-----')
    train_iter, test_iter, total_iter = geniter.generate_iter(
        TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
        ALL_SIZE, all_indices, whole_data_X1,  PATCH_LENGTH_1, padded_data_X1_1, PATCH_LENGTH_2, padded_data_X1_2,
        NC, BATCH_SIZE, gt)

    return train_iter, test_iter, total_iter, y, total_indices


def train(train_loader, test_loader, epochs, numi):
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    if dataset == 'india':
        num_classes = 16
    elif dataset == 'trento':
        num_classes = 6
    # Houston
    elif dataset == 'houston':
        num_classes = 15
    #paviaU
    elif dataset == 'paviau':
        num_classes = 9
    net = TNCCA.TNCCA(NC, num_classes).to(device)
    # net.load_state_dict(torch.load(f'./cls_params/SSFTTnet_pre_params.pkl'))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    # 开始训练
    total_loss = 0
    max_AA_OA = 0
    max_OA = 0
    max_AA = 0
    Train_time = 0
    Test_time = 0
    # min_loss = 100
    for epoch in range(epochs):
        net.train()
        tic1 = time.perf_counter()
        for i, (data1, data2, target) in enumerate(train_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data1, data2)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        toc1 = time.perf_counter()
        Train_time += toc1 - tic1
        tic2 = time.perf_counter()
        y_pred_test, y_test = test_epoch(device, net, test_loader)
        toc2 = time.perf_counter()
        Test_time += toc2 - tic2
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        if oa > max_OA:
            max_AA_OA = aa+oa
            max_OA = oa
            max_AA = aa
            torch.save(net.state_dict(), f'./cls_params/TNCCA_params{dataset}_{numi}.pkl')
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.6f] [max: %.6f] [maxOA: %.6f] [maxAA: %.6f]' % (epoch+1, total_loss / (epoch + 1), loss.item(), max_AA_OA, max_OA, max_AA))

        scheduler.step()
    print('Finished Training')

    return net, device, Train_time, Test_time


def test_epoch(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for data_1, data_2, labels in test_loader:
        data_1, data_2 = data_1.to(device), data_2.to(device)
        outputs = net(data_1, data_2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def test(device, net, test_loader,i):
    count = 0
    # 模型测试
    net.load_state_dict(torch.load(f'./cls_params/TNCCA_params{dataset}_{i}.pkl'))
    net.eval()
    y_pred_test = 0
    y_test = 0
    for data_1, data_2, labels in test_loader:
        data_1, data_2 = data_1.to(device), data_2.to(device)
        outputs = net(data_1, data_2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    if dataset == 'india':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
                        'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods',
                        'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']
    elif dataset == 'trento':
        target_names = ['Apple Tree', 'Building', 'Ground', 'Wood', 'Vineyard', 'Roads']
    elif dataset == 'houston':
        # target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    elif dataset == 'paviau':
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


if __name__ == '__main__':
    file_name = f"./cls_result/TNCCA_report_{dataset}.txt"
    header = ['OA', 'AA', 'kappa']
    file_name_csv = f"./cls_result/TNCCA_report_{dataset}.csv"
    with open(file_name_csv, 'w', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)

    train_loader, test_loader, all_data_loader, y_all, total_indices = create_data_loader()
    for i in range(1):
        net, device, Training_Time, Test_time = train(train_loader, test_loader, 500, i)
        y_pred_test, y_test = test(device, net, test_loader,i)
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        classification = str(classification)
        # 写入实验数据
        if dataset == 'india':
            keys = [oa, aa, kappa, each_acc[0], each_acc[1], each_acc[2], each_acc[3], each_acc[4], each_acc[5],
                each_acc[6], each_acc[7], each_acc[8], each_acc[9], each_acc[10], each_acc[11], each_acc[12],
                each_acc[13], each_acc[14], each_acc[15]]
        elif dataset == 'houston':
            keys = [oa, aa, kappa, each_acc[0], each_acc[1], each_acc[2], each_acc[3], each_acc[4], each_acc[5],
                    each_acc[6], each_acc[7], each_acc[8], each_acc[9], each_acc[10], each_acc[11], each_acc[12],
                    each_acc[13], each_acc[14], ]
        elif dataset == 'paviau':
            keys = [oa, aa, kappa, each_acc[0], each_acc[1], each_acc[2], each_acc[3], each_acc[4], each_acc[5],
                each_acc[6], each_acc[7], each_acc[8]]
        elif dataset == 'trento':
            keys = [oa, aa, kappa, each_acc[0], each_acc[1], each_acc[2], each_acc[3], each_acc[4], each_acc[5]]
        with open(file_name_csv, 'a+', encoding='utf-8', newline='') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(keys)

        with open(file_name, 'w') as x_file:
            x_file.write('{} Training_Time (s)'.format(Training_Time))
            x_file.write('\n')
            x_file.write('{} Test_time (s)'.format(Test_time))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(kappa))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(oa))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(aa))
            x_file.write('\n')
            x_file.write('{} Each accuracy (%)'.format(each_acc))
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))

        get_cls_map.generate_png(all_data_loader, net, y_all, device, total_indices, f"./classification_maps{i+1}")
