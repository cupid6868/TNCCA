import torch
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm


def index_assignment(index, row, col, pad_length ,pos):
    new_assign = {}
    for counter, value in enumerate(index):
        # 获取索引对应的二维坐标（assign_0,assign_1）
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        pos[counter][0] = assign_0 - pad_length
        pos[counter][1] = assign_1 - pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign ,pos


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def select_small_cubic_1(data_size, data_indices, whole_data, patch_length_1, padded_data_1, patch_length_2, padded_data_2, dimension):
    # 先创建（30214 ，11 ，11 ，30）大小的空矩阵
    pos = np.zeros((data_size,2))
    #!----------------1--------------------------------!
    small_cubic_data_1 = np.zeros((data_size, 2 * patch_length_1 + 1, 2 * patch_length_1 + 1, dimension))
    # 根据（一维的索引，原图像的二维维度，patch）得到对应的二维坐标 data_assign.shape = (30124 , 2)
    data_assign_1, pos_1 = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length_1, pos)
    # 函数的作用就是根据每个坐标拿到11*11大小的小图片
    for i in tqdm(range(len(data_assign_1))):
        # 传入的参数分别为padding后的图像，横坐标，纵坐标，patch=5
        small_cubic_data_1[i] = select_patch(padded_data_1, data_assign_1[i][0], data_assign_1[i][1], patch_length_1)
        # small_cubic_data.shape = (30214 , 11 , 11 , 30)
    # !----------------2--------------------------------!
    small_cubic_data_2 = np.zeros((data_size, 2 * patch_length_2 + 1, 2 * patch_length_2 + 1, dimension))
    # 根据（一维的索引，原图像的二维维度，patch）得到对应的二维坐标 data_assign.shape = (30124 , 2)
    data_assign_2, pos_2 = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length_2, pos)
    # 函数的作用就是根据每个坐标拿到11*11大小的小图片
    for i in tqdm(range(len(data_assign_2))):
        # 传入的参数分别为padding后的图像，横坐标，纵坐标，patch=5
        small_cubic_data_2[i] = select_patch(padded_data_2, data_assign_2[i][0], data_assign_2[i][1], patch_length_2)
        # small_cubic_data.shape = (30214 , 11 , 11 , 30)
    return small_cubic_data_1, small_cubic_data_2, pos_1

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, ALL_SIZE, all_indices,
                  whole_data1, PATCH_LENGTH_1, padded_data1_1, PATCH_LENGTH_2, padded_data1_2, INPUT_DIMENSION, batch_size, gt):
    # gt_all = gt[all_indices] - 1
    gt_total = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1
    # 根据一维的索引每个像素都得到11*11大小的小图片 传入的参数分别是（ 除去0像素外的像素大小， 索引 ， HSI原图 ， padding后的HSI ， 输入的维度 ）
    X1_total_data_1, X1_total_data_2, total_pos = select_small_cubic_1(TOTAL_SIZE, total_indices, whole_data1, PATCH_LENGTH_1, padded_data1_1, PATCH_LENGTH_2, padded_data1_2, INPUT_DIMENSION)
    print(f'Shape:X1_total_data_1:{X1_total_data_1.shape}, X1_total_data_2:{X1_total_data_2.shape}')

    # 根据一维的索引每个像素都得到11*11大小的小图片 传入的参数分别是（ 包括0像素的像素大小， 索引 ， lidar原图 ， padding后的Lidar ， 输入的维度 ）
    # X1_all_data, _ = select_small_cubic_1(ALL_SIZE, all_indices, whole_data1, PATCH_LENGTH, padded_data1, INPUT_DIMENSION)
    # print(X1_all_data.shape)


    # 根据一维训练，测试的索引的每个像素都得到11*11大小的小图片 传入的参数分别是（ 像素大小， 索引 ， 原图 ， padding后的图 ， 输入的维度 ）
    X1_train_data, X2_train_data, _ = select_small_cubic_1(TRAIN_SIZE, train_indices, whole_data1, PATCH_LENGTH_1, padded_data1_1, PATCH_LENGTH_2, padded_data1_2, INPUT_DIMENSION)
    print(X1_train_data.shape)

    X1_test_data, X2_test_data, _ = select_small_cubic_1(TEST_SIZE, test_indices, whole_data1, PATCH_LENGTH_1, padded_data1_1, PATCH_LENGTH_2, padded_data1_2, INPUT_DIMENSION)
    print(X1_test_data.shape)


    X1_train_data , X2_train_data = X1_train_data.transpose(0, 3, 1, 2), X2_train_data.transpose(0, 3, 1, 2)
    X1_test_data, X2_test_data = X1_test_data.transpose(0, 3, 1, 2), X2_test_data.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', X1_train_data.shape,  X2_train_data.shape)
    print('after transpose: Xtest  shape: ', X1_test_data.shape, X2_test_data.shape)


    x1_tensor_train = torch.from_numpy(X1_train_data).type(torch.FloatTensor).unsqueeze(1)
    x2_tensor_train = torch.from_numpy(X2_train_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, x2_tensor_train, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(X1_test_data).type(torch.FloatTensor).unsqueeze(1)
    x2_tensor_test = torch.from_numpy(X2_test_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.LongTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, x2_tensor_test, y1_tensor_test)

    X1_total, X2_total = X1_total_data_1.transpose(0, 3, 1, 2), X1_total_data_2.transpose(0, 3, 1, 2)
    # X1_all = X1_all_data.transpose(0, 3, 1, 2)
    X1_total_tensor_data = torch.from_numpy(X1_total).type(torch.FloatTensor).unsqueeze(1)
    X2_total_tensor_data = torch.from_numpy(X2_total).type(torch.FloatTensor).unsqueeze(1)
    total_tensor_data_label = torch.from_numpy(gt_total).type(torch.LongTensor)
    torch_dataset_total = Data.TensorDataset(X1_total_tensor_data, X2_total_tensor_data, total_tensor_data_label)
    #
    # X1_all_tensor_data = torch.from_numpy(X1_all).type(torch.FloatTensor).unsqueeze(1)
    # X2_all_tensor_data = torch.from_numpy(X2_all_data).type(torch.FloatTensor).unsqueeze(1)
    # all_tensor_data_label = torch.from_numpy(gt_all).type(torch.LongTensor)
    # torch_dataset_all = Data.TensorDataset(X1_all_tensor_data, X2_all_tensor_data)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    total_iter = Data.DataLoader(
        dataset=torch_dataset_total,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    # all_iter = Data.DataLoader(
    #     dataset=torch_dataset_all,  # torch TensorDataset format
    #     batch_size=batch_size,  # mini batch size
    #     shuffle=False,
    #     num_workers=0,
    # )

    # return train_iter, test_iter, total_iter, all_iter, total_pos
    return train_iter, test_iter, total_iter# , y_test
