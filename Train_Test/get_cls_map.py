import numpy as np
import matplotlib.pyplot as plt

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            # y[index] = np.array([147, 67, 46]) / 255.
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 1:
            # y[index] = np.array([0, 0, 255]) / 255.
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            # y[index] = np.array([255, 100, 0]) / 255.
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 3:
            # y[index] = np.array([0, 255, 123]) / 255.
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 4:
            # y[index] = np.array([164, 75, 155]) / 255.
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 5:
            # y[index] = np.array([101, 174, 255]) / 255.
            y[index] = np.array([165, 82, 41]) / 255.
        if item == 6:
            # y[index] = np.array([118, 254, 172]) / 255.
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 7:
            # y[index] = np.array([60, 91, 112]) / 255.
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 8:
            # y[index] = np.array([255, 255, 0]) / 255.
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 10:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 11:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 13:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 14:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 15:
            y[index] = np.array([101, 193, 60]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 255]) / 255.
    return y

def generate_png(total_iter, net, gt_hsi, devices, total_indices, path):
    pred_test = []
    for X1, X2, y in total_iter:
        X1 = X1.to(devices)
        X2 = X2.to(devices)
        net.eval()
        pred_test.extend(net(X1,X2).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300,
                       path + '.eps')
    classification_map(y_re, gt_hsi, 300,
                       path + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '_gt.png')
