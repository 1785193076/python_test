# main.py

import numpy as np
from tensorflow.keras.datasets import mnist # 只用它来加载数据
import matplotlib.pyplot as plt

# --- 1. 准备教材：加载和预处理数据 ---

# 加载数据，它已经被分为训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据探索（可选，但推荐）
print("训练图片数据的形状:", train_images.shape) # (60000, 28, 28) -> 6万张 28x28像素的图片
print("训练标签数据的形状:", train_labels.shape) # (60000,) -> 6万个标签
print("第一个训练标签是:", train_labels[0]) # 看看第一个数字是什么

# 显示第一张图片
# plt.imshow(train_images[0], cmap='gray')
# plt.title(f"这是一个数字: {train_labels[0]}")
# plt.show()

# 数据预处理
# a. 扁平化：将 28x28 的图片矩阵转换成 784x1 的向量
#    我们的神经网络输入层需要一个一维向量，而不是一个二维矩阵
#    想象一下把一张像素方格纸从上到下、从左到右拉成一条长长的像素带
num_pixels = train_images.shape[1] * train_images.shape[2] # 28 * 28 = 784
train_images_flat = train_images.reshape(train_images.shape[0], num_pixels).T
test_images_flat = test_images.reshape(test_images.shape[0], num_pixels).T
print("扁平化后训练图片的形状:", train_images_flat.shape) # (784, 60000)

# b. 归一化：将像素值从 0-255 缩放到 0-1
#    这有助于加快训练速度，提高模型的稳定性
train_images_normalized = train_images_flat / 255.
test_images_normalized = test_images_flat / 255.

# c. 标签预处理：One-Hot 编码
#    原始标签是一个数字，比如 5。我们需要把它变成一个向量，其中第5个位置是1，其余都是0。
#    例如: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#    这是因为我们神经网络的输出层有10个神经元，分别代表0到9的概率。
def one_hot(labels, num_classes):
    # 创建一个全零矩阵，行数是类别数，列数是标签数
    one_hot_labels = np.zeros((num_classes, labels.size))
    # np.arange(labels.size) -> [0, 1, 2, ...]
    # labels -> [5, 0, 4, ...]
    # 这行代码在 one_hot_labels[5, 0]、one_hot_labels[0, 1]、one_hot_labels[4, 2]... 的位置上标1
    one_hot_labels[labels, np.arange(labels.size)] = 1
    return one_hot_labels

train_labels_one_hot = one_hot(train_labels, 10)
test_labels_one_hot = one_hot(test_labels, 10)
print("One-Hot编码后第一个标签:\n", train_labels_one_hot[:, 0])


# --- 2. 搭建“大脑”：初始化网络参数 ---

def init_params():
    # 隐藏层 (第一层)
    # W1 的形状是 (128, 784)，因为每个隐藏层神经元(128)都需要连接到每个输入像素(784)
    W1 = np.random.randn(128, 784) * 0.01
    b1 = np.zeros((128, 1))

    # 输出层 (第二层)
    # W2 的形状是 (10, 128)，因为每个输出层神经元(10)都需要连接到每个隐藏层神经元(128)
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2


# --- 3. 开始学习 Part A: 前向传播 ---

# 激活函数：为神经网络引入非线性，让它能学习更复杂的模式
def relu(Z):
    # 如果 Z > 0，则返回 Z；否则返回 0
    return np.maximum(Z, 0)


def softmax(Z):
    # 将一组数字转换为概率分布
    # e^Z / sum(e^Z)
    expZ = np.exp(Z - np.max(Z))  # 减去max(Z)是为了数值稳定性，防止溢出
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    # 第一层计算
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)

    # 第二层计算
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)  # A2 是最终的预测概率

    return Z1, A1, Z2, A2


# --- 3. 开始学习 Part B: 反向传播 ---

def deriv_relu(Z):
    # ReLU的导数：如果 Z > 0，导数是1；否则是0
    return Z > 0


def backward_prop(Z1, A1, Z2, A2, W2, X, Y_one_hot):
    m = Y_one_hot.shape[1]  # 样本数量

    # 对输出层的计算
    # dZ2 是预测与真实值之间的误差
    dZ2 = A2 - Y_one_hot
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # 对隐藏层的计算
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# --- 4. 整合与考试 ---

def get_predictions(A2):
    # A2 是概率分布，取概率最大的那个神经元的索引，即为预测的数字
    return np.argmax(A2, 0)


def get_accuracy(predictions, labels):
    # print(predictions, labels)
    return np.sum(predictions == labels) / labels.size


def gradient_descent(X, Y, Y_labels, alpha, epochs):
    W1, b1, W2, b2 = init_params()
    for i in range(epochs):
        # 前向传播
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # 反向传播
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)

        # 更新参数
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # 每10轮打印一次训练进度
        if i % 10 == 0:
            print("Epoch:", i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y_labels)
            print(f"训练集准确率: {accuracy * 100:.2f}%")

    return W1, b1, W2, b2


# --- 开始训练！---
# alpha是学习率，决定了我们每一步调整参数的幅度
# epochs是训练轮数，表示我们要把整个训练集看多少遍
trained_W1, trained_b1, trained_W2, trained_b2 = gradient_descent(
    train_images_normalized, train_labels_one_hot, train_labels, alpha=0.1, epochs=500
)


# --- 用测试集进行最终考试 ---
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = test_images_normalized[:, index, None]  # 取一列并保持其二维形状
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = test_labels[index]

    print("预测结果: ", prediction[0])
    print("真实标签: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# 在整个测试集上评估准确率
test_predictions = make_predictions(test_images_normalized, trained_W1, trained_b1, trained_W2, trained_b2)
test_accuracy = get_accuracy(test_predictions, test_labels)
print(f"----------\n最终在测试集上的准确率: {test_accuracy * 100:.2f}%")

# 测试几个例子
test_prediction(0, trained_W1, trained_b1, trained_W2, trained_b2)
test_prediction(1, trained_W1, trained_b1, trained_W2, trained_b2)
test_prediction(100, trained_W1, trained_b1, trained_W2, trained_b2)