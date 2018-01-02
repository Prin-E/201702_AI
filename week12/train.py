# -*- coding: utf-8 -*-
# 12주차 : Convolutional Neural Network
# MNIST, Train

import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import numpy as np
import random
import mnistkit2 as mnistkit
import copy

# mini-batch(1000)으로 진행
learning_rate = 1.0
n_batch = 1000

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

# mnist 학습
def train_mnist(train_log_file, data, label):
    print_log(train_log_file, "learning_rate = " + str(learning_rate))
    print_log(train_log_file, "batch = " + str(n_batch))

    # 레이어 설정
    model = km.Sequential()
    model.add(kl.Conv2D(input_shape=(mnistkit._N_ROW, mnistkit._N_COL, 1), filters=32,
                        kernel_size=(8, 8), strides=2, activation='relu'))
    model.add(kl.Flatten())
    model.add(kl.Dense(units=128, activation='relu'))
    model.add(kl.Dense(units=10, activation='softmax')) # sigmoid보다 빠르게 학습이 가능함.

    # 네트워크 컴파일
    model.compile(loss='mean_squared_error',
              optimizer=ko.SGD(lr=learning_rate, decay=0.01, momentum=0.9),
              metrics=['accuracy'])

    # 모델 구조 그리기
    ku.plot_model(model, 'model.png')

    # 학습 진행, 정확도가 90% 이상일 경우 중단
    epoch = 1
    max_epoch = 100
    while epoch <= max_epoch:
        res = model.fit(data, label, epochs=1, batch_size=n_batch)
        acc = res.history['acc'][0]
        loss = res.history['loss'][0]
        print_log(train_log_file, "epoch %03d --- loss=%.4f, acc=%.2f%%" % (epoch, loss, acc*100))
        if acc >= 0.9:
            break
        epoch += 1

    # 가장 좋은 값 출력
    weights = model.get_weights()
    best_param_file = open("best_param.pkl", "w")
    best_param_file.write("%d\n" % len(weights))
    for i in range(len(weights)):
        weight = weights[i]
        for j in range(len(weight.shape)):
            best_param_file.write("%d " % weight.shape[j])
        best_param_file.write("\n")
        weight1d = weight.reshape(-1).astype("float")
        for j in range(len(weight1d)):
            best_param_file.write("%10.5f " % weight1d[j])
        best_param_file.write("\n")
    best_param_file.close()
    print_log(train_log_file, "->best_param.pkl")

if __name__ == '__main__':
    # 학습 log 파일
    train_log_file = open("train_log.txt", "w")
    print_log(train_log_file, "loading MNIST train datas...")

    # 데이터 불러오기
    data, label = mnistkit.loadMNIST_Train()
    
    # 훈련 시작
    train_mnist(train_log_file, data, label)

    # log 파일 닫기
    train_log_file.close()
