# -*- coding: utf-8 -*-
# 12주차 : Convolutional Neural Network
# MNIST, Test

import matplotlib.pyplot as plt
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import numpy as np
import random
import mnistkit2 as mnistkit

learning_rate = 0.01
n_batch = 1000
n_weight = 10

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

# mnist 학습
def test_mnist(output_file, data, label):
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


    # best_param.pkl 불러오기
    weights = []
    best_param_file = open("best_param.pkl", "r")
    num_weight = int(best_param_file.readline())
    for i in range(num_weight):
        shape_line = best_param_file.readline()
        shape_token = shape_line.split()
        shape = []
        num_element = 1
        for j in range(len(shape_token)):
            shape_element = int(shape_token[j])
            shape.append(shape_element)
            num_element *= shape_element
        
        element_line = best_param_file.readline()
        element_token = element_line.split()
        element = []
        for j in range(len(element_token)):
            element_value = float(element_token[j])
            element.append(element_value)
        weight = np.array(element).reshape(shape)
        weights.append(weight)
    best_param_file.close()

    model.set_weights(weights)

    # 평가 시작
    res = model.predict(data, batch_size=n_batch)
    
    # 오류 계산용
    label_errors = np.zeros(n_weight)
    num_labels = np.zeros(n_weight)

    for k in range(len(data)):
        s_label = np.argmax(label[k])
        output_file.write("sample %4d, label=%d -> " % (k, s_label))
        num_labels[s_label] = num_labels[s_label] + 1
        o_label = np.argmax(res[k])

        output_file.write("output=%d " % o_label)
        if s_label != o_label:
            output_file.write("(wrong)\n")
            label_errors[s_label] = label_errors[s_label] + 1
        else:
            output_file.write("(correct)\n")   
    output_file.write("\n")

    # 라벨당 오류 출력
    for i in range(n_weight):
        log = "label %2d - error : %.2f%%" % (i, float(label_errors[i]) / float(num_labels[i]) * 100.0)
        print_log(output_file, log)
    
    # 총 오류 출력
    log = "total - error : %.2f%%" % (float(np.sum(label_errors)) / float(np.sum(num_labels)) * 100.0)
    print_log(output_file, log)

if __name__ == '__main__':
    # 출력 파일
    output_file = open("test_output.txt", "w")

    # 데이터 불러오기
    print_log(output_file, "loading MNIST test datas...")
    data, label = mnistkit.loadMNIST_Test()

    # 테스트 시작
    test_mnist(output_file, data, label)

    # 파일 닫기
    output_file.close()
