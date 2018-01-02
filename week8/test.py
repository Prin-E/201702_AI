# -*- coding: utf-8 -*-
# 8주차 : Single Layer Perceptron
# MNIST, Test

import matplotlib.pyplot as plt
import numpy as np
import random
import mnistkit

learning_rate = 0.01
n_weight = 10   # 1번째 index부터 쓰도록 (계산이 편하게)
n_element = mnistkit._N_PIXEL + 1
n_batch = 1000

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

# sigmoid 함수 (x)->(0.0~1.0)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# mnist 학습
def test_mnist(output_file, test_data):
    # 10x785 행렬 만들기
    weight_matrix = np.array(np.zeros(n_element * n_weight))
    weight_matrix = weight_matrix.reshape(n_weight, n_element)

    # best_param.pkl 불러오기
    pkl_file = open("best_param.pkl", "r")
    for i in range(n_weight):
        pkl_line = pkl_file.readline()
        pkl_line_weight = pkl_line.split()
        for j in range(n_element):
            weight_matrix[i][j] = float(pkl_line_weight[j])
    pkl_file.close()
    
    # 오류 계산용
    label_errors = np.zeros(n_weight)
    num_labels = np.zeros(n_weight)

    for k in range(len(test_data)):
        sample = test_data[k]
        s_data = sample.data
        s_label = sample.label
        output_file.write("sample %4d, label=%d -> " % (k, s_label))
        num_labels[s_label] = num_labels[s_label] + 1

        g_arr = np.dot(weight_matrix, s_data)
        o_arr = sigmoid(g_arr)
        o_label = np.argmax(o_arr)

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
    test_data = mnistkit.loadMNIST_Test()

    # 테스트 시작
    test_mnist(output_file, test_data)

    # 파일 닫기
    output_file.close()
