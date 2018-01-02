# -*- coding: utf-8 -*-
# 8주차 : Single Layer Perceptron
# MNIST, Train

import matplotlib.pyplot as plt
import numpy as np
import random
import mnistkit
import copy

# mini-batch(1000)으로 진행
learning_rate = 0.1
n_weight = 10
n_element = mnistkit._N_PIXEL + 1
n_batch = 1000  # 코드 구조상 배치 1개는 지원안함. 최소 10개 이상 권장

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

# weight matrix 출력
def print_weight_matrix(file, weight_matrix):
    for i in range(n_weight):
        file.write("|")
        for j in range(n_element):
            file.write("%9.4f" % weight_matrix[i][j])
        file.write("|\n")

def weight_matrix_new():
    weight_matrix = np.array(np.zeros(n_element * n_weight))
    weight_matrix = weight_matrix.reshape(n_weight, n_element)
    # 초기값은 -1.0 ~ 1.0 사이로 할당
    for i in range(n_weight):
        for j in range(n_element):
            weight_matrix[i][j] = random.uniform(-1.0, 1.0)
    return weight_matrix

# sigmoid 함수 (x)->(0.0~1.0)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# mnist 학습
def train_mnist(train_log_file, train_data):
    global learning_rate

    print_log(train_log_file, "learning_rate = " + str(learning_rate))
    print_log(train_log_file, "batch = " + str(n_batch))

    # 10x(28*28+1) 가중치 행렬 만들기
    train_log_file.write("initial weight matrix : \n")
    weight_matrix = weight_matrix_new()
    print_weight_matrix(train_log_file, weight_matrix)

    # 정답 레이블용 (10x10) 행렬
    d_matrix = np.array(np.zeros(n_weight * n_weight))
    d_matrix = d_matrix.reshape(n_weight, n_weight)
    for i in range(n_weight):
        d_matrix[i][i] = 1

    error = 0
    low_error = 10000000
    n_loop = 0
    loop = True
    epoch = 1
    max_epoch = 120
    best_epoch = 1
    best_param = copy.deepcopy(weight_matrix)
    mse = 0
    momentum = 0.4
    prev_weight_matrix_delta = np.array(np.zeros(n_element * n_weight)).reshape(n_weight, n_element)

    train_log_file.write("epoch : %4d --------------------------------------------\n" % epoch)
    
    while loop:
        weight_matrix_delta = np.array(np.zeros(n_element * n_weight)).reshape(n_weight, n_element)
        #random.shuffle(train_data)

        for i in range(len(train_data)):
            sample = train_data[i]
            s_label = sample.label
            s_data = sample.data
            train_log_file.write("sample %4d, label=%d -> " % (i, s_label))
            
            #g_arr_2 = np.zeros(n_weight)
            #for j in range(n_weight):
            #    for k in range(n_element):
            #        g_arr_2[j] = g_arr_2[j] + weight_matrix[j][k] * s_data[k]
            g_arr = np.dot(weight_matrix, s_data)
            o_arr = sigmoid(g_arr)
            o_label = np.argmax(o_arr)

            # o_arr 파일 출력
            train_log_file.write("[")
            for j in range(n_weight):
                train_log_file.write("%7.4f" % o_arr[j])
            train_log_file.write("]")
            train_log_file.write(",argmax=%d" % o_label)

            #d_arr = d_matrix[s_label]
            #weight_matrix_delta += learning_rate * np.dot((d_arr.reshape(n_weight,1)-o_arr)*o_arr*(1.0-o_arr),s_data.reshape(1,-1))
            #mse = mse + np.dot((d_arr.reshape(n_weight,1)-o_arr).reshape(1,-1), (d_arr.reshape(n_weight,1)-o_arr))

            for j in range(n_weight):
                o = o_arr[j]
                d = d_matrix[s_label][j]
                weight_matrix_delta[j] = weight_matrix_delta[j] + learning_rate * (d - o) * s_data
                # TODO : gradient descent로 할 경우 모든 weight 값이 0.0으로 줄어듬. 원인 확인해야함.
                # 2017-11-08 : matrix곱은 문제 없는 것으로 보임. 각 weight의 bias 값이 비정상적으로 낮아지는 현상이 확인됨.
                #weight_matrix_delta[j] = weight_matrix_delta[j] + learning_rate * (d - o) * o * (1.0 - o) * s_data
                mse = mse + (d - o) * (d - o) * 0.5
            
            #print_weight_matrix(train_log_file, weight_matrix_delta)

            # 오류 체크
            if o_label != s_label:
                error = error + 1
                train_log_file.write(" (wrong)\n")
            else:
                train_log_file.write(" (correct)\n")
            
            # epoch 계산하기
            n_loop = n_loop + 1
            if n_loop % n_batch == 0:
                # 가장 좋은 파라미터 판별하기 
                if low_error > error:
                    low_error = error
                    best_epoch = epoch
                    best_param = copy.deepcopy(weight_matrix)
                
                # 가중치 더하기
                #train_log_file.write("--- weight matrix delta ---------------------------------\n")
                #print_weight_matrix(train_log_file, weight_matrix_delta)
                #train_log_file.write("---------------------------------------------------------\n")
                weight_matrix = weight_matrix + (prev_weight_matrix_delta * momentum + weight_matrix_delta) / float(n_batch)
                prev_weight_matrix_delta = weight_matrix_delta

                # 가중치 행렬 출력
                train_log_file.write("--- weight matrix ---------------------------------------\n")
                print_weight_matrix(train_log_file, weight_matrix)
                train_log_file.write("---------------------------------------------------------\n")
                
                # epoch 결과 출력
                print_log(train_log_file, "epoch %4d - mse : %8.3f, error : %.2f%%" % (epoch, mse, error / float(n_batch) * 100.0))

                # mse가 특정 값으로 수렴되는데 오류가 높은건 잘못된 지점으로 수렴하는 것이므로 초기값을 다시 부여해서 시작하도록 한다.
                if abs(mse - n_batch * 0.5) < 0.1 and (error / float(n_batch)) >= 0.825:
                    print_log(train_log_file, "Value is getting wrong! Traning is restarted with new weight matrix!")
                    train_log_file.write("initial weight matrix : \n")
                    weight_matrix = weight_matrix_new()
                    print_weight_matrix(train_log_file, weight_matrix)
                    # 좀 더 학습할 수 있도록 최대 epoch 증가
                    max_epoch += 40
                    # learning_rate도 다시 증가
                    learning_rate = 0.1

                # 다음 epoch에 사용하기 위해 값 설정
                error = 0
                epoch += 1
                mse = 0
                learning_rate = learning_rate * 0.99

                # 종료조건 설정 (너무 오래 진행되거나 오류가 15% 미만이면 종료)
                if epoch == max_epoch + 1 or (low_error / float(n_batch)) < 0.15:
                    loop = False
                    break
                else:
                    train_log_file.write("epoch : %4d (learning rate = %6.4f)------------------------------------\n" % (epoch, learning_rate))

    # 가장 좋은 값 출력
    pkl_file = open("best_param.pkl", "w")
    for i in range(n_weight):
        log = ""
        for j in range(n_element):
            log = log + "%8.4f" % best_param[i][j]
        pkl_file.write(log+"\n")
    pkl_file.close()
    print_log(train_log_file, "->best_param.pkl (best epoch : %d)" % best_epoch)
            
if __name__ == '__main__':
    # 학습 log 파일
    train_log_file = open("train_log.txt", "w")
    print_log(train_log_file, "loading MNIST train datas...")

    # 데이터 불러오기
    train_data = mnistkit.loadMNIST_Train()
    
    # 훈련 시작
    train_mnist(train_log_file, train_data)

    # log 파일 닫기
    train_log_file.close()
