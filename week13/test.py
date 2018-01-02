# -*- coding: utf-8 -*-
# 13주차 : Convolutional Neural Network, Feature Map
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

# feature map 그리기
def draw_feature_map(dest_file, feature_map):
    fig, ax = plt.subplots()
    ax.imshow(feature_map, cmap='gray')
    plt.savefig(dest_file)
    
    # flush
    plt.cla()
    plt.clf()
    plt.close()

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

# mnist 학습
def test_mnist(output_file, data, label):
    # 레이어 설정
    inputFeat = kl.Input(shape=(mnistkit._N_ROW, mnistkit._N_COL, 1))
    conv = kl.Conv2D(filters=32, kernel_size=(8, 8), strides=2, activation='relu')(inputFeat)
    flatten = kl.Flatten()(conv)
    dense = kl.Dense(units=128, activation='relu')(flatten)
    dense2 = kl.Dense(units=10, activation='softmax')(dense) # sigmoid보다 빠르게 학습이 가능함.

    model = km.Model(inputs=[inputFeat], outputs=[dense2])
    conv_model = km.Model(inputs=[inputFeat], outputs=[conv])

    # 네트워크 컴파일
    model.compile(loss='mean_squared_error',
              optimizer=ko.SGD(lr=learning_rate, decay=0.01, momentum=0.9),
              metrics=['accuracy'])

    # best_param.h5 불러오기
    model.load_weights("best_param.h5")

    # 평가 시작
    res = model.predict(data, batch_size=n_batch)
    res2 = conv_model.predict(data, batch_size=n_batch)
    
    # 오류 계산용
    label_errors = np.zeros(n_weight)
    num_labels = np.zeros(n_weight)

    # Feature-Map 생성용
    label_feat1 = [] # 2
    label_feat2 = [] # 5

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

        # Feature-Map 출력을 위한 데이터 index 저장
        if s_label == 2:
            if len(label_feat1) < 3:
                label_feat1.append(k)
        if s_label == 5:
            if len(label_feat2) < 3:
                label_feat2.append(k)

    output_file.write("\n")

    # 라벨당 오류 출력
    for i in range(n_weight):
        log = "label %2d - error : %.2f%%" % (i, float(label_errors[i]) / float(num_labels[i]) * 100.0)
        print_log(output_file, log)
    
    # 총 오류 출력
    log = "total - error : %.2f%%" % (float(np.sum(label_errors)) / float(np.sum(num_labels)) * 100.0)
    print_log(output_file, log)

    # Feature-Map 출력
    total_feat_sample_list = []
    total_feat_sample_list.extend(label_feat1)
    total_feat_sample_list.extend(label_feat2)
    total_label_feat = [2, 2, 2, 5, 5, 5]
    #print res2.shape
    for i in range(len(total_feat_sample_list)):
        for filter_index in range(res2.shape[3]):
            class_label = total_label_feat[i]
            sample_index = total_feat_sample_list[i]

            dest_file = '%d_%d_%d.png' % (class_label, sample_index, filter_index)
            feature_map = res2[sample_index, :, :, filter_index]
            draw_feature_map(dest_file, feature_map)


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
