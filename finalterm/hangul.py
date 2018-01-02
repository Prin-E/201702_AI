# -*- coding: utf-8 -*-
# Final Term Project
# 한글 음절 분석기

import matplotlib.pyplot as plt
import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras.utils as ku
import numpy as np
import sys

learning_rate = 0.5
bitmap_size = 32
batch_size = 1175       # 2350/2=1175
num_train_data = 12
form_size = 6

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

# 학습 데이터 불러오기
def load_train_data():
    data_filenames = []
    for i in range(num_train_data):
        data_filenames.append("train-%d" % i)
    return load_data("hangul_data", data_filenames)

# 테스트 데이터 불러오기
def load_test_data(data_filenames):
    return load_data("hangul_data", data_filenames)

# 학습, 테스트 불러오기 공용 코드
def load_data(folder_name, data_filenames):
    bitmaps = []    # 비트맵 (32x32xn)
    characters = [] # 글자 (n)
    forms = []      # 음절 형태 (6xn)
    labels = []     # 초성,중성,종성 구성 (47xn)

    for i in range(len(data_filenames)):
        filename = data_filenames[i]
        file = open("./" + folder_name + "/" + filename + ".txt", "r")
        line = file.readline()
        num_syllabus = int(line)

        for j in range(num_syllabus):
            line = file.readline()
            tokens = line.strip().split()
            character = int(tokens[0])
            form = int(tokens[1])

            # form값에 따라 6개 값을 넣는다. (해당 인덱스에 1, 나머지는 0)
            for k in range(form_size):
                if k == form:
                    forms.append(1)
                else:
                    forms.append(0)
            
            # 초성, 중성, 종성 넣기
            labels.append(int(tokens[2]))
            labels.append(int(tokens[3]))
            labels.append(int(tokens[4]))

            # 32x32 비트맵 불러오기
            for k in range(bitmap_size):
                line = file.readline()
                bitmaps.extend([int(x) for x in line.split()])
            
            characters.append(character)
    
    # 비트맵을 nx32x32x1 numpy 배열로 변환 (0.0~1.0 정규화)
    bitmaps = np.array(bitmaps)
    bitmaps = bitmaps.astype('float') / 255.0
    bitmaps = bitmaps.reshape(-1, bitmap_size, bitmap_size, 1)

    # 음절형태 numpy 배열로 변환
    forms = np.array(forms)
    forms = forms.reshape(-1, form_size)
    
    # 초성, 중성, 종성 numpy 배열 변환
    labels = np.array(labels)
    labels = labels.reshape(-1, 3)
    
    return bitmaps, characters, forms, labels

# 학습, 테스트에 사용할 Keras 모델을 생성한다.
def get_keras_model():
    # 레이어 설정
    inputFeat = kl.Input(shape=(32, 32, 1))
    zp = kl.ZeroPadding2D(padding=(5,5), name='Zero(+5)')(inputFeat)
    conv = kl.Conv2D(filters=1, kernel_size=(11, 11), strides=1, activation='relu', name='Conv2D(1x11x11)')(zp)
    conv2 = kl.Conv2D(filters=3, kernel_size=(7, 7), strides=1, activation='relu', name='Conv2D(3x7x7)')(conv)
    flatten = kl.Flatten()(conv2)
    dense = kl.Dense(units=128, activation='relu', name='Dense-128')(flatten)
    dense2 = kl.Dense(units=32, activation='relu', name='Dense-32')(dense)
    dense3 = kl.Dense(units=6, activation='softmax', name='Dense-6')(dense2)
    model = km.Model(inputs=[inputFeat], outputs=[dense3])
    eval_model = km.Model(inputs=[inputFeat], outputs=[conv2])

    # 네트워크 컴파일
    model.compile(loss='mean_squared_error',
              optimizer=ko.SGD(lr=learning_rate, decay=0.004, momentum=0.8),
              metrics=['accuracy'])

    # 모델 구조 그리기
    ku.plot_model(model, 'model.png')
    
    return model, eval_model

# 한글 음절 학습을 진행한다.
def train_data():
    # 텍스트 출력용 파일
    log_file = open("train_log.txt", "w")

    # 데이터 불러오기
    print_log(log_file, "loading train datas...")
    bitmaps, characters, forms, labels = load_train_data()

    # keras CNN 불러오기
    print_log(log_file, "building keras model (CNN)...")
    model, eval_model = get_keras_model()

    # 학습 진행, 정확도가 99% 이상일 경우 중단
    epoch = 1
    max_epoch = 30
    while epoch <= max_epoch:
        res = model.fit(bitmaps, forms, epochs=1, batch_size=batch_size)
        acc = res.history['acc'][0]
        loss = res.history['loss'][0]
        print_log(log_file, "epoch %03d --- loss=%.4f, acc=%.2f%%" % (epoch, loss, acc*100))
        if acc >= 0.99:
            break
        epoch += 1
    
    # 가장 좋은 값 출력
    km.save_model(model, "best_param.h5")
    print_log(log_file, "->best_param.h5")

def test_data(data_filenames):
    # 텍스트 출력용 파일
    log_file = open("test_output.txt", "w")

    # 데이터 불러오기
    print_log(log_file, "loading test datas...")
    bitmaps, characters, forms, labels = load_test_data(data_filenames)

    # keras CNN 불러오기
    print_log(log_file, "building keras model (CNN)...")
    model, eval_model = get_keras_model()

    # best_param.h5 불러오기
    model.load_weights("best_param.h5")

    res = model.predict(bitmaps, batch_size=batch_size)
    res2 = eval_model.predict(bitmaps, batch_size=batch_size)

    #print res.shape
    #print res2.shape

    num_error = 0
    num_form = np.zeros(6)
    error_form = np.zeros(6)
    for i in range(len(forms)):
        # 문자 정보
        character = characters[i]
        s_form = np.argmax(forms[i])
        o_form = np.argmax(res[i])
        
        # 정답, 오류 판별하기
        result = "correct"
        num_form[s_form] += 1
        if s_form != o_form:
            num_error += 1
            error_form[s_form] += 1

        # 각 문자 테스트 결과 출력
        log_file.write("%d - form : %d (output : %d) -> (%s)\n" % (character, s_form, o_form, result))

        # 임의 문자의 feature map 출력
        '''
        if i % 200 == 0:
            for filter_index in range(res2.shape[3]):
                dest_file = 'feature_map/%d_%d_%d.png' % (s_form, character, filter_index)
                print dest_file
                feature_map = res2[i, :, :, filter_index]
                draw_feature_map(dest_file, feature_map)
        '''

    for i in range(num_form.shape[0]):
        ef = error_form[i] / float(num_form[i])
        print_log(log_file, "form %d (count:%d) -> error = %.2f%%" % (i, num_form[i], ef * 100.0))

    error = num_error / float(len(forms))
    print_log(log_file, "total -> error = %.2f%%" % (error * 100.0))

if __name__ == '__main__':
    is_test = False
    data_filenames = []

    if len(sys.argv) >= 2:
        # 학습 혹은 테스트 모드를 구별한다.
        if sys.argv[1] == '-test':
            is_test = True
        else:
            is_test = False
        
        # 테스트 모드일 경우 테스트 데이터 파일명을 임의 지정할 수 있다.
        if len(sys.argv) >= 3:
            for i in range(len(sys.argv)):
                if i >= 2:
                    data_filenames.append(sys.argv[i])
        else:
            print ("usage")
            print ("train : python hangul.py")
            print ("test  : python hangul.py -test [filenames...]")
            exit(0)

    if is_test:
        print ("run as test mode ...")
        test_data(data_filenames)
    else:
        print ("run as train mode ...")
        train_data()
