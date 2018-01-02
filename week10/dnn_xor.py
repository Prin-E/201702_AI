# -*- coding: utf-8 -*-
# 10주차 : Depp Neural Network
# XOR
import numpy as np
import random
import matplotlib.pyplot as plt

# 정답이 허용하는 오차를 0.1로 설정함
epsilon = 0.1

# 초기학습률
initial_learning_rate = 1.0

# 함수 type
func_sigmoid = 0
func_relu = 1

# Sigmoid
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# ReLU
def relu(x):
    if x > 0.0:
        return x
    return 0.0

# gradient descent of sigmoid
def gd_sigmoid(x):
    return x * (1.0 - x)

# gradient descent of ReLU
def gd_relu(x):
    if x > 0.0:
        return 1.0
    return 0.0

# 뉴런 객체
class newron_t:
    def __init__(self, length, func_type):
        self.inputs = np.zeros(length)
        self.weights = np.zeros(length)
        self.bias = 0
        self.func_type = func_type
        self.random()

    def output(self):
        val = self.bias
        for i in range(len(self.weights)):
            val = val + self.inputs[i] * self.weights[i]
        
        s = 0
        if self.func_type is func_sigmoid:
            s = sigmoid(val)
        if self.func_type is func_relu:
            s = relu(val)

        return s

    # gradient descent
    def gd(self):
        val = self.output()
        if self.func_type is func_sigmoid:
            val = gd_sigmoid(val)
        if self.func_type is func_relu:
            val = gd_relu(val)
        return val

    # initial random values
    def random(self):
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1.0, 1.0)
        self.bias = random.uniform(-1.0, 1.0)

    # train weights by error
    def adjust(self, error):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + error * self.inputs[i] * learning_rate
        self.bias = self.bias + error * learning_rate

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

if __name__ == '__main__':
    learning_rate = initial_learning_rate

    train_log = open("train_log.txt", "w")
    print_log(train_log, "maximum accepted error : %.2f" % epsilon)

    # Train 데이터는 학습이 제대로 이루어지도록 0,1 값 부근 (-0.1~+0.1)의 임의 실수로 잡음
    inputs = []
    solution = []
    for i in range(50):
        inputs.append([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
        inputs.append([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)])
        solution.append(0.0)
        solution.append(0.0)
    for i in range(50):
        inputs.append([random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)])
        inputs.append([random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)])
        solution.append(1.0)
        solution.append(1.0)
    
    # Two Layers; M(2x4), N(4x3)
    # 각 newron마다 bias값이 있으므로 실제 행렬로는 M(3x5), N(5x4)로 나타낼 수 있을 것이다.
    m1 = newron_t(2, func_relu)
    m2 = newron_t(2, func_relu)
    m3 = newron_t(2, func_relu)
    m4 = newron_t(2, func_relu)
    n1 = newron_t(4, func_sigmoid)
    n2 = newron_t(4, func_sigmoid)
    n3 = newron_t(4, func_sigmoid)
    out = newron_t(3, func_sigmoid)

    error = 0
    prev_error = 1000000
    loop = True
    epoch = 1
    none_error_loop = 0
    grow_error_loop = 0

    while loop:
        print_log(train_log, "epoch %4d ........" % epoch)
        error = 0

        for i in range(len(inputs)):
            # forward propogation -------------------------------
            m1.inputs = m2.inputs = m3.inputs = m4.inputs = inputs[i]
            n1.inputs = [m1.output(), m2.output(), m3.output(), m4.output()]
            n2.inputs = [m1.output(), m2.output(), m3.output(), m4.output()]
            n3.inputs = [m1.output(), m2.output(), m3.output(), m4.output()]
            out.inputs = [n1.output(), n2.output(), n3.output()]
            # ----------------------------------------------------

            train_log.write("%.4f xor %.4f = %.6f " % (inputs[i][0], inputs[i][1], out.output()))
            if abs(solution[i] - out.output()) < epsilon:
                train_log.write(" (correct)\n")
            else:
                train_log.write(" (bad)\n")
                error += 1

            # back propogation -----------------------------------
            # out
            out_error = out.gd() * (solution[i] - out.output())
            out.adjust(out_error)

            # out->N
            n1_error = n1.gd() * out_error * out.weights[0]
            n2_error = n2.gd() * out_error * out.weights[1]
            n3_error = n3.gd() * out_error * out.weights[2]
            n1.adjust(n1_error)
            n2.adjust(n2_error)
            n3.adjust(n3_error)
            
            # N->M
            m1_error = m1.gd() * (n1_error * n1.weights[0] + n2_error * n2.weights[0] + n3_error * n3.weights[0])
            m2_error = m2.gd() * (n1_error * n1.weights[1] + n2_error * n2.weights[1] + n3_error * n3.weights[1])
            m3_error = m3.gd() * (n1_error * n1.weights[2] + n2_error * n2.weights[2] + n3_error * n3.weights[2])
            m4_error = m4.gd() * (n1_error * n1.weights[3] + n2_error * n2.weights[3] + n3_error * n3.weights[3])
            m1.adjust(m1_error)
            m2.adjust(m2_error)
            m3.adjust(m3_error)
            m4.adjust(m3_error)
            # ----------------------------------------------------
            
        # 오류 출력
        print_log(train_log, " --- error : %.2f%%" % (100.0 * error / len(solution)))
        if error == 0:
            none_error_loop += 1
        else:
            none_error_loop = 0

        # 오류가 0%임을 확신할 때 (10 epoch 동안 오류가 나타나지 않음) 루프 종료
        if none_error_loop >= 10:
            loop = False
            break

        # 이전 오류율과 비교
        if prev_error <= error:
            grow_error_loop += 1
        else:
            grow_error_loop = 0
        prev_error = error

        # 오류가 더 이상 줄어들지 않을 때 (20 epoch 동안 줄어들지 않음) 초기값 다시 부여
        if grow_error_loop >= 20:
            print_log(train_log, "it's something wrong! re-setting initial weights!")
            epoch = 1
            none_error_loop = 0
            grow_error_loop = 0
            m1.random()
            m2.random()
            m3.random()
            n1.random()
            n2.random()
            n3.random()
            out.random()
            learning_rate = initial_learning_rate
            continue

        # epoch 증가
        epoch = epoch + 1
        learning_rate = learning_rate * 0.99
        if epoch >= 201:
            loop = False

    print_log(train_log, "M : ")
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (m1.bias, m1.weights[0], m1.weights[1]))
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (m2.bias, m2.weights[0], m2.weights[1]))
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (m3.bias, m3.weights[0], m3.weights[1]))
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (m4.bias, m4.weights[0], m4.weights[1]))
    print_log(train_log, "N : ")
    print_log(train_log, "| %8.4f %8.4f %8.4f %8.4f %8.4f |" % (n1.bias, n1.weights[0], n1.weights[1], n1.weights[2], n1.weights[3]))
    print_log(train_log, "| %8.4f %8.4f %8.4f %8.4f %8.4f |" % (n2.bias, n2.weights[0], n2.weights[1], n2.weights[2], n2.weights[3]))
    print_log(train_log, "| %8.4f %8.4f %8.4f %8.4f %8.4f |" % (n3.bias, n3.weights[0], n3.weights[1], n3.weights[2], n3.weights[3]))
    print_log(train_log, "O : ")
    print_log(train_log, "| %8.4f %8.4f %8.4f %8.4f |" % (out.bias, out.weights[0], out.weights[1], out.weights[2]))
    train_log.close()

    print("\n\nTEST!")

    # Test 데이터는 0 0, 0 1, 1 0, 1 1로 진행함
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    solution = [0, 1, 1, 0]

    test_output = open("test_output.txt", "w")
    p_color = []

    print_log(test_output, "maximum accepted error : %.2f" % epsilon)

    for i in range(len(inputs)):
        m1.inputs = m2.inputs = m3.inputs = m4.inputs = inputs[i]
        n1.inputs = [m1.output(), m2.output(), m3.output(), m4.output()]
        n2.inputs = [m1.output(), m2.output(), m3.output(), m4.output()]
        n3.inputs = [m1.output(), m2.output(), m3.output(), m4.output()]
        out.inputs = [n1.output(), n2.output(), n3.output()]
        
        log =  "%d xor %d = %d (test : %.6f, " % (inputs[i][0], inputs[i][1], solution[i], out.output())
        if abs(solution[i] - out.output()) < epsilon:
            log += "correct)"
            p_color.append("g*")
        else:
            log += "wrong)"
            p_color.append("r*")
        print_log(test_output, log)

    test_output.close()
