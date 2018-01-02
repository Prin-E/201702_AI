# -*- coding: utf-8 -*-
# 9주차 : Multi Layer Perceptron
# XOR
import numpy as np
import random
import matplotlib.pyplot as plt

learning_rate = 1

# 시그모이드
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# gradient descent
def gd(x):
    return x * (1.0 - x)

# 뉴런 객체
class newron_t:
    def __init__(self):
        self.inputs = [0.0, 0.0, 0.0]
        self.weights = [0.0, 0.0, 0.0]
        self.bias = 0
        self.random()
    
    def output(self):
        v = self.inputs[0] * self.weights[0] + self.inputs[1] * self.weights[1] + self.inputs[2] * self.inputs[2] + self.bias
        return sigmoid(v)

    def random(self):
        self.weights[0] = random.uniform(0.0, 1.0)
        self.weights[1] = random.uniform(0.0, 1.0)
        self.weights[2] = random.uniform(0.0, 1.0)
        self.bias = random.uniform(0.0, 1.0)

    def adjust(self, error):
        self.weights[0] = self.weights[0] + error * self.inputs[0] * learning_rate
        self.weights[1] = self.weights[1] + error * self.inputs[1] * learning_rate
        self.weights[2] = self.weights[2] + error * self.inputs[2] * learning_rate
        self.bias = self.bias + error * learning_rate

# 파일 출력 & print 동시에 하기 편하게
def print_log(file, str):
    file.write(str + "\n")
    print(str)

if __name__ == '__main__':
    train_log = open("train_log.txt", "w")
    print_log(train_log, "maximum accepted error : 0.1")

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

    #inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
    #solution = [0.0, 1.0, 1.0, 0.0]

    n1 = newron_t()
    n2 = newron_t()
    n3 = newron_t()
    out = newron_t()

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
            # forward propogation
            n1.inputs = [inputs[i][0], inputs[i][1], 0]
            n2.inputs = [inputs[i][0], inputs[i][1], 0]
            n3.inputs = [inputs[i][0], inputs[i][1], 0]
            out.inputs = [n1.output(), n2.output(), n3.output()]

            train_log.write("%.4f xor %.4f = %.6f" % (inputs[i][0], inputs[i][1], out.output()))
            if abs(solution[i] - out.output()) < 0.1:
                train_log.write(" (correct)\n")
            else:
                train_log.write(" (bad)\n")
                error += 1

            # back propogation
            out_error = gd(out.output()) * (solution[i] - out.output())
            out.adjust(out_error)
            n1.adjust(gd(n1.output()) * out_error * out.weights[0])
            n2.adjust(gd(n2.output()) * out_error * out.weights[1])
            n3.adjust(gd(n3.output()) * out_error * out.weights[2])

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
            n1.random()
            n2.random()
            n3.random()
            out.random()
            learning_rate = 1
            continue

        # epoch 증가
        epoch = epoch + 1
        learning_rate = learning_rate * 0.99
        if epoch >= 201:
            loop = False

    print_log(train_log, "N : ")
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (n1.bias, n1.weights[0], n1.weights[1]))
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (n2.bias, n2.weights[0], n2.weights[1]))
    print_log(train_log, "| %8.4f %8.4f %8.4f |" % (n3.bias, n3.weights[0], n3.weights[1]))
    print_log(train_log, "O : ")
    print_log(train_log, "| %8.4f %8.4f %8.4f %8.4f |" % (out.bias, out.weights[0], out.weights[1], out.weights[2]))

    train_log.close()

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    solution = [0, 1, 1, 0]

    test_output = open("test_output.txt", "w")
    print_log(test_output, "maximum accepted error : 0.1")
    p_color = []

    for i in range(len(inputs)):
        n1.inputs = [inputs[i][0], inputs[i][1], 0]
        n2.inputs = [inputs[i][0], inputs[i][1], 0]
        n3.inputs = [inputs[i][0], inputs[i][1], 0]
        out.inputs = [n1.output(), n2.output(), n3.output()]
        
        log =  "%d xor %d = %d (test : %.6f, " % (inputs[i][0], inputs[i][1], solution[i], out.output())
        if abs(solution[i] - out.output()) < 0.1:
            log += "correct)"
            p_color.append("g*")
        else:
            log += "wrong)"
            p_color.append("r*")
        print_log(test_output, log)

    test_output.close()
