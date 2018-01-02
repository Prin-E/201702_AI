# -*- coding: utf-8 -*-
# 6주차 : Perceptron

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy
import copy
import sys
import os.path as op

def get_determinant(equation, point):
    determinant = equation[0] * point[0] + equation[1] * point[1] + equation[2] * point[2]
    #if equation[0] * equation[1] > 0:
    #    determinant = determinant * -1
    return determinant

def g(z):
    if z > 0:
        return 1
    return 0

# 텍스트 파일 불러오기
def load_data(filename):
    list = []
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        tokens = line.strip().split()
        list.append([float(tokens[0]), float(tokens[1])])
        file.close()
    return list

def train_perceptron(learning_rate):
    # 연어, 농어 데이터
    salmon = load_data('salmon_train.txt')
    seabass = load_data('seabass_train.txt')
    
    # [bias, 정답(salmon=0, seabass=1)] 추가
    for s in salmon:
        s.extend([1, 0])
    for s in seabass:
        s.extend([1, 1])
    
    # 출력 파일
    outname = 'train_log_%.4f.txt' % (learning_rate)
    outfile = open(outname, 'w')
    
    # 초기 방정식
    equation = []
    equation.append(random.uniform(-5.0, 5.0))
    equation.append(random.uniform(-5.0, 5.0))
    equation.append(random.uniform(-200, 200))
    #equation = [2.0, 4.0, -100.0]
    
    # ------------------
    log = 'initial equation: %.3fx%+.3fy%+.3f=0' % (equation[0], equation[1], equation[2])
    print(log)
    outfile.write(log + "\n")
    log = 'learning rate: %.4f' % (learning_rate)
    print(log)
    outfile.write(log + "\n")
    
    # 데이터
    data = []
    data.extend(salmon)
    data.extend(seabass)
    
    # 반복 횟수용
    t = 1
    flag = True
    
    # 가장 오류율이 낮은 방정식
    best_equation = [equation[0], equation[1], equation[2]]
    best_error = len(data)
    best_t = 0
    
    while flag:
        for i in range(len(data)):
            x = data[i]
            o = g(get_determinant(equation, x))
            #print get_determinant(equation, x), o, x[3], (x[3] - o)
            for j in range(len(equation)):
                equation[j] = equation[j] + learning_rate * (x[3] - o) * x[j]
        
        # 오류 계산
        error = 0
        for i in range(len(data)):
            x = data[i]
            o = g(get_determinant(equation, x))
            error += abs(x[3] - o)
        
        log = '(T=%3d) equation: %.3fx%+.3fy%+.3f=0, error: %.2f%%' % (t, equation[0], equation[1], equation[2], float(error)/len(data)*100)
        print(log)
        outfile.write(log + "\n")
        
        if error < best_error:
            best_equation = []
            best_equation.extend(equation)
            best_error = error
            best_t = t
        
        # 반복 횟수에 제한 주기
        # (학습 횟수가 5000회를 넘어섰거나 오류율이 10% 미만이거나 학습이 진척이 되지 않을 경우)
        t = t + 1
        if t > 5000 or error < len(data)/10 or (t-best_t) > 1000:
          flag = False
          break
            
    # 가장 오류율이 낮은 답
    log = '(best(T=%d)) equation: %.3fx%+.3fy%+.3f=0, error: %.2f%%' % (best_t, best_equation[0], best_equation[1], best_equation[2], float(best_error)/len(data)*100)
    print(log)
    outfile.write(log + "\n")
    
    # 출력 파일 닫기
    outfile.close()
    
    return best_equation
    
def test_perceptron(learning_rate, equation):
    # 연어, 농어 데이터
    salmon = load_data('salmon_test.txt')
    seabass = load_data('seabass_test.txt')
    
    # 출력파일
    outname = 'test_output_%.4f.txt' % (learning_rate)
    outfile = open(outname, 'w')

    # 오류율 계산용
    correct = 0
    wrong = 0
    cost = 0.0
    
    # 그래프용
    fig, ax = plt.subplots()
    
    # 연어
    correct_x = []
    correct_y = []
    wrong_x = []
    wrong_y = []
    cost_part = 0
    for data in salmon:
        x, y = data
        # 판별식으로 분류하기
        determinant = get_determinant(equation, [x, y, 1])
        if determinant >= 0:
            wrong_x.append(x)
            wrong_y.append(y)
            outfile.write('salmon (body:%.1f, tail:%.1f) : wrong, cost:%.2f\n' % (x, y, abs(determinant)))
            cost_part += abs(determinant)
        else:
            correct_x.append(x)
            correct_y.append(y)
            outfile.write('salmon (body:%.1f, tail:%.1f) : correct\n' % (x, y))
    # 연어 그래프 출력
    ax.plot(correct_x, correct_y, 'g^', label='salmon_correct', fillstyle='none')
    ax.plot(wrong_x, wrong_y, 'r^', label='salmon_wrong')
    
    # 연어 오류율 계산
    correct += len(correct_x)
    wrong += len(wrong_x)
    cost += cost_part
    outfile.write('--salmon-- correct:%d, wrong:%d, total:%d, cost:%.2f\n' % (len(correct_x), len(wrong_x), len(correct_x)+len(wrong_x), cost_part))
    outfile.write('\n')
    
    # 농어
    correct_x = []
    correct_y = []
    wrong_x = []
    wrong_y = []
    cost_part = 0
    for data in seabass:
        x, y = data
        # 판별식으로 분류하기
        determinant = get_determinant(equation, [x, y, 1])
        if determinant <= 0:
            wrong_x.append(x)
            wrong_y.append(y)
            outfile.write('seabass (body:%.1f, tail:%.1f) : wrong, cost:%.2f\n' % (x, y, abs(determinant)))
            cost_part += abs(determinant)
        else:
            correct_x.append(x)
            correct_y.append(y)
            outfile.write('seabass (body:%.1f, tail:%.1f) : correct\n' % (x, y))
    # 농어 그래프 출력
    ax.plot(correct_x, correct_y, 'bs', label='seabass_correct', fillstyle='none')
    ax.plot(wrong_x, wrong_y, 'rs', label='seabass_wrong')
    
    # 농어 오류율 계산
    correct += len(correct_x)
    wrong += len(wrong_x)
    cost += cost_part
    outfile.write('--seabass-- correct:%d, wrong:%d, total:%d, cost:%.2f\n' % (len(correct_x), len(wrong_x), len(correct_x)+len(wrong_x), cost_part))
    outfile.write('\n')
    
    # 총 오류율 출력
    error_rate = wrong / float(correct + wrong)
    graph_title = 'error: %.2f%%, cost: %.2f, equation: %.3fx%+.3fy%+.3f=0' % (error_rate*100, cost, equation[0], equation[1], equation[2])
    log = 'test case - ' + graph_title
    print(log)
    outfile.write('====================================================================\n')
    outfile.write('correct:%d, wrong:%d, total:%d\n' % (correct, wrong, correct+wrong))
    outfile.write(log)

    # 파일 닫기
    print("->" + outname)
    outfile.close()

    # 선 그리기
    line_x = [40, 130]
    line_y = [(-equation[0]*line_x[0]-equation[2])/equation[1], (-equation[0]*line_x[1]-equation[2])/equation[1]]
    ax.plot(line_x, line_y, 'k-', lw=2, label='classifier')
    
    # 그래프 grid, xy 축 설정 및 출력
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_xlabel('length of body')
    ax.set_ylabel('length of tail')
    ax.set_xlim((40, 130))
    ax.set_ylim((-5, 30))
    ax.set_title(log)
    plt.savefig('test_output_%.4f.png' % (learning_rate))
    plt.show()

def run_exp(learning_rate):    
    # 실행
    equation = train_perceptron(learning_rate)
    test_perceptron(learning_rate, equation)

if __name__ == '__main__':
    # command-line argument 받기
    argumentNum = len(sys.argv)
    
    if argumentNum == 2:
        learning_rate = float(sys.argv[1])
        run_exp(learning_rate)
    else:
        print ("Usage: %s [learning_rate]" % op.basename(sys.argv[0]))
