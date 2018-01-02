# -*- coding: utf-8 -*-
# 3주차 : SA(Simulated Annealing)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import numpy
import copy

if __name__ == '__main__':
    # 직선방정식 (ax+by+c=0)
    equation = [2.0, -1.0, -180.0]
    
    # 가장 낮은 오류 판별용
    low_error_equation = copy.deepcopy(equation)
    low_error = 1.0

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
        
    def train_sa():
        global equation
        global low_error_equation
        global low_error
    
        # 연어, 농어 데이터
        salmon = load_data('salmon_train.txt')
        seabass = load_data('seabass_train.txt')

        # 출력 파일
        outfile = open('train_log.txt', 'w')

        # 오류율 체크 (1.0 = 100%)
        prev_error = 1.0
        new_error = 0.0

        # 반복 횟수용
        T = 100

        while T >= 0.001:
            error = 0
            
            # 연어
            for s in salmon:
                determinant = equation[0] * s[0] + equation[1] * s[1] + equation[2]
                if determinant >= 0:
                    error += 1

            # 농어
            for s in seabass:
                determinant = equation[0] * s[0] + equation[1] * s[1] + equation[2]
                if determinant <= 0:
                    error += 1
            
            # 오류율 계산    
            new_error = error / float(len(salmon) + len(seabass))
            
            # 정보 출력
            log = '(T: %.3f) error: %.2f%%, equation: %.3fx%+.3fy%+.3f=0' % (T, new_error*100, equation[0], equation[1], equation[2])
            print(log)
            outfile.write(log + '\r\n')
            
            # 가장 낮은 오류 갱신하기
            if low_error > new_error:
                low_error_equation = copy.deepcopy(equation)
                low_error = new_error
            
            # 오류가 낮아지거나 이전 오류율와의 차이값이 일정 수준일 경우 직선방정식을 조금씩 이동하기
            error_delta = new_error - prev_error
            if error_delta <= 0:
                equation[0] += random.uniform(-0.01, 0.01)
                equation[1] += random.uniform(-0.01, 0.01)
                equation[2] += random.uniform(-10.0, 10.0)
            else:
                r = random.uniform(0.0, 1.0)
                if r < numpy.exp(-error_delta / T):
                    equation[0] += random.uniform(-0.01, 0.01)
                    equation[1] += random.uniform(-0.01, 0.01)
                    equation[2] += random.uniform(-10.0, 10.0)
            
            prev_error = new_error
            T = 0.99 * T

        # 가장 낮은 오류 출력, 파일 닫기
        log = 'lowest error: %.2f%%, equation: %.3fx%+.3fy%+.3f=0' % (low_error*100, low_error_equation[0], low_error_equation[1], low_error_equation[2])
        print(log)
        outfile.write(log)
        outfile.close()
    
    def test_sa():
        global low_error_equation
        
        # 연어, 농어 데이터
        salmon = load_data('salmon_test.txt')
        seabass = load_data('seabass_test.txt')
        
        # 출력파일
        outfile = open('test_output.txt', 'w')

        # 오류율 계산용
        correct = 0
        wrong = 0
        
        # 그래프용
        fig, ax = plt.subplots()
        
        # 연어
        correct_x = []
        correct_y = []
        wrong_x = []
        wrong_y = []
        for data in salmon:
            x, y = data
            # 판별식으로 분류하기
            determinant = low_error_equation[0] * x + low_error_equation[1] * y + low_error_equation[2]
            if determinant >= 0:
                wrong_x.append(x)
                wrong_y.append(y)
                outfile.write('salmon (body:%.1f, tail:%.1f) : wrong\n' % (x, y))
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
        outfile.write('--salmon-- correct:%d, wrong:%d, total:%d\n' % (len(correct_x), len(wrong_x), len(correct_x)+len(wrong_x)))
        outfile.write('\n')
        
        # 농어
        correct_x = []
        correct_y = []
        wrong_x = []
        wrong_y = []
        for data in seabass:
            x, y = data
            # 판별식으로 분류하기
            determinant = low_error_equation[0] * x + low_error_equation[1] * y + low_error_equation[2]
            if determinant <= 0:
                wrong_x.append(x)
                wrong_y.append(y)
                outfile.write('seabass (body:%.1f, tail:%.1f) : wrong\n' % (x, y))
            else:
                correct_x.append(x)
                correct_y.append(y)
                outfile.write('seabass (body:%.1f, tail:%.1f) : correct\n' % (x, y))
        # 농어 그래프 출력
        ax.plot(correct_x, correct_y, 'bs', label='seabass_correct', fillstyle='none')
        ax.plot(wrong_x, wrong_y, 'rs', label='seabass_wrong')
        
        # 선 그리기
        line_x = [50, 125]
        line_y = [(-low_error_equation[0]*line_x[0]-low_error_equation[2])/low_error_equation[1], (-low_error_equation[0]*line_x[1]-low_error_equation[2])/low_error_equation[1]]
        ax.plot(line_x, line_y, 'k-', lw=2, label='classifier')
        
        # 농어 오류율 계산
        correct += len(correct_x)
        wrong += len(wrong_x)
        outfile.write('--seabass-- correct:%d, wrong:%d, total:%d\n' % (len(correct_x), len(wrong_x), len(correct_x)+len(wrong_x)))
        outfile.write('\n')
        
        # 총 오류율 출력
        error_rate = wrong / float(correct + wrong)
        graph_title = 'error: %.2f%%, equation: %.3fx%+.3fy%+.3f=0' % (error_rate*100, low_error_equation[0], low_error_equation[1], low_error_equation[2])
        log = 'test case - ' + graph_title
        print(log)
        outfile.write('====================================================================\n')
        outfile.write('correct:%d, wrong:%d, total:%d\n' % (correct, wrong, correct+wrong))
        outfile.write(log)
        
        # 그래프 grid, xy 축 설정 및 출력
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_xlabel('length of body')
        ax.set_ylabel('length of tail')
        ax.set_xlim((50, 125))
        ax.set_ylim((-5, 30))
        ax.set_title(log)
        plt.savefig('test_output.png')
        plt.show()
        
        # 출력파일 닫기
        outfile.close()
        
    # 실행
    train_sa()
    test_sa()
