# -*- coding: utf-8 -*-
# 4주차 : GA(Genetic Algorithm)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import numpy
import copy
import sys
import os.path as op

class gene_t:
    def __init__(self):
        self.equation = [0, 0, 0]
        # 방정식의 초기값을 임의 구간 값으로 지정
        self.equation[0] = random.uniform(0.1, 5.0)
        self.equation[1] = random.uniform(-5.0, -0.1)
        self.equation[2] = random.uniform(-400.0, 200.0)
        self.cost = 100000.0
        self.error = 1.0
        self.priority = 0
    
    # A < B (정렬용)
    def __lt__(self, other):
        return self.cost < other.cost

    # A == B
    def __eq__(self, other):
        return self.equation[0] == other.equation[0] and self.equation[1] == other.equation[1] and self.equation[2] == other.equation[2]
    
    # A != B
    def __ne__(self, other):
        if self == other:
            return 0
        return 1
    
    # str()
    def __str__(self):
        return "error: %.2f%%, cost: %.2f, equation: %.3fx%+.3fy%+.3f=0\n" % (self.error, self.cost, self.equation[0], self.equation[1], self.equation[2])

    # print()
    def __repr__(self):
        return str(self)

    # 부모로부터 유전자 상속받기
    def inherit(self, p1, p2, mutProb):
        r = random.uniform(0.0, 1.0)
        # 돌연변이 판정 후 수식 갱신하기
        if r <= mutProb:
            # 돌연변이는 초기값과 동일한 임의 구간 값으로 지정
            self.equation[0] = random.uniform(0.1, 5.0)
            self.equation[1] = random.uniform(-5.0, -0.1)
            self.equation[2] = random.uniform(-400.0, 200.0)
        else:
            # Uniform Crossover
            r = random.uniform(0.0, 1.0)
            if r <= 0.5:
                self.equation[0] = p1.equation[0]
                self.equation[1] = p2.equation[1]
                self.equation[2] = p1.equation[2]
            else:
                self.equation[0] = p2.equation[0]
                self.equation[1] = p1.equation[1]
                self.equation[2] = p2.equation[2]

    # 비용 계산하기 (오류가 발생한 데이터의 determinant 값의 누적을 사용함)
    def get_cost(self, salmon, seabass):
        error = 0
        cost = 0
        cnt = len(salmon) + len(seabass)
        
        # 연어
        for s in salmon:
            determinant = self.equation[0] * s[0] + self.equation[1] * s[1] + self.equation[2]
            if determinant >= 0:
                error += 1
                cost += abs(determinant)

        # 농어
        for s in seabass:
            determinant = self.equation[0] * s[0] + self.equation[1] * s[1] + self.equation[2]
            if determinant <= 0:
                error += 1
                cost += abs(determinant)

        # 비용 갱신, 우선순위 계산
        self.cost = cost
        self.error = float(error) / float(cnt)
        
        # 우선순위 계산 (값이 높을 수록 우선순위가 높다고 설정)
        if float(error) / float(cnt) < 0.125:
            self.priority = 4
        else:
            if float(error) / float(cnt) < 0.25:
                self.priority = 3
            else:
                if float(error) / float(cnt) < 0.375:
                    self.priority = 2
                else:
                    self.priority = 1

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

def train_ga(popSize, eliteNum, mutProb):
    # 연어, 농어 데이터
    salmon = load_data('salmon_train.txt')
    seabass = load_data('seabass_train.txt')

    # 출력 파일
    outname = 'train_log_%d_%d_%.2f.txt' % (popSize, eliteNum, mutProb)
    outfile = open(outname, 'w')

    # 초기 유전자 생성
    gene_list = []
    for i in range(1, popSize):
        g = gene_t()
        g.get_cost(salmon, seabass)
        gene_list.append(g)
    gene_list.sort()

    # 반복 횟수용
    T = 100

    while T >= 0.001:
        # 다음 세대에 사용할 유전자 목록
        next_gen_gene_list = []
        
        # elite는 그대로 상속받기
        for i in range(1, eliteNum):
            next_gen_gene_list.append(gene_list.pop())
        
        # 룰렛
        roulette = []
        
        # 우선순위가 높은 유전자가 선택될 확률을 높이도록 우선순위 값만큼 중복으로 넣어주기
        for g in gene_list:
            for pr in range(1, g.priority):
                roulette.append(copy.deepcopy(g))

        # 유전자 상속 시작
        for i in range(eliteNum, popSize):
            p1 = random.choice(roulette)
            p2 = random.choice(roulette)
            g = gene_t()
            g.inherit(p1, p2, mutProb)
            g.get_cost(salmon, seabass)
            next_gen_gene_list.append(g)

        gene_list = next_gen_gene_list
        gene_list.sort()

        # 정보 출력
        best_gene = gene_list[0]
        log = '(T: %.3f) error: %.2f%%, cost: %.2f, equation: %.3fx%+.3fy%+.3f=0' % (T, best_gene.error*100, best_gene.cost, best_gene.equation[0], best_gene.equation[1], best_gene.equation[2])
        print(log)
        outfile.write(log + '\r\n')
        
        # T값 감소
        T = 0.99 * T
        
        # 엘리트 유전자가 모두 같은 경우 더 이상 계산이 필요없는 것으로 간주
        done = 1
        for i in range(2, eliteNum):
            if gene_list[i] != best_gene:
                done = 0
                break
        if done == 1:
            T = 0

    # 가장 좋은 유전자 출력, 파일 닫기
    best_gene = gene_list[0]
    log = '(result) error: %.2f%%, cost: %.2f, equation: %.3fx%+.3fy%+.3f=0' % (best_gene.error*100, best_gene.cost, best_gene.equation[0], best_gene.equation[1], best_gene.equation[2])
    print(log)
    outfile.write(log)

    # 파일 닫기
    print("->" + outname)
    outfile.close()
    
    # 가장 좋은 유전자 반환
    return best_gene

def test_ga(popSize, eliteNum, mutProb, best_gene):
    # 수식
    low_error_equation = best_gene.equation
    
    # 연어, 농어 데이터
    salmon = load_data('salmon_test.txt')
    seabass = load_data('seabass_test.txt')
    
    # 출력파일
    outname = 'test_output_%d_%d_%.2f.txt' % (popSize, eliteNum, mutProb)
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
        determinant = low_error_equation[0] * x + low_error_equation[1] * y + low_error_equation[2]
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
        determinant = low_error_equation[0] * x + low_error_equation[1] * y + low_error_equation[2]
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
    graph_title = 'error: %.2f%%, cost: %.2f, equation: %.3fx%+.3fy%+.3f=0' % (error_rate*100, cost, low_error_equation[0], low_error_equation[1], low_error_equation[2])
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
    line_y = [(-low_error_equation[0]*line_x[0]-low_error_equation[2])/low_error_equation[1], (-low_error_equation[0]*line_x[1]-low_error_equation[2])/low_error_equation[1]]
    ax.plot(line_x, line_y, 'k-', lw=2, label='classifier')
    
    # 그래프 grid, xy 축 설정 및 출력
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_xlabel('length of body')
    ax.set_ylabel('length of tail')
    ax.set_xlim((50, 125))
    ax.set_ylim((-5, 30))
    ax.set_title(log)
    plt.savefig('test_output_%d_%d_%.2f.png' % (popSize, eliteNum, mutProb))
    plt.show()

def run_exp(popSize, eliteNum, mutProb):    
    # 실행
    best_gene = train_ga(popSize, eliteNum, mutProb)
    test_ga(popSize, eliteNum, mutProb, best_gene)

if __name__ == '__main__':
    # command-line argument 받기
    argumentNum = len(sys.argv)
    
    if argumentNum == 4:
        popSize = int(sys.argv[1])
        eliteNum = int(sys.argv[2])
        mutProb = float(sys.argv[3])
        run_exp(popSize, eliteNum, mutProb)
    else:
        print ("Usage: %s [populationSize] [eliteNum] [mutationProb]" % op.basename(sys.argv[0]))
