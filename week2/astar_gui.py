# 2주차 : A*
##
# BFS 샘플 코드를 수정해서 적용함.
# 버튼 설명
# [run] : 1 step씩 이동, 길찾기 완료후 경로 출력
# [init] : 초기 상태로 되돌리기

import tkinter as Tk

map = []
map.append(list(range(1, 6)))
map.append(list(range(6, 11)))
map.append(list(range(11, 16)))
map.append(list(range(16, 21)))
map.append(list(range(21, 26)))

# 막힌 곳은 0으로 표시
map[0][3] = 0
map[1][1] = 0
map[2][1] = 0
map[1][3] = 0
map[2][3] = 0
map[4][3] = 0

# 노드의 좌표로부터 이름 반환
def getNodeName(location):
    return map[location[0]][location[1]]
    
# 해당 노드로 이동 가능한지 확인
def isExist(location, toVisit, alreadyVisited):
    if location[0] < 0:
        return False
    
    if location[1] < 0:
        return False
    
    if location[0] > 4:
        return False

    if location[1] > 4:
        return False
        
    # 막힌 곳 판정
    if getNodeName(location) == 0:
        return False
        
    # 이미 방문해야 할 목록에 들어있는지 판정
    for astar_node in toVisit:
        if location == astar_node:
            return False
        
    # 이미 방문했던 곳 판정
    if location in alreadyVisited:
        return False
        
    return True
    
# 대상 노드로 이동하기 위한 휴리스틱 계산
def heuristic(location, destination):
    hx = abs(destination[0] - location[0])
    hy = abs(destination[1] - location[1])
    return hx + hy
    
# 큐에 삽입할 A* 노드 생성
# [g+h,h,g,[x,y],[prev_x,prev_y]]
def make_astar_node(location, prev_location, destination, g):
    astar_node = [0, heuristic(location, destination), g, location, prev_location]
    astar_node[0] = astar_node[1] + astar_node[2]
    return astar_node
    
# 윈도우 콜백 클래스
class App:
    # A* 초기화
    def astar_reset(self):
        self.start = [2, 0]  # 11
        self.end = [2, 4]    # 15
        self.depth = 1
        self.alreadyVisited = []
        self.toVisit = []
        
        # 이전 경로를 저장하기 위한 map
        self.prev_node = []
        self.prev_node.append(list(range(1, 6)))
        self.prev_node.append(list(range(6, 11)))
        self.prev_node.append(list(range(11, 16)))
        self.prev_node.append(list(range(16, 21)))
        self.prev_node.append(list(range(21, 26)))
        
        # GUI map 사각형 색상 초기화
        for row in range(0, len(self.gui_map_rect)):
            for col in range(0, len(self.gui_map_rect[row])):
                if map[row][col] != 0:
                    self.canvas.itemconfig(self.gui_map_rect[row][col], fill='white')
                self.canvas.itemconfig(self.gui_map_text[row][col], text=map[row][col])
        
        # GUI 라벨 초기화
        self.label['text'] = u'탐색 횟수: 0'
        
        astar_node = make_astar_node(self.start, self.start, self.end, 0)
        self.toVisit.append(astar_node)
        
    def __init__(self, master):
        # 맵을 그릴 캔버스 생성
        self.canvas = Tk.Canvas(master, width = 800, height = 600)
        self.canvas.pack()
        
        # 버튼 생성
        self.button = Tk.Button(master, text = 'run', command = self.run)
        self.button.pack()
        self.button2 = Tk.Button(master, text = 'init', command = self.astar_reset)
        self.button2.pack()
        
        # 탐색 횟수 출력
        self.label = Tk.Label(master, text = u'탐색 횟수: 0')
        self.label.pack()
        
        # GUI 표시용 사각형, 텍스트 map
        self.gui_map_rect = []
        self.gui_map_rect.append(list(range(1, 6)))
        self.gui_map_rect.append(list(range(6, 11)))
        self.gui_map_rect.append(list(range(11, 16)))
        self.gui_map_rect.append(list(range(16, 21)))
        self.gui_map_rect.append(list(range(21, 26)))
        self.gui_map_text = []
        self.gui_map_text.append(list(range(1, 6)))
        self.gui_map_text.append(list(range(6, 11)))
        self.gui_map_text.append(list(range(11, 16)))
        self.gui_map_text.append(list(range(16, 21)))
        self.gui_map_text.append(list(range(21, 26)))
        
        # 맵 그리기
        for row in range(len(map)):
            for col in range(len(map[0])):
                if map[row][col] == 0:
                    fillColor = 'black'
                else:
                    fillColor = 'white'
                
                self.gui_map_rect[row][col] = self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
                self.gui_map_text[row][col] = self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = str(map[row][col]))
        
        # A* 초기화
        self.astar_reset()
        
    # A* 경로 텍스트로 출력
    def show_astar_path(self, location):
        path = str(getNodeName(location))
        self.canvas.itemconfig(self.gui_map_rect[location[0]][location[1]], fill='green')
        while True:
            prev_location = self.prev_node[location[0]][location[1]]
            if prev_location == location:
                break
            path = str(getNodeName(prev_location)) + '->' + path
            self.canvas.itemconfig(self.gui_map_rect[prev_location[0]][prev_location[1]], fill='green')
            location = prev_location
        return path
    
    # A* 알고리즘 반복 수행
    def run(self):
        # 앞으로 방문해야 할 노드가 남아있으면 루프 반복
        if len(self.toVisit) != 0:
            # 방문해야 할 노드 목록의 첫 번째로 현재 노드 이동
            current = self.toVisit.pop(0)
            
            # 현재 노드 칠하기
            current_location = current[3]
            prev_location = current[4]
            row = current_location[0]
            col = current_location[1]
            
            # 이전 경로를 참고하여 진행 방향 표시
            state = u'(start)'
            if prev_location[0] > current_location[0]:
                state = u'↑'
            if prev_location[0] < current_location[0]:
                state = u'↓'
            if prev_location[1] < current_location[1]:
                state = u'→'
            if prev_location[1] > current_location[1]:
                state = u'←'
            if current_location == self.end:
                state = u'(end)'
                
            # 휴리스틱 값 표시
            state += '\n' + str([current[0], current[1], current[2]])
            
            # 색상, 텍스트 변경하기
            self.canvas.itemconfig(self.gui_map_rect[row][col], fill='red')
            self.canvas.itemconfig(self.gui_map_text[row][col], text=str(map[row][col]) + '\n\n' + state)
            
            # 이전 경로 저장
            self.prev_node[row][col] = prev_location
            
            # 현재 노드의 자식 노드(인접 노드)를 방문해야 할 목록에 추가
            childList = []
            childList.append([current_location[0], current_location[1] - 1])
            childList.append([current_location[0] + 1, current_location[1]])
            childList.append([current_location[0], current_location[1] + 1])
            childList.append([current_location[0] - 1, current_location[1]])
            
            for child in childList:
                # 갈 수 있는 노드인 경우에만 추가
                if isExist(child, self.toVisit, self.alreadyVisited) == True:
                    candidate_node = make_astar_node(child, current_location, self.end, self.depth)
                    self.toVisit.append(candidate_node)
                    
                    # GUI 갱신
                    self.canvas.itemconfig(self.gui_map_rect[child[0]][child[1]], fill='yellow')
                    candidate_text = str(map[child[0]][child[1]]) + '\n\n\n' + str([candidate_node[0], candidate_node[1], candidate_node[2]])
                    self.canvas.itemconfig(self.gui_map_text[child[0]][child[1]], text=candidate_text)
            
            # 이미 방문한 노드에 현재 노드 추가
            self.alreadyVisited.append(current_location)
            
            self.label['text'] = u'탐색 횟수 : %d' % (self.depth)
            self.depth = self.depth + 1
            self.toVisit.sort()
            
            nodeName = getNodeName(current_location)
            print(nodeName, self.toVisit, self.alreadyVisited)
            
            # 목표 지점에 도달할 경우 탐색 중지
            if self.end == current_location:
                self.label['text'] = self.label['text'] + '\n' + self.show_astar_path(current_location) # 경로 표시
                self.toVisit = [] # 단순히 toVisit 목록을 비워서 다음단계 진행을 멈추도록 함.
                return

# 메인
root = Tk.Tk()
app = App(root)
root.mainloop()
