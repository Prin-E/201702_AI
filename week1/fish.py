# 1주차 : 어종 분류

# input_data.txt 파일 읽기
infile = open("input_data.txt", "r")
lines = infile.readlines()
infile.close()

# 한글이 제대로 출력되지 않아서 부득이하게 영어로 작성하였습니다...
print("Read from input_data.txt");
print("Categorizing by (body-tail)");
print("----------------------------------------");

# output_data.txt에 기록할 string 목록
outlines = []

for line in lines:
    # 한줄씩 읽기, 내용이 없는 경우 건너뛰기
    tokens = line.strip().split()
    if len(tokens) < 2:
        continue
    
    # 토큰 읽기
    # 문자열->정수로 변환
    body = int(tokens[0])
    tail = int(tokens[1])
    
    # body-tail <  70 = salmon
    # body-tail >= 70 = seabass
    fish = ""
    if body - tail < 70:
        fish = "salmon"
    else:
        fish = "seabass"
    
    # 출력 내용 작성
    outline = "body: " + str(body) + " tail: " + str(tail) + " ==> " + fish
    outlines.append(outline + "\n")
    print(outline);

print("----------------------------------------");
    
# output_data.txt 파일 쓰기
outfile = open("output_result.txt", "w")
outfile.writelines(outlines)
outfile.close()

print("-> output_result.txt");
