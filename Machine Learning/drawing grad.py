import numpy as np
import matplotlib.pyplot as plt

def f(x,y) :
    return 2*x**2 + 4*x*y + 5*y**2 - 6*x + 2*y + 10
def dx(x,y) :
    return  4*x + 4*y -6
def dy(x,y) :
    return 4*x + 10*y + 2

xi = np.linspace(-5, 20, 100) # -5에서 시작 20에서 끝 100개로 나눠서 배열 생성
yi = np.linspace(-6, 6, 100)

X, Y = np.meshgrid(xi, yi) # xi, yi로 좌표행렬 만듬
Z = f(X,Y)  # f 함수 대입

xj = np.linspace(-5, 20, 13)
yj = np.linspace(-6, 6, 7)
X1, Y1 = np.meshgrid(xj, yj) # xj, yj 로 좌표행렬 만듬
Dx = dx(X1, Y1)  # dx 값 구함
Dy = dy(X1, Y1)  # dy 값 구함

plt.figure(figsize=(10,5))  # plt 객치 크기
plt.contour(X, Y, Z, levels = np.logspace(0, 3, 10)) # 등고선 표현하기 (logspace 사용해서 하한 0, 상한 3 사이의 10개의 로그)
plt.quiver(X1, Y1, Dx, Dy, color='red', scale = 1000, minshaft = 4) # 화살표 그리기 (x1,y1은 벡터 위치, u,v는 벡터 성분)
plt.xlabel('x')
plt.ylabel('y')
plt.show()