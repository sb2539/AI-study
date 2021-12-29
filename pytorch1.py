from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist.data
y = mnist.target

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/7, random_state=0) # 학습, 검증 데이터 분할
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(list(map(int, y_train)))  # int 형 탠서위한 long탠서
y_test = torch.LongTensor(list(map(int, y_test)))

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

X_train = X_train.view(-1, 1, 28, 28).float() # 학습 데이터 첫번째 차원 토치에 맡기고 2번째 차원 1 3번째 차원 28 네번째 차원 28??
X_test = X_test.view(-1, 1, 28, 28).float()
print(X_train.shape) # 학습데이터 미니배치 60000개
print(X_test.shape)  # 검증 데이터 미니배치 10000개

train = TensorDataset(X_train, y_train)
test = TensorDataset(X_test, y_test)
BATCH_SIZE = 32
loader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
loader_test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

class CNN(nn.Module):   # cnn 모듈 생성
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x): # conv1-relu-conv2-maxpool-relu-conv3-maxpool-relu-fc1-relu-fc2-softmax
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p = 0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p = 0.5, training=self.training)
        x = x.view(-1, 3*3*64)  # x 를 (?,3*3*64) 크기로 변경
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def fit(model, loader_train):  # 학습 위한 함수
    optimizer = torch.optim.Adam(model.parameters()) # adam 사용
    error = nn.CrossEntropyLoss()  # 크로스엔트로피
    EPOCHS = 1
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(loader_train):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()  # grad 초기화
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 50 == 0:  # 50으로 나눠질 때 마다 출력
                print('에포크 : {} [{}/{} ({:.0f}%)]\t 손실함수 : {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(loader_train),
                    100.*batch_idx / len(loader_train),
                    loss.data,
                    correct*100./(BATCH_SIZE*(batch_idx+1))
                ))
def evaluate(model):  # 정확도 계산
    correct = 0
    for test_imgs, test_labels in loader_test:
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output, 1)[1]
        correct += (predicted == test_labels).sum()
    print("test data accuracy: {:3f}%".format(float(correct)/(len(loader_test)*BATCH_SIZE)))

cnn = CNN() # cnn 모델 객체 생성
evaluate(cnn) # 학습 전 정확도
fit(cnn, loader_train)
cnn.eval()
evaluate(cnn)
index = 10
data = X_test[index].view(-1, 1, 28, 28).float()
output = cnn(data)
print('{} 번째 학습데이터의 테스트 결과 : {}'.format(index, output))
_, predicted = torch.max(output, 1)
print('{} 번째 데이터 예측 : {}'.format(index, predicted.numpy()))
print('실제 레이블 : {}'.format((y_test[index])))