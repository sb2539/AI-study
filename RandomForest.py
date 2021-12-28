import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

iris = datasets.load_iris()  # 데이터 읽어들임
print('Class names:', iris.target_names)  # iris target 이름 출력
print('target : [0:setosa, 1:versicolor, 2:virginica]') # 타겟 번호 지정
print('No. of Data:', len(iris.data))  # 데이터 개수 출력
print('Feature names :', iris.feature_names) # 속성 값 출력

data = pd.DataFrame({
    'sepal length': iris.data[:, 0], 'sepal width': iris.data[:,1], 'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3], 'species':iris.target
})
print(data.head())

x = data[['sepal length', 'sepal width', 'petal length', 'petal width']] # 입력
y = data['species'] # 출력
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 데이터 분할
print("No. of training data: ", len(x_train)) # 입력의 학습데이터 개수 출력
print("No. of test data: ", len(y_test)) # 출력의 테스트 데이터 개수 출력

forest = RandomForestClassifier(n_estimators=100) # 모델 생성
forest.fit(x_train, y_train) # 랜덤 포레스트 학습

y_pred = forest.predict(x_test) # 추론
print('Accuracy :', metrics.accuracy_score(y_test, y_pred)) # 예측값과 테스트 데이터 이용한 정확도 출력