from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data # 입력
y = iris.target # 출력

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

abc = AdaBoostClassifier(n_estimators=100, learning_rate=1) # 모델 생성
model = abc.fit(X_train, y_train) # ada 학습

y_pred = model.predict(X_test) # 예측
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
