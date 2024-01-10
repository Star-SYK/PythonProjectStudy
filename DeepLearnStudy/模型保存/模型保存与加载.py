
import joblib
from 回归.逻辑回归 import logistic_regression

if __name__ == "__main__":
    model_name = "model_CancerPrediction.pkl"
    model = logistic_regression()
    joblib.dump(model,model_name)
    estimator = joblib.load(model_name)
    print("逻辑回归权重系数为:\n", estimator.coef_)
    print("偏置为:\n", estimator.intercept_)
