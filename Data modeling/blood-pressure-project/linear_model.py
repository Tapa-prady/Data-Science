import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class LinearModel:
    def __init__(self, model_name="", degree=2, alpha=1.0):
        self.model_name = model_name
        self.degree = degree
        self.alpha = alpha
        self.model = None
        self.feature_names = None
        self.slope = None
        self.intercept = None
        self.rsquared = None

    def fit(self, x, y):
        x_frame = pd.DataFrame(x)
        y_values = pd.Series(y).astype(float).to_numpy()

        if x_frame.shape[1] == 1:
            self.feature_names = [x_frame.columns[0] if len(x_frame.columns) else "x"]
        else:
            self.feature_names = list(x_frame.columns)

        x_values = x_frame.astype(float).copy()
        self.model = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            Ridge(alpha=self.alpha),
        )
        self.model.fit(x_values, y_values)

        y_pred = self.model.predict(x_values)
        self.rsquared = r2_score(y_values, y_pred)
        self.intercept = self.model.named_steps["ridge"].intercept_
        self.slope = self.model.named_steps["ridge"].coef_[0] if len(self.model.named_steps["ridge"].coef_) > 0 else 0.0
        return self

    def predict(self, x):
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        x_frame = pd.DataFrame(x)
        if x_frame.shape[1] == 1 and self.feature_names:
            x_frame.columns = [self.feature_names[0]]
        return self.model.predict(x_frame.astype(float))

    def plot_model(self, x_min, x_max, color="black", num_points=200):
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        if self.feature_names is None or len(self.feature_names) != 1:
            raise ValueError("Plotting is only supported for single-feature models.")

        x_values = np.linspace(x_min, x_max, num_points)
        x_frame = pd.DataFrame({self.feature_names[0]: x_values})
        y_values = self.predict(x_frame)
        plt.plot(x_values, y_values, color=color)

    def print_model_info(self):
        print(f"LinearModel({self.model_name}):")
        print(f"Polynomial degree: {self.degree}")
        print(f"Features: {self.feature_names}")
        print(f"Intercept: {self.intercept}")
        print(f"R-squared: {self.rsquared}")