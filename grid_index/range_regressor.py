import numpy as np
from sklearn.tree import DecisionTreeRegressor

class RangeRegressor:
    def __init__(self, data):
        data_sorted, data_cdf = self.compute_dimension_cdf(data)

        data_sorted = np.array(data_sorted).reshape(-1, 1)
        data_cdf = np.array(data_cdf).reshape(-1, 1)

        regressor_depth = 4 # 5

        # Initialize and Fit regression model
        self.regressor = DecisionTreeRegressor(max_depth=regressor_depth)
        # Train the regressor for the dimension values and their CDF.
        self.regressor.fit(data_sorted, data_cdf)

        # plot the regressor just for info
        # self.plot_regressor(data_sorted, data_cdf)


    def predict_cdf(self, data):
        """
        Method that predicts the cdf for a given batch of values.
        The input to the method and predict is always a list of values, i.e.,
            [1, 3, 5] and not just a single value 1 for example.
        :param data: the values that we want to do the prediction for.
        :return: predicted cdf per value
        """
        return self.regressor.predict(data)

    def plot_regressor(self, data_sorted, data_cdf):
        """
        Plot the estimated cdf with the data.
        :param data_sorted:
        :param data_cdf:
        :return:
        """
        import matplotlib.pyplot as plt

        y_pred = self.predict_cdf(data_sorted)

        plt.plot(data_sorted, data_cdf, 'ro')
        plt.show()

        # Plot the results
        plt.figure()
        plt.scatter(data_sorted, data_cdf, s=20, edgecolor="black", c="darkorange", label="data")
        plt.plot(data_sorted, y_pred, color="cornflowerblue", label="max_depth=4", linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Regression")
        plt.legend()
        plt.show()

        y_pred = np.array(y_pred).reshape(-1, 1)
        from sklearn import tree
        tree.plot_tree(self.regressor)
        plt.show()

    def compute_dimension_cdf(self, dimension_values):
        x, counts = np.unique(dimension_values, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]
