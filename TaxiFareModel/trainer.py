# imports
import pandas as pd
from encoders import TimeFeaturesEncoder, DistanceTransformer
from utils import compute_rmse
from data import get_data, clean_data, X_y_definition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe = Pipeline([('preproc', self.preproc_pipeline()),
                         ('linear_model', LinearRegression())])
        self.pipeline = pipe

    def preproc_pipeline(self):
        """preprocessing pipe line of time and distance columns"""
        preproc_pipe = ColumnTransformer([('distance', self.distance_pipeline(),
                                           ["pickup_latitude",
                                            "pickup_longitude",
                                            'dropoff_latitude',
                                            'dropoff_longitude']),
                                           ('time', self.time_pipeline(),
                                            ['pickup_datetime'])],
                                          remainder="drop")
        return preproc_pipe

    def time_pipeline(self):
        """defines the time pipeline"""
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        return time_pipe

    def distance_pipeline(self):
        """defines the distance pipeline"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        return dist_pipe


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df, test=False)
    # set X and y
    X, y = X_y_definition(df)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    pipeline = Trainer(X_train, y_train)
    pipeline.run()
    # evaluate
    evaluation_result = pipeline.evaluate(X_test, y_test)
    print(evaluation_result)
