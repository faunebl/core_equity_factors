import numpy as np
import polars as pl 

import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.decomposition import KernelPCA

#! Code has been adapted from code I wrote during my master's thesis, which is also available on my githug

def get_linear_regression(x_col: str, y_col: str, df: pl.DataFrame, model_type: str, intercept: bool = True, start_year: int =2008):
    if model_type == 'ols':
        model = LinearRegression(fit_intercept=intercept)
    elif model_type == 'ridge':
        model = Ridge(fit_intercept=intercept)
    elif model_type == 'lasso':
        model = Lasso(fit_intercept=intercept)
    elif model_type == 'elastic net':
        model = ElasticNet(fit_intercept=intercept)

    df = df.filter(pl.col('date').dt.year().ge(start_year))

    X = df.select(pl.col(x_col)).to_series().to_list()
    y = df.select(pl.col(y_col)).to_series().to_list()
    
    ind = -int(len(X)/3)

    reg = model.fit(X[:ind], y[:ind])

    out_of_sample = np.dot(a=reg.coef_, b=np.transpose(X[ind:]))
    in_sample = reg.predict(X[:ind])
    predicted = np.append(in_sample, out_of_sample)

    with_predicted = df.with_columns(pl.Series(name = "predictions", values = predicted))
    with_predicted = with_predicted.select('date', y_col, 'predictions') 

    sns.set_theme(context='notebook', rc={'figure.figsize':(20,5)}, palette=sns.cubehelix_palette(2), style='white')
    sns.lineplot(data = with_predicted.to_pandas().set_index('date'))
    plt.axvline(with_predicted.select('date')[ind].to_series().to_list()[0], 0,1, color = 'blue')
    plt.text(x=with_predicted.select('date')[ind].to_series().to_list()[0] + relativedelta(years=1), y = max(out_of_sample) + 0.05, s = "Out of Sample")

    # print("Coefficients: \n", reg.coef_)
    plt.text(x=with_predicted.select('date')[0].to_series().to_list()[0] - relativedelta(years=3, month=10),y=0,s=f"Mean squared error (in sample): {mean_squared_error(y[:ind], in_sample):.2f}")
    plt.text(x=with_predicted.select('date')[0].to_series().to_list()[0] - relativedelta(years=3, month=10),y=-0.01,s=f"Coefficient of determination (in sample): {r2_score(y[:ind], in_sample):.2f}")
    plt.text(x=with_predicted.select('date')[0].to_series().to_list()[0] - relativedelta(years=3, month=10),y=-0.02,s=f"Mean squared error (out of sample): {mean_squared_error(y[ind:], out_of_sample):.2f}")
    plt.text(x=with_predicted.select('date')[0].to_series().to_list()[0] - relativedelta(years=3, month=10),y=-0.03,s= f"Coefficient of determination (out of sample): {r2_score(y[ind:], out_of_sample):.2f}")

    print(f"Mean squared error (in sample): {mean_squared_error(y[:ind], in_sample):.2f}")
    print(f"Coefficient of determination (in sample): {r2_score(y[:ind], in_sample):.2f}")
    print(f"Mean squared error (out of sample): {mean_squared_error(y[ind:], out_of_sample):.2f}")
    print(f"Coefficient of determination (out of sample): {r2_score(y[ind:], out_of_sample):.2f}")
    print(f"Average of the prediction : {np.mean(out_of_sample)}")
    print(f"Average of the true values : {np.mean(y[ind:])}")
    print(f"Std of the prediction : {np.std(out_of_sample)}")
    print(f"Std of the true values : {np.std(y[ind:])}")

    return


def get_logistic_regression(df: pl.DataFrame, penalty: str, x_col: str, y_col: str = 'quantiles'):
    model = LogisticRegression(penalty=penalty, multi_class='multinomial')
    X = df.select(pl.col(x_col)).to_series().to_list()
    y = df.select(pl.col(y_col)).to_series().to_list()

    ind = -int(len(X)/3)

    X_test = X[ind:]
    X_train = X[:ind]

    y_test = y[ind:]
    y_train = y[:ind]

    reg = model.fit(X_train, y_train)

    out_of_sample = reg.predict(X_test)
    in_sample = reg.predict(X_train)

    print('Accuracy:', accuracy_score(y_train, in_sample))
    print('Precision:', precision_score(y_train, in_sample, average='weighted'))
    print('Recall:', recall_score(y_train, in_sample, average='weighted'))
    print('F1 score:', f1_score(y_train, in_sample, average='weighted'))
    print('Confusion matrix:\n', confusion_matrix(y_train, in_sample))

    print('Accuracy (Out of Sample):', accuracy_score(y_test, out_of_sample))
    print('Precision (Out of Sample):', precision_score(y_test, out_of_sample, average='weighted'))
    print('Recall (Out of Sample):', recall_score(y_test, out_of_sample, average='weighted'))
    print('F1 score (Out of Sample):', f1_score(y_test, out_of_sample, average='weighted'))
    print('Confusion matrix (Out of Sample):\n', confusion_matrix(y_test, out_of_sample))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    cm = confusion_matrix(y_train, in_sample)
    axs[0].matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[0].text(j, i, cm[i, j], ha='center', va='center')
    axs[0].set_xlabel('Predicted Class')
    axs[0].set_ylabel('True Class')
    axs[0].set_title('In Sample')

    cm = confusion_matrix(y_test, out_of_sample)
    axs[1].matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1].text(j, i, cm[i, j], ha='center', va='center')
    axs[1].set_xlabel('Predicted Class')
    axs[1].set_ylabel('True Class')
    axs[1].set_title('Out of Sample')

    plt.tight_layout()
    plt.show()
    
    return