# core_equity_factors

Econometrics project M2 203

## Installation

```bash
$ pip install core_equity_factors
```

## Usage

- TODO

## Plan

PART 1: Extracting the Core Equity Factor

    Principal Component Analysis (PCA):
        Function: perform_pca(returns)
        Purpose: To extract the first principal component from the monthly returns of European stocks.
        Details: This function will perform PCA on the returns data and return the first principal component, which represents the core equity factor. If the eigenvector of the first principal component contains mainly negative values, the function should return the negative of the first principal component to ensure a positive correlation with individual stock returns.

    Estimating Exposures:
        Function: estimate_exposures(returns, factor)
        Purpose: To estimate the exposures of each stock to the core equity factor using a linear model.
        Details: This function will fit a linear regression model for each stock's returns against the core equity factor to estimate the beta coefficients (exposures).

    Computing Portfolio Weights:
        Function: compute_portfolio_weights(returns, exposures)
        Purpose: To compute the weights of the equity portfolio designed to replicate the core equity factor.
        Details: This function will solve a constrained optimization problem to find the portfolio weights that minimize the portfolio variance, subject to the constraints that the weights sum to 1, are non-negative, and that the weighted sum of the exposures equals 1.

PART 2: Estimating Alpha and Assessing Estimation Errors

    Estimating Alpha:
        Function: estimate_alpha(portfolio_returns, benchmark_returns)
        Purpose: To estimate the alpha of the replicating portfolio against the market benchmark.
        Details: This function will fit a linear regression model of the portfolio returns against the benchmark returns to estimate the alpha.

    Assessing Estimation Errors:
        Function: compute_alpha_confidence_interval(portfolio_returns, benchmark_returns)
        Purpose: To compute the 95% confidence interval of the estimated alpha.
        Details: This function will use statistical methods to compute the confidence interval of the alpha, taking into account the estimation errors in the covariance matrix.

PART 3: Estimating the Global Trend and Investment Strategy

    Estimating the Global Trend:
        Function: estimate_trend(price_index)
        Purpose: To estimate the global trend and its slope from a local linear trend model.
        Details: This function will fit a local linear trend model to the price index of the estimated core portfolio to estimate the trend component and its slope.

    Retrieving Investment Strategy Track Record:
        Function: retrieve_investment_strategy(trend_slope, core_portfolio_returns, risk_free_rate)
        Purpose: To retrieve the track record of an investment strategy based on the trend slope.
        Details: This function will simulate the investment strategy, investing in the core portfolio when the trend slope is positive and in cash when the trend slope is non-positive.

    Testing Sharpe Ratio:
        Function: test_sharpe_ratio(strategy_returns)
        Purpose: To test if the Sharpe ratio of the investment strategy is due to luck.
        Details: This function will compute the Sharpe ratio of the investment strategy and perform a statistical test to determine if the Sharpe ratio is significantly different from zero.

Additional Helper Functions

    Data Loading:
        Function: load_data(file_path)
        Purpose: To load the dataset from the provided Excel file.
        Details: This function will read the monthly returns data from the Excel file and return it as a DataFrame.

    Plotting Results:
        Function: plot_results(data, title, xlabel, ylabel)
        Purpose: To plot the results of the analysis.
        Details: This function will create plots to visualize the results, such as the core equity factor, portfolio weights, trend components, and investment strategy performance.


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`core_equity_factors` was created by Faune Blanchard. It is licensed under the terms of the MIT license.

## Credits

`core_equity_factors` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
