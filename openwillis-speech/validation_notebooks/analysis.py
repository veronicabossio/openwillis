import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def distribution(df, var, var_name, plot_type, fontsize=10, dpi=200, x_limits=None, y_limits=None, filename=None):
    """
    This function, `distribution`, visualizes the distribution of a specified variable from a given DataFrame.
     It starts by removing any missing values from the selected variable, then calculates its mean, standard deviation, and kurtosis.
     Depending on the `plot_type` specified, it creates either a density plot or a histogram to showcase the distribution,
     annotating the plot with the variable's mean, standard deviation, and kurtosis, along with the sample size.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        var (str): variable name
        var_name (str): variable name for the plot title
        plot_type (str): type of plot to be generated
         can be 'density plot' or 'histogram'
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        x_limits (tuple, optional): x-axis limits for the plot (min, max)
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    # first, exclude any nans
    data = df[var].dropna()

    # get mean and std
    data_mean = data.mean()
    data_std = data.std()
    data_kurt = stats.kurtosis(data)

    # start plot
    fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)

    # create density plot
    if plot_type == "density plot":
        sns.kdeplot(data, color="#2E822D", fill=True, ax=ax)
    else:
        plt.hist(data, bins=10, color="#b8d4b4", edgecolor="black")

    # set x-axis limits if provided
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)

    # make the plot
    string = f"$\mu$ = {data_mean:.1f}; $\sigma$ = {data_std:.1f}\n$Kurtosis$ = {data_kurt:.1f}"
    ax.set_title(f"{var_name}\n{string}", pad=10, fontsize=fontsize)
    ax.set_ylabel(
        "Density" if plot_type == "density plot" else "Frequency", fontsize=fontsize
    )
    ax.set_xlabel(f"$n = {len(data)}$", fontsize=fontsize)

    ax.tick_params(axis="y", labelsize=fontsize)  # Set y-tick label size
    ax.tick_params(axis="x", labelsize=fontsize)  # Set x-tick label size

    plt.tight_layout()

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def ttest(df, var, values):
    """
    This function, `ttest`, performs a t-test to compare the means of a variable between two specified groups.
     It extracts the data for the two groups based on the values provided, then calculates the t-statistic and p-value
     using the `ttest_ind` function from the `scipy.stats` module. The function returns the t-statistic and p-value
     to assess the significance of the difference between the two groups' means.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        var (str): variable name
        values (list): values to compare

    Return:
    -------
        t_stat (float): t-statistic for the test
        p_val (float): p-value of the test
    """
    # extract data for the two groups
    group1 = df[df[var] == values[0]][var]
    group2 = df[df[var] == values[1]][var]

    # perform t-test
    t_stat, p_val = stats.ttest_ind(group1, group2)

    return t_stat, p_val


def calculate_labelpad_and_left(y_max):
    """Calculate labelpad and left margin based on the maximum y value."""
    y_max_log = np.log10(y_max)
    if y_max_log < -2:
        labelpad = 35
        left = 0.3
    elif y_max_log < 1:
        labelpad = 20
        left = 0.25
    elif y_max_log < 3:
        labelpad = 10
        left = 0.2
    else:
        labelpad = 10
        left = 0.15
    return labelpad, left


def boxplot_ttest(
    var_name, group1, group2, label1, label2, pval_thresh, fontsize=10, dpi=200, filename=None
):
    """The function performs a t-test between two groups and then visualizes the results using a boxplot.
     Initially, it calculates the t-statistic and p-value to compare the means of `group1` and `group2`.
     If the p-value is less than a predefined threshold (`pval_thresh`), it creates a boxplot with the data from both groups,
     highlighting the difference in their distributions.
     The boxplot includes notches and mean lines for a clearer visual representation of the data's central tendency and variance,
     and it's annotated with the p-value and sample sizes for each group.

    Args:
    -------
        var_name (str): name of the variable
        group1 (pandas:Series): data for group 1
        group2 (pandas:Series): data for group 2
        label1 (str): label for group 1
        label2 (str): label for group 2
        pval_thresh (float): p-value threshold for significance
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    # perform t-test
    _, pval = stats.ttest_ind(group1, group2)

    # calculate scale for y-axis
    scale = 10 ** (np.floor(np.log10(max(group1.max(), group2.max()))) - 1)
    exponent = int(np.log10(scale))

    if np.abs(exponent) >= 3:
        group1 = group1 / scale
        group2 = group2 / scale

    # combine data into single df
    data_to_plot = [group1, group2]

    # Calculate labelpad and left margin
    labelpad, left = calculate_labelpad_and_left(max(group1.max(), group2.max()))

    if pval < pval_thresh:

        # start plot
        fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)
        plt.gca().set_facecolor("#f5f5f5" if pval > 0.05 else "white")
        bp = ax.boxplot(
            data_to_plot, patch_artist=True, notch=True, showmeans=True, meanline=True
        )
        colors = ["#5D9B5C", "#B0D2B0"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(f"{var_name}\np-val: {pval:.2f}", pad=10, fontsize=fontsize)
        if np.abs(exponent) < 3:
            ax.set_ylabel(f"{var_name}", fontsize=fontsize, labelpad=labelpad)
        else:
            ax.set_ylabel(
                f"{var_name} (x$10^{{{exponent}}}$)",
                fontsize=fontsize,
                labelpad=labelpad,
            )

        ax.set_xticklabels(
            [f"{label1}\n$n={len(group1)}$", f"{label2}\n$n={len(group2)}$"],
            fontsize=fontsize,
        )
        ax.tick_params(axis="y", labelsize=fontsize)  # Set y-tick label size
        ax.tick_params(axis="x", labelsize=fontsize)  # Set x-tick label size

        # the yticklabel should be X.XX, with with the decimal always being there
        if np.abs(exponent) >= 3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

        # Adjust subplot parameters to ensure consistent plot area width
        plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.1)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()


def scatterplot_rmse(
    dep_var, ind_var, dep_var_label, ind_var_label, rmse, rsq, fontsize=10, dpi=200, filename=None
):
    """The function creates a scatter plot to compare an independent variable (`ind_var`) with a dependent variable (`dep_var`),
     which is useful for evaluating the accuracy of a model.
     Points representing the relationship between these variables are plotted, along with a y=x line to serve
     as a reference for perfect prediction. The plot is annotated with the root mean square error (RMSE)
     and the coefficient of determination (R²) to quantify the model's accuracy, alongside the count of data points.
     The plot's title and axes are labeled with the respective variable names and statistical metrics.

    Args:
    -------
        dep_var (pandas:Series): dependent variable data
        ind_var (pandas:Series): independent variable data
        dep_var_label (str): label for the dependent variable
        ind_var_label (str): label for the independent variable
        rmse (float): root mean square error
        rsq (float): coefficient of determination
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """

    # create figure
    fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)

    # plot points with y = x line
    ax.scatter(ind_var, dep_var, alpha=0.5, color="#b8d4b4")
    ax.plot(ind_var, ind_var, color="darkgreen")

    # Calculate labelpad and left margin
    y_max = max(dep_var.max(), ind_var.max())
    labelpad, left = calculate_labelpad_and_left(y_max)

    # add title and labels
    ax.set_title(
        f"Model Accuracy\nRMSE = {rmse:.3f}; $R^2$ = {rsq:.3f}\n$n = {len(ind_var)}$",
        pad=10,
        fontsize=fontsize,
    )
    ax.set_xlabel(ind_var_label, fontsize=fontsize)
    ax.set_ylabel(dep_var_label, fontsize=fontsize, labelpad=labelpad)

    ax.tick_params(axis="y", labelsize=fontsize)  # Set y-tick label size
    ax.tick_params(axis="x", labelsize=fontsize)  # Set x-tick label size

    # Adjust subplot parameters to ensure consistent plot area width
    plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.1)

    plt.tight_layout()
    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def corr_test(df_plot, dep_var, ind_var):
    """
    This function calculates the Pearson correlation coefficient between two variables in a given DataFrame.
     It extracts the data for the dependent and independent variables, then computes the correlation coefficient
     and its p-value using the `pearsonr` function from the `scipy.stats` module.
     The function returns the correlation coefficient, p-value, and coefficient of determination (R²) for the relationship.

    Args:
    -------
        df_plot (pandas:DataFrame): DataFrame containing the data
        dep_var (str): dependent variable name
        ind_var (str): independent variable name

    Return:
    -------
        pval (float): p-value of the statistical test
        rsq (float): coefficient of determination (R²) for the relationship
    """
    dep_var_data = df_plot[dep_var]
    ind_var_data = df_plot[ind_var]

    # conduct pearson correlation
    r_value, pval = stats.pearsonr(dep_var_data, ind_var_data)
    rsq = r_value**2

    return pval, rsq


def mlr_test(df_plot, dep_var, ind_vars, print_stats=False, fit_reduced=False):
    """
    This function performs a multiple linear regression to estimate the relationship between two variables in a given DataFrame.
    It extracts the data for the dependent and independent variables, then runs a multiple linear regression model
    using the `OLS` function from the `statsmodels` module.
    The function returns the p-value, coefficient, confidence interval, and partial R² for the independent variable in the regression model.

    Args:
    -------
        df_plot (pandas:DataFrame): DataFrame containing the data
        dep_var (str): dependent variable name
        ind_var (list): independent variable names
        the first element in the list is the variable of interest
        print_stats (bool): whether to print the regression model summary

    Return:
    -------
        pval (float): p-value of the statistical test
        coef (float): coefficient for the independent variable in the regression model
        partial_r2 (float): partial R² for the first independent variable
        conf_int (tuple): confidence interval for the coefficient of the first independent variable
    """
    X = df_plot[ind_vars]
    X = sm.add_constant(X)
    y = df_plot[dep_var]
    model = sm.OLS(y, X).fit()

    # get results
    pval = model.pvalues[ind_vars[0]]
    coef = model.params[ind_vars[0]]
    conf_int = model.conf_int().loc[ind_vars[0]].tolist()

    if print_stats:
        print(model.summary())

    if not fit_reduced:
        return pval, coef, None, conf_int

    # Calculate partial R²
    full_r2 = model.rsquared

    # Fit reduced model
    X_reduced = df_plot[ind_vars[1:]]
    X_reduced = sm.add_constant(X_reduced)
    model_reduced = sm.OLS(y, X_reduced).fit()

    reduced_r2 = model_reduced.rsquared

    partial_r2 = (full_r2 - reduced_r2) / (1 - reduced_r2)

    return pval, coef, partial_r2, conf_int


def lme_test(df_plot, dep_var, ind_vars, random_effects, print_stats=False, fit_reduced=False):
    """
    This function performs a Linear Mixed Effects (LME) model to estimate the relationship between the dependent variable 
    and independent variables in a given DataFrame. It includes a random effect to account for repeated measures.

    Args:
    -------
        df_plot (pandas:DataFrame): DataFrame containing the data
        dep_var (str): dependent variable name
        ind_vars (list): independent variable names (the first element in the list is the variable of interest)
        random_effect (str): name of the variable to use as the random effect
        print_stats (bool): whether to print the regression model summary

    Return:
    -------
        pval (float): p-value of the statistical test
        coef (float): coefficient for the independent variable in the regression model
        partial_r2 (float): partial R² for the first independent variable
        conf_int (tuple): confidence interval for the coefficient of the first independent variable
    """
    # Construct the formula for the full model
    fixed_effects = ' + '.join(ind_vars + [random_effects[1]])
    formula_full = f"{dep_var} ~ {fixed_effects}"
    
    # Fit the full model
    if len(random_effects) == 2:
        model_full = smf.mixedlm(formula_full, df_plot, groups=df_plot[random_effects[0]],
                                 re_formula=f"~{random_effects[1]}").fit(method=["lbfgs"])
    else:
        model_full = smf.mixedlm(formula_full, df_plot, groups=df_plot[random_effects[0]]).fit(method=["lbfgs"])

    # Get results for the variable of interest
    pval = model_full.pvalues[ind_vars[0]]
    coef = model_full.params[ind_vars[0]]
    conf_int = model_full.conf_int().loc[ind_vars[0]].tolist()

    if print_stats:
        print(model_full.summary())
    
    if not fit_reduced:
        return pval, coef, None, conf_int

    # Construct the formula for the reduced model (excluding the first independent variable)
    fixed_effects_reduced = ' + '.join(ind_vars[1:])
    formula_reduced = f"{dep_var} ~ {fixed_effects_reduced}"
    
    # Fit the reduced model
    if len(random_effects) == 2:
        model_reduced = smf.mixedlm(formula_reduced, df_plot, groups=df_plot[random_effects[0]],
                                    re_formula=f"~{random_effects[1]}").fit(method=["lbfgs"])
    else:
        model_reduced = smf.mixedlm(formula_reduced, df_plot, groups=df_plot[random_effects[0]]).fit(method=["lbfgs"])
    

    # Calculate residual sum of squares (RSS)
    full_marginal_r2 = model_full.fittedvalues.var() / (model_full.fittedvalues.var() + model_full.scale + model_full.cov_re.iloc[0,0] + model_full.cov_re.iloc[1,1])
    reduced_marginal_r2 = model_reduced.fittedvalues.var() / (model_reduced.fittedvalues.var() + model_reduced.scale + model_reduced.cov_re.iloc[0,0] + model_reduced.cov_re.iloc[1,1])
    
    # Partial R2 calculation
    partial_r2 = (full_marginal_r2 - reduced_marginal_r2) / (1 - reduced_marginal_r2)

    return pval, coef, partial_r2, conf_int


def scatterplot(
    df,
    dep_var,
    ind_vars,
    dep_var_label,
    ind_var_label,
    title,
    pval_thresh,
    option,
    print_stats=False,
    poster_format=False,
    fontsize=10,
    dpi=200,
    filename=None,
):
    """
    This function creates a scatter plot to visualize the relationship between two variables in a given DataFrame.
     It starts by removing any missing values from the dependent and independent variables, then performs a statistical test
     to determine the significance of the relationship. If the p-value is below a predefined threshold (`pval_thresh`),
     the function generates a scatter plot with a regression line to represent the data points' distribution.
     If the `option` is set to 'corr', it calculates the Pearson correlation coefficient and coefficient of determination (R²),
     while for 'mlr', it runs a multiple linear regression model to estimate the relationship.
     If the `option` is set to 'lme', it runs a Linear Mixed Effects (LME) model to estimate the relationship.
     The plot is annotated with the statistical metrics, variable names, and sample size for reference.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        dep_var (str): dependent variable name
        ind_vars (list): independent variable names
            the first element in the list is the variable of interest
        dep_var_label (str): label for the dependent variable
        ind_var_label (str): label for the independent variable
        title (str): title for the plot
        pval_thresh (float): p-value threshold for significance
        option (str): type of statistical test to perform
         can be 'corr' for Pearson correlation or 'mlr' for multiple linear regression
         or 'lme' for Linear Mixed Effects model
        print_stats (bool): whether to print the regression model summary
        poster_format (bool): whether to use a poster-friendly format for the plot
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    # get only columns we need
    if option == "corr":
        df_plot = df.dropna(subset=[dep_var, ind_vars[0]]).copy()

        # conduct pearson correlation
        pval, rsq = corr_test(df_plot, dep_var, ind_vars[0])
    elif option == "mlr":
        df_plot = df.dropna(subset=[dep_var] + ind_vars).copy()

        # run model
        pval, coef, _, _ = mlr_test(df_plot, dep_var, ind_vars, print_stats)
    elif option == "lme":
        df_plot = df.dropna(subset=["id", "visit", dep_var] + ind_vars).copy()

        # run model
        pval, coef, _, _ = lme_test(df_plot, dep_var, ind_vars, ["id", "visit"], print_stats, fit_reduced=False)

    if pval < pval_thresh:

        if poster_format:
            fig, ax = plt.subplots(figsize=(3.15, 3.5), dpi=dpi)
            ax.set_facecolor("#c1e0b0")

            sns.regplot(
                x=df_plot[dep_var],
                y=df_plot[ind_vars[0]],
                scatter_kws={"color": "#586a53ff", "alpha": 0.5},
                line_kws={"color": "#323f2e"},
                ax=ax,
            )
        else:
            fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)
            ax.set_facecolor("#f5f5f5" if pval > 0.05 else "white")

            sns.regplot(
                x=df_plot[dep_var],
                y=df_plot[ind_vars[0]],
                scatter_kws={"color": "#b8d4b4", "alpha": 0.5},
                line_kws={"color": "darkgreen"},
                ax=ax,
            )

        # Calculate labelpad and left margin
        y_max = max(df_plot[dep_var].max(), df_plot[ind_vars[0]].max())
        labelpad, left = calculate_labelpad_and_left(y_max)

        if option == "corr":
            ax.set_title(
                f"{title}\n $R^2$ = {rsq:.3f}; p-value = {pval:.3f}\n$n = {len(df_plot)}$",
                pad=10,
                fontsize=fontsize,
            )
        elif option == "mlr" and poster_format:
            ax.set_title(
                f"{ind_var_label}\ncoef = {coef:.3f}; p-value = {pval:.3f}\n$n = {len(df_plot)}$",
                pad=10,
                fontsize=fontsize,
                color="#323f2e",
            )
        else:
            ax.set_title(
                f"{ind_var_label}\ncoef = {coef:.3f}; p-value = {pval:.3f}\n$n = {len(df_plot)}$",
                pad=10,
                fontsize=fontsize,
            )

        if poster_format:
            ax.set_ylabel(
                ind_var_label, fontsize=fontsize, color="#323f2e", labelpad=labelpad
            )
            ax.set_xlabel(dep_var_label, fontsize=fontsize, color="#323f2e")

            # Change tick colors
            ax.tick_params(axis="x", colors="#323f2e", labelsize=fontsize)
            ax.tick_params(axis="y", colors="#323f2e", labelsize=fontsize)

            # Update edge color if necessary (currently set to match the facecolor)
            ax.spines["top"].set_color("#323f2e")
            ax.spines["right"].set_color("#323f2e")
            ax.spines["left"].set_color("#323f2e")
            ax.spines["bottom"].set_color("#323f2e")
        else:
            ax.set_ylabel(ind_var_label, fontsize=fontsize, labelpad=labelpad)
            ax.set_xlabel(dep_var_label, fontsize=fontsize)
            ax.tick_params(axis="y", labelsize=fontsize)
            ax.tick_params(axis="x", labelsize=fontsize)

        # Adjust subplot parameters to ensure consistent plot area width
        plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.1)

        plt.tight_layout()
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()


def predict_response(df, clinical_score, dhm, arm, pvalue=1, dpi=200, filename=None):
    """
    The function prepares and analyzes data to study how a digital health measure (DHM)
     predicts the change in a clinical score for patients within a specific treatment arm of a clinical trial.
     It focuses on certain visits, computes the difference in clinical scores between two visits, and calculates
     the mean DHM value across initial visits. The data is then organized to relate these DHM values with the
     clinical score changes. Finally, it employs the `scatterplot` function to visually represent and analyze
     the relationship between the DHM and the clinical score reduction, specifically for the selected treatment arm,
     using a scatter plot and multiple linear regression.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        clinical_score (str): name of the clinical score
        dhm (str): name of the digital health measure
        arm (int): treatment arm to analyze
        pvalue (float): minimum p-value to display scatter plot
        dpi (int): dpi for scatter plot

    Return:
    -------
        None
    """
    # limiting to visits we need
    df_visits = df[df["visit"].isin([1, 2, 9])]

    # calculating reduction in clinical score
    cscore_diff = (
        df_visits[df_visits["visit"] == 2].set_index("id")[clinical_score]
        - df_visits[df_visits["visit"] == 9].set_index("id")[clinical_score]
    )

    # getting mean value of measure from screening and baseline
    dhm_mean = df_visits[df_visits["visit"].isin([1, 2])].groupby("id")[dhm].mean()

    # creating the df that will go into the function
    df_predict_response = (
        df[["id", "age", "male", "white", "black", "arm"]]
        .drop_duplicates()
        .set_index("id")
    )

    df_predict_response = df_predict_response.join(
        cscore_diff.rename("reduction_in_clinical_score")
    )
    df_predict_response = df_predict_response.join(dhm_mean.rename(dhm))
    df_predict_response.reset_index(inplace=True)

    # running the scatterplot + mlr function
    scatterplot(
        df_predict_response[df_predict_response["arm"] == arm],
        "reduction_in_clinical_score",
        [dhm, "age", "male", "white", "black"],
        f"Reduction in {clinical_score.upper()}",
        dhm.replace("_", " ").title(),
        None,
        pvalue,
        "mlr",
        dpi=dpi,
        filename=filename,
    )


def train_rf_model(df_train, features, clinical_score):
    """
    The function trains a Random Forest model to predict a clinical score from specified features.
        It preprocesses the training data by imputing missing values and standardizing features,
        then fits a Random Forest Regressor to the data. The function returns the trained model,
        imputer, and scaler for further analysis and testing.

    Args:
    -------
        df_train (pandas:DataFrame): training data
        features (list): list of feature names to use in the model
        clinical_score (str): name of the clinical score to predict

    Return:
    -------
        rf_model (RandomForestRegressor): trained Random Forest model
        imputer (SimpleImputer): imputer object used for preprocessing
        scaler (StandardScaler): scaler object used for preprocessing
    """
    # only keeping the necessary columns
    training_set = df_train[features + [clinical_score]]

    # filling in nans through imputation
    imputer = SimpleImputer(strategy="mean")
    training_set_imputed = pd.DataFrame(
        imputer.fit_transform(training_set),
        columns=training_set.columns,
        index=df_train.index,
    )

    # defining the feature matrix and the target vector
    X_train = training_set_imputed.drop(clinical_score, axis=1)
    y_train = training_set_imputed[clinical_score].reset_index(drop=True)

    # scaling the feature matrix
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # train random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    return rf_model, imputer, scaler


def test_rf_model(rf_model, imputer, scaler, df_test, features, clinical_score):
    """
    The function tests a Random Forest model to predict a clinical score from specified features.
        It preprocesses the testing data by imputing missing values and standardizing features,
        then evaluates the model's performance on the test data. The function returns the actual
        and predicted values for the clinical score to assess the model's accuracy.

    Args:
    -------
        rf_model (RandomForestRegressor): trained Random Forest model
        imputer (SimpleImputer): imputer object used for preprocessing
        scaler (StandardScaler): scaler object used for preprocessing
        df_test (pandas:DataFrame): testing data
        features (list): list of feature names to use in the model
        clinical_score (str): name of the clinical score to predict

    Return:
    -------
        y_test (pandas:DataFrame): actual values for the clinical score
        y_pred (pandas:DataFrame): predicted values for the clinical score
    """
    # only keeping necessary columns
    testing_set = df_test[features + [clinical_score]]

    # filling in nans through imputation
    testing_set_imputed = pd.DataFrame(
        imputer.transform(testing_set), columns=testing_set.columns, index=df_test.index
    )

    # defining the feature matrix and the target vector
    X_test = testing_set_imputed.drop(clinical_score, axis=1)
    y_test = testing_set_imputed[clinical_score]

    # scaling the feature matrix
    X_test_scaled = scaler.transform(X_test)

    # testing on testing data
    y_pred = rf_model.predict(X_test_scaled)

    return y_test, y_pred


def train_and_test(clinical_score, features, df_train, df_test, dpi=200, filename=None):
    """
    The function trains a Random Forest model to predict a clinical score
     from specified features, evaluates its performance on test data, and
     visualizes the results. It preprocesses the data by imputing missing values and
     standardizing features, then calculates the model's accuracy using RMSE and R².
     Finally, it outputs the residuals and predicted values for further analysis.

    Args:
    -------
        clinical_score (str): name of the clinical score to predict
        features (list): list of feature names to use in the model
        df_train (pandas:DataFrame): training data
        df_test (pandas:DataFrame): testing data

    Return:
    -------
        residuals (pandas:DataFrame): residuals from the model
        y_pred (pandas:DataFrame): predicted values for the clinical score
    """
    # training
    rf_model, imputer, scaler = train_rf_model(df_train, features, clinical_score)

    # testing
    y_test, y_pred = test_rf_model(
        rf_model, imputer, scaler, df_test, features, clinical_score
    )

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rsq = r2_score(y_test, y_pred)

    # calculate residuals
    residuals = pd.DataFrame({"residuals": (y_test - y_pred)}, index=y_test.index)

    # save y_pred as df
    y_pred = pd.DataFrame({"y_pred": y_pred}, index=y_test.index)

    # plot a figure with the results
    if clinical_score == "panss_change":
        scatterplot_rmse(
            y_test,
            y_pred["y_pred"],
            "Actual change in PANSS",
            "Predicted change in PANSS",
            rmse,
            rsq,
            dpi=dpi,
            filename=filename,
        )
    elif clinical_score == "panss_ptot":
        scatterplot_rmse(
            y_test,
            y_pred["y_pred"],
            "Actual PANSS-P",
            "Predicted PANSS-P",
            rmse,
            rsq,
            dpi=dpi,
            filename=filename,
        )
    elif clinical_score == "panss_ntot":
        scatterplot_rmse(
            y_test,
            y_pred["y_pred"],
            "Actual PANSS-N",
            "Predicted PANSS-N",
            rmse,
            rsq,
            dpi=dpi,
            filename=filename,
        )
    else:
        scatterplot_rmse(
            y_test,
            y_pred["y_pred"],
            f"Actual {clinical_score.upper()}",
            f"Predicted {clinical_score.upper()}",
            rmse,
            rsq,
            dpi=dpi,
            filename=filename,
        )

    return residuals, y_pred, rf_model


def plot_feature_importance(rf_model, features, top=10, fontsize=10, dpi=200, filename=None):
    """
    The function plots the feature importances of the Random Forest model.

    Args:
    -------
        rf_model (RandomForestRegressor): trained Random Forest model
        features (list): list of feature names used in the model
        top (int): number of top features to display
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """

    # calculate feature importances
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_, index=features, columns=["importance"]
    ).sort_values("importance", ascending=False)[:top]

    fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)
    ax.barh(
        [x.replace('_', ' ').title() for x in feature_importances.index.tolist()], feature_importances["importance"], color="#b8d4b4"
    )
    ax.set_xlabel("Feature Importance", fontsize=fontsize)
    ax.set_ylabel("Features", fontsize=fontsize)
    ax.set_title("Feature Importance from Random Forest", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)

    plt.gca().invert_yaxis()
    # plt.tight_layout()
    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def residuals_vs_response(df, clinical_score, residuals):
    """
    The function examines the relationship between model prediction errors (residuals) and changes in a clinical score,
     focusing on specific visits in a clinical trial. It calculates the change in clinical score between visits, averages
     the residuals for patients, and then generates scatter plots to visualize these relationships for all patients
     and separately for different treatment arms. This analysis helps identify any systematic patterns or biases in the
     model's predictions across different patient groups.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        clinical_score (str): name of the clinical score
        residuals (pandas:DataFrame): residuals from the model

    Return:
    -------
        None
    """

    # adding residuals column to original df
    df = df.join(residuals["residuals"])

    # filtering df to the visits we care about
    df_filtered = df[df["visit"].isin([2, 9])]

    # calculating change in clinical score
    cscore_diff = (
        df_filtered[df_filtered["visit"] == 2].set_index("id")[clinical_score]
        - df_filtered[df_filtered["visit"] == 9].set_index("id")[clinical_score]
    )

    # getting mean residual from screening and baseline
    residuals_mean = (
        df_filtered[df_filtered["visit"] == 2].groupby("id")["residuals"].mean()
    )

    # removing duplicate entries, resetting index
    df_plot = df[["id", "age", "arm"]].drop_duplicates().set_index("id")

    # adding change in clinical_score and residuals to the df
    df_plot = df_plot.join(cscore_diff.rename("reduction_in_clinical_score"))
    df_plot = df_plot.join(residuals_mean.rename("residuals"))

    # get only columns we need
    df_plot = df_plot.dropna(subset=["reduction_in_clinical_score", "residuals"]).copy()

    # creating plots
    scatterplot(
        df_plot,
        "reduction_in_clinical_score",
        "residuals",
        f"Reduction in {clinical_score.upper()}",
        "Residuals / Inflation",
        "Residuals vs. Response",
        1,
        "corr",
    )
    scatterplot(
        df_plot[df_plot["arm"] == 1],
        "reduction_in_clinical_score",
        "residuals",
        f"Reduction in {clinical_score.upper()}",
        "Residuals / Inflation",
        "Residuals vs. KarXT Response",
        1,
        "corr",
    )
    scatterplot(
        df_plot[df_plot["arm"] == 0],
        "reduction_in_clinical_score",
        "residuals",
        f"Reduction in {clinical_score.upper()}",
        "Residuals / Inflation",
        "Residuals vs. Placebo Response",
        1,
        "corr",
    )


def cohens_d(group1, group2):
    """
    The function `cohens_d` calculates Cohen's d, a measure of effect size used to indicate the
     standardized difference between two groups. It computes the pooled standard deviation of the
     two groups, which is a weighted average of their standard deviations. Then, it calculates
     the difference in means between the two groups and divides this by the pooled standard deviation to
     obtain Cohen's d. This value provides a sense of how distinct the two groups are, with larger values
     indicating a more substantial difference in means relative to their variability.

    Args:
    -------
        group1 (pandas:Series): data for group 1
        group2 (pandas:Series): data for group 2

    Return:
    -------
        effect_size (float): Cohen's d effect size
    """
    # calculate the pooled standard deviation of two groups
    pooled_std = np.sqrt(
        (
            (len(group1) - 1) * np.std(group1, ddof=1) ** 2
            + (len(group2) - 1) * np.std(group2, ddof=1) ** 2
        )
        / (len(group1) + len(group2) - 2)
    )

    # calculate cohen's d
    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std

    return effect_size


def response_to_treatment(
    df, endpoint, endpoint_name, min_y=None, max_y=None, fontsize=10, dpi=200, filename=None
):
    """
    The function `response_to_treatment` assesses and visualizes the response to a
     treatment by comparing the effect size between a treatment group and a placebo group at
     specific visits in a clinical trial. It calculates Cohen's d to quantify the effect size
     between the two groups at the final visit. The function then filters the dataset for relevant
     visits and arms, computes the mean and standard error for the specified endpoint, and plots
     these statistics over time for both groups. The plot includes error bars representing the
     standard error, facilitating a visual comparison of the treatment's effectiveness over time
     against the placebo, with the effect size providing a standardized measure of the treatment's impact.


    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        endpoint (str): name of the endpoint to analyze
        endpoint_name (str): name of the endpoint for plotting
        min_y (float, optional): minimum y-value for the plot. Default is None.
        max_y (float, optional): maximum y-value for the plot. Default is None.
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """

    # calculate effect size
    plcb_group = df[(df["arm"] == 0) & (df["visit"] == 9)][endpoint]
    trmt_group = df[(df["arm"] == 1) & (df["visit"] == 9)][endpoint]
    effectsize = cohens_d(trmt_group, plcb_group)

    df_filtered = df[df["visit"].isin([2, 6, 8, 9])]
    df_filtered = df_filtered[df_filtered["arm"].isin([0, 1])]
    df_filtered = df_filtered.dropna(subset=[endpoint])
    summary_df = (
        df_filtered.groupby(["arm", "visit"])
        .agg({endpoint: ["mean", stats.sem]})
        .reset_index()
    )
    summary_df.columns = ["arm", "visit", "endpoint_mean", "endpoint_sem"]

    # create plot
    fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)

    plcb_data = summary_df[summary_df["arm"] == 0]
    ax.errorbar(
        x=plcb_data["visit"],
        y=plcb_data["endpoint_mean"],
        yerr=plcb_data["endpoint_sem"],
        label="Placebo",
        color="grey",
        fmt="o-",
        capsize=5,
    )

    trtm_data = summary_df[summary_df["arm"] == 1]
    ax.errorbar(
        x=trtm_data["visit"],
        y=trtm_data["endpoint_mean"],
        yerr=trtm_data["endpoint_sem"],
        label="KarXT",
        color="#5D9B5C",
        fmt="o-",
        capsize=5,
    )

    xtick_labels = {2: "Baseline", 6: "W2", 8: "W4", 9: "W5"}
    ax.set_xticks([2, 6, 8, 9])
    ax.set_xticklabels([xtick_labels[x] for x in [2, 6, 8, 9]], fontsize=fontsize)

    y_max = max(summary_df["endpoint_mean"] + summary_df["endpoint_sem"])
    labelpad, left = calculate_labelpad_and_left(y_max)

    ax.set_title(
        f"Treatment response\n$n, Placebo= {df[df['arm'] == 0]['id'].nunique()}$; $n, KarXT = {df[df['arm'] == 1]['id'].nunique()}$\nEffect size = {effectsize:.2f}",
        pad=10,
        fontsize=fontsize,
    )
    ax.set_ylabel(endpoint_name, fontsize=fontsize, labelpad=labelpad)

    if min_y is not None and max_y is not None:
        plt.ylim([min_y, max_y])
    elif endpoint == "panss":
        plt.ylim([75, 105])

    ax.legend(title="Arm", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)

    # Adjust subplot parameters to ensure consistent plot area width
    plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.1)

    plt.tight_layout()
    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def enrich_trial_using_residuals(df, endpoint, cutoff, filename=None):
    """
    The function segments a clinical trial population into two groups based on
     the residuals from a predictive model: one with residuals below a specified
     cutoff (enriched) and another with residuals above (unenriched). It then
     applies the `response_to_treatment` function to compare the treatment response
     across the original, enriched, and unenriched datasets. This approach helps in
     evaluating whether patients predicted more accurately by the model show different
     treatment responses, potentially guiding more personalized and effective clinical trial designs.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        endpoint (str): name of the endpoint to analyze
        cutoff (float): cutoff value for residuals to enrich the dataset
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """

    # find ids with residuals below cutoff
    ids_with_residuals_below_cutoff = df[df["residuals"] < cutoff]["id"].unique()

    # find ids with residuals below cutoff
    ids_with_residuals_above_cutoff = df[df["residuals"] >= cutoff]["id"].unique()

    # filter the df to only include those ids
    df_enriched = df[df["id"].isin(ids_with_residuals_below_cutoff)]

    df_unenriched = df[df["id"].isin(ids_with_residuals_above_cutoff)]

    response_to_treatment(df, endpoint, endpoint.upper(), filename=filename.split(".")[0] + "_original.png" if filename else None)
    response_to_treatment(df_enriched, endpoint, endpoint.upper(), filename=filename.split(".")[0] + "_enriched.png" if filename else None)
    response_to_treatment(df_unenriched, endpoint, endpoint.upper(), filename=filename.split(".")[0] + "_unenriched.png" if filename else None)


def fit_lme_model(df, dep_var, ind_vars, random_effects):
    """
    Fit a linear mixed effects model.
    
    Args:
    -------
        df (pandas: DataFrame): DataFrame containing the data.
        dep_var (str): Dependent variable name.
        ind_vars (list): Independent variable names.
        random_effect (str): Name of the variable to use as the random effect.

    Return:
    -------
        model (MixedLMResults): Fitted mixed effects model.
        rss (float): Residual sum of squares of the model.
        pvalues (pd.Series): P-values for the coefficients.
    """
    formula = f"{dep_var} ~ {' + '.join(ind_vars)}"
    model = smf.mixedlm(formula, df, groups=df[random_effects[0]],
                        re_formula=f"~{random_effects[1]}").fit(method=["lbfgs"])
    
    marginal_r2 = model.fittedvalues.var() / (model.fittedvalues.var() + model.scale + model.cov_re.iloc[0,0] + model.cov_re.iloc[1,1])
    
    return model, marginal_r2, model.pvalues

def forward_feature_selection(df, dep_var, initial_vars, candidate_vars, random_effects, significance_level=0.10):
    """
    Perform forward feature selection to build an LME model.

    Args:
    -------
        df (pandas: DataFrame): DataFrame containing the data.
        dep_var (str): Dependent variable name.
        initial_vars (list): List of initial variables (e.g., demographics).
        candidate_vars (list): List of candidate variables (e.g., DHMs) to be added.
        random_effects (str): Name of the variable to use as the random effect.
        significance_level (float): Significance level for adding new variables.

    Return:
    -------
        best_model (MixedLMResults): Best fitted mixed effects model.
        selected_vars (list): List of selected variables.
        var_contributions (list): List of tuples containing variable names and their contributions to the model.
    """
    selected_vars = initial_vars.copy()
    remaining_vars = candidate_vars.copy()
    best_model, best_r2, _ = fit_lme_model(df, dep_var, selected_vars, random_effects)
    baseline_r2 = best_r2
    var_contributions = []

    while remaining_vars:
        best_pvalue = float('inf')
        best_var = None
        best_r2 = baseline_r2
        for var in remaining_vars:
            current_vars = selected_vars + [var]
            try:
                model, r2, pvalues = fit_lme_model(df, dep_var, current_vars, random_effects)
            except Exception as e:
                print(f"Error fitting model with variable {var}: {e}")
                continue
            
            # Check the p-value of the added variable
            pvalue = pvalues.get(var, float('inf'))
            
            if pvalue < best_pvalue and pvalue < significance_level:
                best_model = model
                best_r2 = r2
                best_var = var
                best_pvalue = pvalue
        
        if best_var is not None:
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
            variance_contribution = (best_r2 - baseline_r2) / (1 - baseline_r2)
            var_contributions.append((best_var, variance_contribution))
            baseline_r2 = best_r2
        else:
            break
    
    return best_model, selected_vars, var_contributions

def fit_mlr_model(df, dep_var, ind_vars):
    """
    Fit a multiple linear regression model.
    
    Args:
    -------
        df (pandas: DataFrame): DataFrame containing the data.
        dep_var (str): Dependent variable name.
        ind_vars (list): Independent variable names.

    Return:
    -------
        model (RegressionResultsWrapper): Fitted regression model.
        r2 (float): R-squared of the model.
        pvalues (pd.Series): P-values for the coefficients.
    """
    X = df[ind_vars]
    X = sm.add_constant(X)
    y = df[dep_var]
    model = sm.OLS(y, X).fit()
    
    r2 = model.rsquared
    
    return model, r2, model.pvalues

def forward_feature_selection_mlr(df, dep_var, initial_vars, candidate_vars, significance_level=0.10):
    """
    Perform forward feature selection to build a MLR model.

    Args:
    -------
        df (pandas: DataFrame): DataFrame containing the data.
        dep_var (str): Dependent variable name.
        initial_vars (list): List of initial variables (e.g., demographics).
        candidate_vars (list): List of candidate variables (e.g., DHMs) to be added.
        significance_level (float): Significance level for adding new variables.

    Return:
    -------
        best_model (RegressionResultsWrapper): Best fitted regression model.
        selected_vars (list): List of selected variables.
        var_contributions (list): List of tuples containing variable names and their contributions to the model.
    """
    selected_vars = initial_vars.copy()
    remaining_vars = candidate_vars.copy()
    best_model, best_r2, _ = fit_mlr_model(df, dep_var, selected_vars)
    baseline_r2 = best_r2
    var_contributions = []

    while remaining_vars:
        best_pvalue = float('inf')
        best_var = None
        best_r2 = baseline_r2
        for var in remaining_vars:
            current_vars = selected_vars + [var]
            try:
                model, r2, pvalues = fit_mlr_model(df, dep_var, current_vars)
            except Exception as e:
                print(f"Error fitting model with variable {var}: {e}")
                continue
            
            # Check the p-value of the added variable
            pvalue = pvalues.get(var, float('inf'))
            
            if pvalue < best_pvalue and pvalue < significance_level:
                best_model = model
                best_r2 = r2
                best_var = var
                best_pvalue = pvalue
        
        if best_var is not None:
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
            variance_contribution = (best_r2 - baseline_r2) / (1 - baseline_r2)
            var_contributions.append((best_var, variance_contribution))
            baseline_r2 = best_r2
        else:
            break
    
    return best_model, selected_vars, var_contributions


def boxplot_sites_multiple(var_names, all_groups, labels, title="Boxplots of Variables Across Sites", fontsize=10, dpi=200, filename=None, site_spacing=1.0):
    """
    Creates box plots to visualize the distribution of multiple variables across different sites.

    Args:
        var_names (list): List of names of the variables to plot.
        all_groups (list of lists): List of lists of groups, where each inner list represents the groups for a specific variable.
        labels (list): List of labels to identify each site.
        title (str): Title for the plot (default: "Boxplots of Variables Across Sites").
        fontsize (int): Font size for labels and ticks (default: 10).
        dpi (int): Dots per inch for the plot resolution (default: 200).
        filename (str, optional): Filename to save the plot.
        site_spacing (float): Spacing factor between sites (default: 1.0). Increase to add more space.

    Returns:
        None
    """

    num_vars = len(var_names)
    num_sites = len(labels)
    
    # Calculate scale for y-axis for each variable
    scales = []
    exponents = []
    for groups in all_groups:
        max_val = max([group.max() for group in groups])
        scale = 10 ** (np.floor(np.log10(max_val)) - 1)
        exponent = int(np.log10(scale))
        scales.append(scale)
        exponents.append(exponent)

    # Scale groups if necessary
    scaled_groups = []
    for i, groups in enumerate(all_groups):
        if np.abs(exponents[i]) >= 3:
            scaled_groups.append([group / scales[i] for group in groups])
        else:
            scaled_groups.append(groups)

    # Combine data into single list
    data_to_plot = []
    for i in range(num_sites):
        site_data = []
        for j in range(num_vars):
            site_data.extend(scaled_groups[j][i])
        data_to_plot.append(site_data)

    # Calculate labelpad and left margin
    labelpad, left = calculate_labelpad_and_left(max([group.max() for groups in scaled_groups for group in groups]))

    # Calculate positions with increased spacing
    positions = []
    for i in range(num_vars):
        positions.extend([x + i * site_spacing for x in range(1, num_sites + 1)])

    # Start plot with adjusted figure size
    fig, ax = plt.subplots(figsize=(30, 10), dpi=dpi)
    plt.gca().set_facecolor("white")

    # Create a list to store legend handles
    legend_handles = []

    # Plot each variable's data
    for i, groups in enumerate(scaled_groups):
        bp = ax.boxplot(
            groups, positions=positions[i*num_sites:(i+1)*num_sites], widths=0.2, patch_artist=True, notch=True, showmeans=True, meanline=True
        )
        
        # Set box color and add to legend handles
        color = plt.cm.viridis(i / num_vars)
        for box in bp['boxes']:
            box.set(facecolor=color)
        legend_handles.append(bp['boxes'][0])  # Use the first box as the legend handle

    # Set plot title
    ax.set_title(title, pad=10, fontsize=fontsize)
    ax.set_ylabel("Value", fontsize=fontsize, labelpad=labelpad)

    # Adjust x-ticks based on increased spacing
    ax.set_xticks(range(1, num_sites + 1))
    ax.set_xticklabels([f"{label}\n$n={len(groups)}$" for label, groups in zip(labels, scaled_groups[0])], fontsize=fontsize, rotation=0, ha="center")

    # Add legend
    ax.legend(legend_handles, var_names, loc='upper right') 

    ax.tick_params(axis='y', labelsize=fontsize)  # Set y-tick label size
    ax.tick_params(axis='x', labelsize=fontsize)  # Set x-tick label size

    # Add a dashed line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Adjust subplot parameters to ensure consistent plot area width
    plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.4)  # Adjusted bottom margin for rotated labels

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def tukey_anova(df, measure, group):
    """
    The function `tukey_anova` performs a Tukey's HSD test to compare the means of a variable across different groups.
        It takes a DataFrame, the name of the variable to analyze, and the group variable to compare. The function then
        conducts a one-way ANOVA test to determine if there are significant differences in the variable means across the groups.
        If the ANOVA test is significant, it performs a Tukey's HSD test to identify specific group differences. The function
        returns a summary table with the pairwise comparisons between groups, including the group names, mean differences,
        confidence intervals, and p-values.

    Args:
    -------
        df (pandas:DataFrame): DataFrame containing the data
        measure (str): name of the variable to analyze
        group (str): name of the group variable to compare

    Return:
    -------
        tukey_summary (pandas:DataFrame): summary table with pairwise comparisons between groups
    """
    tukey = pairwise_tukeyhsd(endog=df[measure], groups=df[group], alpha=0.05)
    tukey_summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    return tukey_summary

def boxplot_anova(var_name, groups, labels, pval_thresh, fontsize=10, dpi=200):
    """
    The function performs an ANOVA between multiple groups and then visualizes the results using a boxplot.
    It calculates the F-statistic and p-value to compare the means of the groups.
    If the p-value is less than a predefined threshold (`pval_thresh`), it creates a boxplot with the data from all groups,
    highlighting the difference in their distributions.
    The boxplot includes notches and mean lines for a clearer visual representation of the data's central tendency and variance,
    and it's annotated with the p-value and sample sizes for each group.

    Args:
    -------
        var_name (str): name of the variable
        groups (list of pandas.Series): data for each group
        labels (list of str): labels for each group
        pval_thresh (float): p-value threshold for significance
        fontsize (int): font size for labels and ticks

    Return:
    -------
        None
    """
    # perform ANOVA
    f_stat, pval = stats.f_oneway(*groups)

    # calculate scale for y-axis
    max_val = max([group.max() for group in groups])
    scale = 10 ** (np.floor(np.log10(max_val)) - 1)
    exponent = int(np.log10(scale))

    if np.abs(exponent) >= 3:
        groups = [group / scale for group in groups]

    # combine data into single list
    data_to_plot = groups

    # Calculate labelpad and left margin
    labelpad, left = calculate_labelpad_and_left(max([group.max() for group in groups]))

    if pval < pval_thresh:

        # start plot
        fig, ax = plt.subplots(figsize=(3.3, 3.7), dpi=dpi)
        plt.gca().set_facecolor("#f5f5f5" if pval > 0.05 else "white")
        bp = ax.boxplot(
            data_to_plot, patch_artist=True, notch=True, showmeans=True, meanline=True
        )
        colors = ["#5D9B5C", "#B0D2B0", "#FFB347", "#77DD77"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(f"{var_name}\np-val: {pval:.2e}", pad=10, fontsize=fontsize)
        if np.abs(exponent) < 3:
            ax.set_ylabel(f"{var_name}", fontsize=fontsize, labelpad=labelpad)
        else:
            ax.set_ylabel(f"{var_name} (x$10^{{{exponent}}}$)", fontsize=fontsize, labelpad=labelpad)

        ax.set_xticklabels([f"{label}\n$n={len(group)}$" for label, group in zip(labels, groups)], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)  # Set y-tick label size
        ax.tick_params(axis='x', labelsize=fontsize)  # Set x-tick label size

        # the yticklabel should be X.XX, with the decimal always being there
        if np.abs(exponent) >= 3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

        # Adjust subplot parameters to ensure consistent plot area width
        plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.1)

        plt.tight_layout()
        plt.show()

def distribution_grid(ax, df, var, var_name, plot_type, fontsize=10, dpi=200, x_limits=None):
    """
    This function, `distribution`, visualizes the distribution of a specified variable from a given DataFrame.
    It starts by removing any missing values from the selected variable, then calculates its mean, standard deviation, and kurtosis.
    Depending on the `plot_type` specified, it creates either a density plot or a histogram to showcase the distribution,
    annotating the plot with the variable's mean, standard deviation, and kurtosis, along with the sample size.

    Args:
    -------
        ax (matplotlib.axes.Axes): The axes on which to plot.
        df (pandas:DataFrame): DataFrame containing the data
        var (str): variable name
        var_name (str): variable name for the plot title
        plot_type (str): type of plot to be generated
            can be 'density plot' or 'histogram'
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        x_limits (tuple, optional): x-axis limits for the plot (min, max)

    Return:
    -------
        None
    """
    # first, exclude any nans
    data = df[var].dropna()

    # get mean and std
    data_mean = data.mean()
    data_std = data.std()
    data_kurt = stats.kurtosis(data)

    # create density plot
    if plot_type == "density plot":
        sns.kdeplot(data, color="#2E822D", fill=True, ax=ax)
    else:
        ax.hist(data, bins=10, color="#b8d4b4", edgecolor="black")

    # set x-axis limits if provided
    if x_limits:
        ax.set_xlim(x_limits)

    # make the plot
    string = f"$\mu$ = {data_mean:.1f}; $\sigma$ = {data_std:.1f}\n$Kurtosis$ = {data_kurt:.1f}"
    ax.set_title(f"{var_name}\n{string}", pad=10, fontsize=fontsize)
    ax.set_ylabel("Density" if plot_type == "density plot" else "Frequency", fontsize=fontsize)
    ax.set_xlabel(f"$n = {len(data)}$", fontsize=fontsize)

    ax.tick_params(axis='y', labelsize=fontsize)  # Set y-tick label size
    ax.tick_params(axis='x', labelsize=fontsize)  # Set x-tick label size

    plt.tight_layout()

def boxplot_visits(measure, group, df, title, xticks_labels, fontsize=10, dpi=200, filename=None):
    """
    The function `boxplot_visits` creates a box plot to visualize the distribution of residuals across different visits.
        It takes a DataFrame containing the data, the name of the variable to analyze, the group variable to compare,
        the title for the plot, and the labels for the x-axis ticks. The function then plots the data using a box plot,
        with each box representing the distribution of the residuals for a specific visit. The plot includes a dashed line
        at y=0 to indicate the reference point, and it annotates the plot with Tukey's HSD test results for pairwise comparisons
        between visits. This analysis helps identify any significant differences in the residuals across visits.

    Args:
    -------
        measure (str): name of the variable to analyze
        group (str): name of the group variable to compare
        df (pandas:DataFrame): DataFrame containing the data
        title (str): title for the plot
        xticks_labels (list): labels for the x-axis ticks
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    tukey_summary = tukey_anova(df, measure, group)
    if filename is not None:
        print(tukey_summary)
    baseline_comparisons = tukey_summary[(tukey_summary['group1'] == 2) | (tukey_summary['group2'] == 2)].reset_index()

    # Plotting the residuals boxplot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    sns.boxplot(x=group, y=measure, data=df, ax=ax, color='#c1e0b0')
    ax.axhline(y=0, color='r', linestyle='--')

    ## code below added to add horizontal lines for standard deviations
    # Calculate the overall standard deviation of the residuals 
    residuals = df[measure]  # Assuming 'measure' column contains residuals
    std_dev = residuals.std()

    # Add horizontal lines for 1 and 2 standard deviations
    ax.axhline(y=std_dev, color='sandybrown', linestyle='--', linewidth=1, label='1 SD')
    ax.axhline(y=-std_dev, color='sandybrown', linestyle='--', linewidth=1)
    ax.axhline(y=2*std_dev, color='peachpuff', linestyle='--', linewidth=1, label='2 SD')
    ax.axhline(y=-2*std_dev, color='peachpuff', linestyle='--', linewidth=1)

    # Add a legend to explain the lines
    ax.legend()
    ##

    
    # for index, row in baseline_comparisons.iterrows():
    #     x1 = df[group].unique().tolist().index(row['group1'])
    #     x2 = df[group].unique().tolist().index(row['group2'])
    #     y = max(df[measure]) + 6 * index
    #     h = 1
    #     col = 'k'
    #     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    #     ax.text((x1 + x2) * 0.5, y + h, f"p={row['p-adj']:.1e}", ha='center', va='bottom', color=col, fontsize=fontsize)

    xticks_positions = range(len(xticks_labels))
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels, fontsize=fontsize)

    ax.set_ylabel('Residuals', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if filename:
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

def boxplot_visits_multiple(measures, group, df, title, xticks_labels, fontsize=10, dpi=200, filename=None):
    """
    The function `boxplot_visits` creates box plots to visualize the distribution of multiple measures across different visits.
        It takes a DataFrame containing the data, a list of variable names to analyze, the group variable to compare,
        the title for the plot, and the labels for the x-axis ticks. The function then plots the data using multiple box plots,
        with each set of boxes representing the distribution of the measures for a specific visit. The plot includes a dashed line
        at y=0 to indicate the reference point, and it annotates the plot with Tukey's HSD test results for pairwise comparisons
        between visits. This analysis helps identify any significant differences in the measures across visits.

    Args:
    -------
        measures (list): list of names of the variables to analyze
        group (str): name of the group variable to compare
        df (pandas:DataFrame): DataFrame containing the data
        title (str): title for the plot
        xticks_labels (list): labels for the x-axis ticks
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    num_measures = len(measures)
    unique_groups = df[group].unique()
    num_groups = len(unique_groups)
    
    fig, ax = plt.subplots(figsize=(25, 8), dpi=dpi)
    
    # Set colors
    colors = sns.color_palette("viridis", num_measures)
    
    # Calculate positions
    positions = []
    for i in range(num_groups):
        for j in range(num_measures):
            positions.append(i * (num_measures + 1) + j + 1)

    # Plotting each measure's data
    for i, measure in enumerate(measures):
        data = [df[df[group] == grp][measure].values for grp in unique_groups]
        pos = [p for j, p in enumerate(positions) if j % (num_measures) == i]
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, notch=True, showmeans=True, meanline=True)

        # Set box color to specific shades
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
    
    # Set xticks and xticklabels
    xtick_positions = [i * (num_measures + 1) + (num_measures + 1) / 2 for i in range(num_groups)]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticks_labels, fontsize=fontsize)
    
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_ylabel('Values', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # Create legend
    legend_handles = [Patch(color=colors[i], label=measures[i]) for i in range(num_measures)]
    ax.legend(handles=legend_handles, title="Measures", fontsize=fontsize, title_fontsize=fontsize, loc='upper right')
    
    # # Annotate with Tukey's HSD results
    # for measure in measures:
    #     tukey_summary = tukey_anova(df, measure, group)
    #     if filename is None:
    #         print(f"Tukey's HSD results for {measure}:\n", tukey_summary)
    
    if filename:
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

def boxplot_sites(var_name, groups, labels, fontsize=10, dpi=200, filename=None):
    """
    The function boxplot_sites creates a box plot to visualize the distribution of a variable across different sites.
        It takes a list of groups, where each group represents the variable values for a specific site, and a list of labels
        to identify each site. The function then plots the data using a box plot, with each box representing the distribution
        of the variable values for a site. The plot includes the mean, median, and quartiles for each site, allowing for a
        visual comparison of the variable's distribution across different sites.

    Args:
    -------
        var_name (str): name of the variable to plot
        groups (list): list of groups, where each group represents the variable values for a specific site
        labels (list): list of labels to identify each site
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    # Calculate scale for y-axis
    max_val = max([group.max() for group in groups])
    scale = 10 ** (np.floor(np.log10(max_val)) - 1)
    exponent = int(np.log10(scale))

    if np.abs(exponent) >= 3:
        groups = [group / scale for group in groups]

    # Combine data into single list
    data_to_plot = groups

    # Calculate labelpad and left margin
    labelpad, left = calculate_labelpad_and_left(max([group.max() for group in groups]))

    # Start plot
    fig, ax = plt.subplots(figsize=(20, 7.4), dpi=dpi)
    plt.gca().set_facecolor("white")
    bp = ax.boxplot(
        data_to_plot, patch_artist=True, notch=True, showmeans=True, meanline=True
    )

    # Set box color to #5D9B5C
    for box in bp['boxes']:
        box.set(facecolor='#c1e0b0')
    
    ax.set_title(f"{var_name}", pad=10, fontsize=fontsize)
    if np.abs(exponent) < 3:
        ax.set_ylabel(f"{var_name}", fontsize=fontsize, labelpad=labelpad)
    else:
        ax.set_ylabel(f"{var_name} (x$10^{{{exponent}}}$)", fontsize=fontsize, labelpad=labelpad)

    ax.set_xticklabels([f"{label}\n$n={len(group)}$" for label, group in zip(labels, groups)], fontsize=fontsize, rotation=0, ha="center")
    ax.tick_params(axis='y', labelsize=fontsize)  # Set y-tick label size
    ax.tick_params(axis='x', labelsize=fontsize)  # Set x-tick label size

    # the yticklabel should be X.XX, with the decimal always being there
    if np.abs(exponent) >= 3:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Add a dashed line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Adjust subplot parameters to ensure consistent plot area width
    plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.4)  # Adjusted bottom margin for rotated labels

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def boxplot_sites_adj(var_name, groups, labels, fontsize=12, dpi=200, filename=None):
    """
    The function boxplot_sites creates a box plot to visualize the distribution of a variable across different sites.
    This function differs from boxplot_sites only in that it adds horizontal lines indicated +/- 1 and 2 standard deviations for the residuals. 
        It takes a list of groups, where each group represents the variable values for a specific site, and a list of labels
        to identify each site. The function then plots the data using a box plot, with each box representing the distribution
        of the variable values for a site. The plot includes the mean, median, and quartiles for each site, allowing for a
        visual comparison of the variable's distribution across different sites.

    Args:
    -------
        var_name (str): name of the variable to plot
        groups (list): list of groups, where each group represents the variable values for a specific site
        labels (list): list of labels to identify each site
        fontsize (int): font size for labels and ticks
        dpi (int): dots per inch for the plot resolution
        filename (str, optional): filename to save the plot

    Return:
    -------
        None
    """
    # Calculate scale for y-axis
    max_val = max([group.max() for group in groups])
    scale = 10 ** (np.floor(np.log10(max_val)) - 1)
    exponent = int(np.log10(scale))

    if np.abs(exponent) >= 3:
        groups = [group / scale for group in groups]

    # Combine data into single list
    data_to_plot = groups

    # Calculate labelpad and left margin
    labelpad, left = calculate_labelpad_and_left(max([group.max() for group in groups]))

    # Start plot
    fig, ax = plt.subplots(figsize=(20, 7.4), dpi=dpi)
    plt.gca().set_facecolor("white")
    bp = ax.boxplot(
        data_to_plot, patch_artist=True, notch=True, showmeans=True, meanline=True
    )
    
    # Calculate the overall standard deviation of the residuals
    all_residuals = np.concatenate(groups)
    std_dev = np.std(all_residuals)

    # Add horizontal lines for 1 and 2 standard deviations
    ax.axhline(y = std_dev, color='sandybrown', linestyle='--', linewidth=1, label='1 SD')
    ax.axhline(y = -std_dev, color='sandybrown', linestyle='--', linewidth=1)
    ax.axhline(y = 2*std_dev, color='peachpuff', linestyle='--', linewidth=1, label='2 SD')
    ax.axhline(y = -2*std_dev, color='peachpuff', linestyle='--', linewidth=1)
    
    # Add legend for the lines
    ax.legend()

    # Set box color to #5D9B5C
    for box in bp['boxes']:
        box.set(facecolor='#c1e0b0')
    
    ax.set_title(f"{var_name}", pad=10, fontsize=fontsize)
    if np.abs(exponent) < 3:
        ax.set_ylabel(f"{var_name}", fontsize=fontsize, labelpad=labelpad)
    else:
        ax.set_ylabel(f"{var_name} (x$10^{{{exponent}}}$)", fontsize=fontsize, labelpad=labelpad)

    ax.set_xticklabels([f"{label}\n$n={len(group)}$" for label, group in zip(labels, groups)], fontsize=fontsize, rotation=0, ha="center")
    ax.tick_params(axis='y', labelsize=fontsize)  # Set y-tick label size
    ax.tick_params(axis='x', labelsize=fontsize)  # Set x-tick label size

    # the yticklabel should be X.XX, with the decimal always being there
    if np.abs(exponent) >= 3:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Add a dashed line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Adjust subplot parameters to ensure consistent plot area width
    plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.4)  # Adjusted bottom margin for rotated labels

    if filename:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def barplot_counts_by_visit(df, variable, visit_column, title, fontsize=10, dpi=200, filename=None, y_min=None, y_max=None):
    """
    Creates bar plots of task counts by task name, faceted by visit.

    Args:
        df (pandas.DataFrame): DataFrame with 'visit' and 'task_name' columns.
        variable (str): The name of the column to count values from.
        visit_column (str): The name of the visit column.
        title (str): Plot title.
        fontsize (int): Font size.
        dpi (int): DPI.
        filename (str, optional): Save filename.

    Returns:
        None
    """

    # Count task occurrences for each visit
    df_grouped = df.groupby([visit_column, variable]).size().reset_index(name = 'count')
    
    visits = df_grouped[visit_column].unique()
    num_visits = len(visits)

    # Determine the number of rows and columns for the subplots
    ncols = 1  # Number of columns
    nrows = (num_visits + ncols - 1) // ncols 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows), dpi=dpi)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]  # Handle single subplot case

    for i, visit in enumerate(visits):
        visit_data = df_grouped[df_grouped[visit_column] == visit]
       
        ax = axes[i] if i < len(axes) else plt.subplots(figsize=(20,10), dpi=dpi)

        sns.barplot(x=variable, y='count', data=visit_data, ax=ax, color="#4e6641")

        # Add count numbers above the bars
        for p in ax.patches:  # Iterate over the bars (patches)
            height = p.get_height()  # Get the height of the bar
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),  # Annotate
                        ha='center', va='bottom', fontsize=fontsize - 2, xytext=(0, 5),
                        textcoords='offset points')

        ax.margins(y=0.1) # add space above bars to make room for numbers/counts
        ax.set_title(f"{visit}", fontsize=fontsize)
        ax.set_xlabel(variable.replace('_', ' ').title(), fontsize=fontsize)
        ax.set_ylabel("Number completed at this visit", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize, rotation=45)
        ax.tick_params(axis='y', labelsize=fontsize)

        # Set y-axis limits if provided
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        elif y_min is not None:
            ax.set_ylim(ymin=y_min)
        elif y_max is not None:
            ax.set_ylim(ymax=y_max)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):  # Iterate from the next unused subplot to the end
        axes[j].set_axis_off()

    fig.suptitle(title, fontsize=fontsize + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()