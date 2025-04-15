# basic functions for statistical analyses
import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scikit_posthocs as sp
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats

pio.renderers.default = 'notebook'
pio.templates.default = 'simple_white'


from numpy.polynomial.polynomial import polyfit
from plotly.subplots import make_subplots
from scipy.stats import kendalltau, pearsonr, spearmanr

# -----------------------------------------

def compute_agg_data(
    data,
    group_var,
    sep_var,
    overlay_var,
    plot_var,
    central_tendency='mean',
    error_type='sem',
    observed=False
    ):
    '''
    Takes in an input dataframe and aggregates it using functions for central tendency and error.
    Returns a dataframe for central tendency, one for error, and associated xlabels.

    Parameters
    ==========
    data : pandas dataframe
        aggregated input dataframe in long format
    group_var : str
        name of the column in data by which data will be grouped.
    sep_var, overlay_var : str
        names of the columns for which there should be one unique value. Column will be aggregated by this one unique
        value, or if there is more than one unique value, will be changed to np.nan out of caution.
    plot_var : str
        name of the column in data representing the readout of interest to aggregate.
    central_tendency : str
        which measure of central tendency to use when aggregating data. Default is 'mean'.
    error_type : str
        which measure of error to use when aggregating data. Default is 'sem'.
    observed : bool
        whether or not to include missing data in categorical variables. Default is False.
    '''
    grouped_data = data[[sep_var, overlay_var, group_var, plot_var]].groupby(group_var, observed=observed)

    unique_val_func = lambda x: x.unique()[0] if x.nunique() == 1 else np.nan
    agg_data = grouped_data.agg({sep_var:unique_val_func,
                                        overlay_var:unique_val_func,
                                        plot_var:central_tendency}).sort_index()
    error_data   = grouped_data.agg({sep_var:unique_val_func,
                                        overlay_var:unique_val_func,
                                        plot_var:error_type}).sort_index()
    
    xlabels = agg_data.index.values
    agg_data = agg_data.reset_index().rename(columns={'index': group_var})
    error_data   = error_data.reset_index().rename(columns={'index': group_var})
    agg_data.index = xlabels
    error_data.index = xlabels

    return agg_data, error_data, xlabels


# -----------------------------------------


def agg_plot(
    data,
    plot_var,
    color_var,
    sep_var=None,
    group_var=None,
    overlay_var=None,
    central_tendency='mean',
    error_type='sem',
    datapoint_var='Mouse',
    plot_mode='bar',
    bar_mode='group',
    plot_datapoints=False,
    plot_datalines=False,
    agg_marker_size=15,
    agg_marker_shape='',
    error_width=3,
    colors='slategrey',
    color_datapoints=False,
    opacity=1,
    title_text=None,
    x_title=None,
    y_title=None,
    y_range=None,
    x_dtick=None,
    y_dtick=None,
    match_y_ranges=True,
    add_hline=True,
    hline_y=0,
    text_size=18,
    font_family='Arial',
    plot_width=600,
    plot_height=600,
    tick_angle=45,
    h_spacing=0.1,
    save_path=None,
    plot_scale=5
    ):
    '''
    A general purpose plotting function to plot summary data in bar, point, or line graph form, separating
    across up to three variables. Returns an interactive plotly graph, and optionally saves the graph in static
    or interactive format.

    Ensure that group_var, sep_var, and overlay_var are of dtype pd.Categorical, or None. At least one of
    these variables must not be None.

    For coloring, color_var must be one of sep_var, group_var, or overlay_var, or None. If one of those variables,
    colors must be a dict with a key-value pair for each unique entry and an associated color, or a string to uniformly
    color all the datapoints. If color_var is None, colors must be a string.

    Parameters
    ==========
    data : pandas dataframe
        aggregated input dataframe in long format
    plot_var : str
        name of the column in data representing the readout of interest
    color_var : str
        name of the column by which to color code the data. Must be of type pd.Categorical.
    group_var : str
        name of the column in data by which to group within a subplot. Must be of type pd.Categorical. Default is None.
    sep_var : str
        name of the column in data by which to separate across subplots. Must be of type pd.Categorical. Default is None.
    overlay_var : str
        name of the column in data by which to overlay within a subplot. Must be of type pd.Categorical. Default is None.
    central_tendency : str
        which measure of central tendency to use when aggregating data. Default is 'mean'.
    error_type : str
        which measure of error to use when aggregating data. Default is 'sem'.
    datapoint_var : str
        name of the column in data representing each individual subject name. Default is 'Mouse'.
    plot_mode : str
        one of 'bar', 'line', or 'point' for which type of plot is desired. Default is 'bar'.
    bar_mode : str
        how to group bars along the overlay_var variable. One of 'group' or 'overlay'. Only used if plot_mode=='bar'. Default is 'group'.
    plot_datapoints, plot_datalines : bool
        whether or not to plot individual subject datapoints of datalines. Defaults are False.
    agg_marker_size : int or float
        size of the datapoint representing the aggregated data of each group. Only used if plot_mode=='point'. Default is 15.
    agg_marker_shape : str
        pattern of bars in each bar plot. Only used if plot_mode=='bar'. Default is ''.
    error_width : int or float
        width or error bars. Default is 3.
    colors : dict or str
        dictionary where each key is each unique color_var type and each value is its corresponding color, or a string representing
        one color for all datapoints. If color_var is None, must be a string. Default is 'slategrey'.
    color_datapoints : bool
        whether or not to individually color individual subjects' datapoints. Otherwise, all individual datapoints are
        black. Only used if plot_datapoints or plot_datalines==True. Default is False.
    opacity : int or float
        number representing opacity of aggregated data. Must range between [0,1]. Default is 1.
    title_text : str
        master title of the entire plot. Default is None.
    x_title, y_title : str
        title of the y-axis. Default is None.
    y_range : tuple
        tuple representing min and max y-axis values of plot range. If None, it is the range of the input data. Default is None.
    match_y_ranges : bool
        whether or not to match the y-axis ranges across subplots. Default is True.
    add_hline: bool
        whether or not to add a solid black line at a specified y-axis value. Default is True.
    hline_y : int or float
        y-axis value at which to draw a solid black line. Only used if add_hline==True. Default is 0.
    text_size : int or float
        size of all the text in the plot. Default is 18.
    font_family : str
        font family used in the plot. Default is 'Arial'.
    plot_width, plot_height : int or float
        width and height of the plot. Defaults are 600 and 600.
    tick_angle : int
        angle at which x-axis label text is displayed. Default is 45.
    h_spacing : float
        spacing between subplots. Only used if sep_var is not None. Default is 0.1.
    save_path : str
        file path location including filename where plot should be saved. If None, plot will not be saved. Default is None.
    plot_scale : int
        size scaling of the plot. Only used if save_path is not None and if save_path extension is of a static type. Default is 5.
    '''

    vars_dict = {
        'sep_var'    : sep_var,
        'group_var'  : group_var,
        'overlay_var': overlay_var
        }
    
    # configure colors
    if not (color_var in [sep_var, group_var, overlay_var]) | (color_var is None):
        raise Exception("Invalid color_var, must be equal to sep_var, group_var, overlay_var, or None.")
    if (type(colors) == str) & (color_var is not None):
        colors = {unique_val:colors for unique_val in data[color_var].unique()}     # set uniform color if color is a string
    if (type(colors) != str) & (color_var is None):
        raise Exception("If color_var is None, colors must be a string.")

    # add placeholder for columns that are not specified
    for var_key, var in vars_dict.items():
        if var is None:
            data[var_key] = ''
            data[var_key] = pd.Categorical(data[var_key])
            vars_dict[var_key] = var_key

    # make sure all relevant variables are of type categorical
    assert np.all([data[col].dtype.name == 'category' if col is not None else True for col in vars_dict.values()]), \
        '{s}, {g}, and {o} must be of type pd.Categorical'.format(s=vars_dict['sep_var'],
                                                                g=vars_dict['group_var'],
                                                                o=vars_dict['overlay_var'])

    # initialize plot
    subplot_titles = data[vars_dict['sep_var']].unique().sort_values()
    fig = make_subplots(rows=1, cols=len(subplot_titles), subplot_titles=subplot_titles,
                        horizontal_spacing=h_spacing, shared_yaxes=match_y_ranges)
    
    # separate data by variables and plot
    for i, sep in enumerate(data[vars_dict['sep_var']].unique().sort_values()):
        sep_data = data[data[vars_dict['sep_var']] == sep]
        for overlay in sep_data[vars_dict['overlay_var']].unique().sort_values():
            overlay_data = sep_data[sep_data[vars_dict['overlay_var']] == overlay]
            agg_data, error_data, xlabels = compute_agg_data(
                data             = overlay_data,
                sep_var          = vars_dict['sep_var'],
                overlay_var      = vars_dict['overlay_var'],
                group_var        = vars_dict['group_var'],
                plot_var         = plot_var,
                central_tendency = central_tendency,
                error_type       = error_type
                )
            agg_colors = [colors[c] for c in agg_data[color_var].values] if color_var is not None else \
                        np.repeat(colors, agg_data.shape[0])

            if plot_mode.lower() == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=xlabels,
                        y=agg_data[plot_var][xlabels].values,
                        error_y=dict(type='data', array=error_data[plot_var][xlabels].values, visible=True, width=error_width),
                        name=overlay,
                        marker=dict(color=agg_colors, line=dict(width=1, color='black'), opacity=opacity),
                        marker_pattern_shape=agg_marker_shape
                    ),
                    row=1,
                    col=i+1
                )
            elif plot_mode.lower() == 'point':
                fig.add_trace(
                    go.Scattergl(
                        x=xlabels,
                        y=agg_data[plot_var][xlabels].values,
                        error_y=dict(type='data', array=error_data[plot_var][xlabels].values, visible=True, width=error_width),
                        name=overlay,
                        mode='markers',
                        marker=dict(color=agg_colors, size=agg_marker_size, line=dict(width=1, color='black'), opacity=opacity),
                    ),
                    row=1,
                    col=i+1
                )
            elif plot_mode.lower() == 'line':
                fig.add_trace(
                    go.Scattergl(
                        x=xlabels,
                        y=agg_data[plot_var][xlabels].values,
                        error_y=dict(type='data', array=error_data[plot_var][xlabels].values, visible=True, width=error_width),
                        name=overlay,
                        mode='lines+markers',
                        marker=dict(color=agg_colors, size=agg_marker_size),
                        line=dict(color=agg_colors[0], width=4),
                    ),
                    row=1,
                    col=i+1
                )
            else:
                raise Exception("Invalid plot_mode. Must be one of 'bar', 'point', or 'line'.")
        
            # plot individual datapoints
            if plot_datapoints | plot_datalines:
                datapoint_plot_mode = 'lines+markers' if (plot_datapoints & plot_datalines) else 'lines' if plot_datalines else 'markers'
                for point in overlay_data[datapoint_var].unique():
                    point_data = overlay_data[overlay_data[datapoint_var] == point].sort_values(vars_dict['group_var'])
                    point_color = [colors[c] for c in point_data[color_var].values] if color_var is not None and color_datapoints else \
                                   np.repeat('slategrey', point_data.shape[0])
                    line_color = point_color[0]
                    fig.add_trace(
                        go.Scattergl(
                            x=point_data[vars_dict['group_var']].values,
                            y=point_data[plot_var].values,
                            mode=datapoint_plot_mode,
                            marker=dict(color=point_color, symbol='circle-open', opacity=0.4, size=10),
                            line=dict(width=1, color=line_color),
                            name=str(point),
                        ),
                        row=1,
                        col=i+1
                    )

    # configure plot
    if add_hline:
        fig.add_hline(y=hline_y, row=1, col='all', line_width=1, opacity=1, line_color='black')
    fig.update_layout(
        dragmode="pan",
        font=dict(size=text_size, family=font_family),
        title_text=title_text,
        yaxis_title=y_title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
        barmode=bar_mode
    )
    fig.update_xaxes(tickangle=tick_angle, title_text=x_title, dtick=x_dtick)
    fig.update_yaxes(range=y_range, dtick=y_dtick)

    # save plot
    if save_path is not None:
        save_dirname = os.path.dirname(save_path)
        if not os.path.exists(save_dirname):
            os.mkdir(save_dirname)
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


# -----------------------------------------


def plot_correlation(
    data_x,
    data_y,
    x_title,
    y_title,
    groups=None,
    colors='slategrey',
    color_palette=px.colors.qualitative.Safe,
    corr_method='pearson',
    title="",
    textinfo=None,
    plot_fits=True,
    plot_identity=False,
    same_xy_scale=False,
    x_range=None,
    y_range=None,
    text_size=18,
    font_family='Arial',
    marker_size=7,
    outline_width=1,
    line_width=2,
    point_opacity=0.8,
    plot_height=600,
    plot_width=600,
    save_path=None,
    plot_scale=5
    ):
    """
    Plots correlation between two input vectors (data_x and data_y). Given an optional associated vector
    called groups, 
    Given two sorted vectors ('data_x' & 'data_y') and an optional associated 'groups' vector, plot the
    correlation between them, coloring each group separately.

    Parameters:
    ==========
    data_x, data_y : numpy 1d vector or list
        data values to be plotted along x-axis and y-axis respectively
    x_title, y_title : str
        title text for x-axis and y-axis respectively
    groups : numpy 1d vector or list
        the group identity that each datapoint belongs to. Must be of same length as data_x and data_y, or None. Default is None.
    colors : None or str or vector/list of colors
        colors to plot the datapoints. If None, color_palette will be discretized based on the number of
        unique values in groups. Each datapoint will be assigned a color based on the group it belongs to.
        If a list/vector of colors, length must be equal to length of data and the colors should match the
        group that each datapoint belongs to. If a string, all datapoints will be that color. Default is 'slategrey'.
    color_palette : plotly express color palette
        a color palette which is only used if colors = None (see above argument, colors). Default is px.colors.qualitative.Safe.
    corr_method : str
        which type of correlation test to perform. One of 'pearson', 'spearman', 'kendall'. Default is 'pearson'.
    title : str
        title for overall plot. Note that the title will also include the r and p values for
        line of best fit following the text provided. Default is ''.
    textinfo : None or 1d vector or list
        the text that should appear over each datapoint during hover. If not None, must be of same length as data. Default is None.
    plot_fits : boolean
        whether or not to plot the lines of best fit for each unique group type. If False, only datapoints will be plotted.
        Default is True.
    plot_identity : boolean
        whether or not to plot the identity line as a black dotted line. Default is False.
    same_xy_scale : boolean
        whether to set the ranges of the x- and y-axes to the same ranges. Default is False.
    x_range, y_range : None or tuple
        the range to set the x-axis and y-axis, respectively. If both are of type tuple and same_xy_scale
        is set to False, the plot will be updated with the specified axis ranges. Defaults are None.
    text_size : int or float
        size of all the text in the plot. Default is 18.
    font_family : str
        font family used in the plot. Default is 'Arial'.
    marker_size : float
        size of each datapoint. Default is 7.
    outline_width : float
        width of the outlines of each datapoint. Default is 1.
    line_width : float
        width of the line of best fit. Only used if plot_fits is True. Default is 2.
    point_opacity : float
        how opaque each datapoint should be, from [0,1]. Default is 0.8.
    plot_height, plot_width : int
        the height and width, respectively, of the entier plot. Defaults are 600 and 600, respectively.
    save_path : str
        file path location including filename where plot should be saved. If None, plot will not be saved. Default is None.
    plot_scale : int
        size scaling of the plot. Only used if save_path is not None and if save_path extension is of a static type. Default is 5.
    """

    # prepare data
    groups = np.array(groups) if groups is not None else np.repeat('', len(data_x))

    if colors is None:
        len(np.unique(groups))
        colors = np.repeat("x", len(data_x)).astype(object)
        for i, group in enumerate(np.unique(groups)):
            colors[groups == group] = color_palette[i]

    # compute correlation
    corr_data = pd.DataFrame(
        {"X": data_x,
         "Y": data_y,
         "groups": groups,
         "colors": colors}
    )
    corr_dict = {'pearson'  : pearsonr,
                 'spearman' : spearmanr,
                 'kendall'  : kendalltau}
    corr_func = corr_dict[corr_method]
    r, p = corr_func(corr_data.X, corr_data.Y)

    # plot correlation
    fig = go.Figure()
    if textinfo is None:
        textinfo = np.repeat("", corr_data.shape[0])
    fig.add_trace(
        go.Scattergl(
            x=corr_data.X,
            y=corr_data.Y,
            mode="markers",
            marker_size=marker_size,
            marker=dict(
                line=dict(color="black", width=outline_width), color=colors, opacity=point_opacity
            ),
            text=[
                text + ", (" + str(np.round(x, 4)) + ", " + str(np.round(y, 4)) + ")"
                for text, (x, y) in zip(textinfo, zip(corr_data.X, corr_data.Y))
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )

    # plot lines of best fit
    if plot_fits:
        for group in np.unique(groups):
            sub_x_data = corr_data[groups == group].X
            sub_y_data = corr_data[groups == group].Y
            color = corr_data[groups == group]['colors'].values[0]
            b, m = polyfit(sub_x_data, sub_y_data, 1)
            line_x = np.linspace(sub_x_data.min(), sub_x_data.max(), 10)
            fig.add_trace(
                go.Scattergl(
                    x=line_x,
                    y=(m * line_x + b),
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    name=group,
                )
            )

    # plot identity line
    if plot_identity:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.add_trace(
            go.Scattergl(
                x=np.linspace(min_val, max_val),
                y=np.linspace(min_val, max_val),
                mode="lines",
                line=dict(color="black", width=line_width, dash='dash'),
                name="Identity<br>Line",
            )
        )

    # configure plot
    if same_xy_scale:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.update_xaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=x_title,
        )
        fig.update_yaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=y_title,
        )
    else:
        fig.update_xaxes(title_text=x_title)
        fig.update_yaxes(title_text=y_title)
        if (type(x_range) is tuple) & (type(y_range) is tuple):
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

    fig.update_layout(
        dragmode="pan",
        font=dict(size=text_size, family=font_family),
        title_text=title
        + " r = "
        + str(np.round(r, 4))
        + ", p = "
        + str(np.round(p, 6)),
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
    )

    # save plot
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)

def get_linear_regression_traces(df, metric, x_col):
    
    line_trace = None
    
    df = df.dropna(subset=[x_col, metric])
    X = sm.add_constant(df[[x_col]])
    y = df[metric]
    model = sm.OLS(y, X).fit()

    r_squared = model.rsquared
    coef = model.params[x_col]
    intercept = model.params['const']
    pval = model.pvalues[x_col]

    # Prepare regression line
    x_vals = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_vals = intercept + coef * x_vals

    scatter_trace = go.Scatter(
        x=df[x_col], y=df[metric],
        mode='markers',
        #marker=dict(color='royalblue', opacity=0.5, size=6),
        marker=dict(color='#1f77b4', opacity=0.5, size=6),
        name='Data'
    )

    if x_col != 'Grade_Ordinal':
        line_trace = go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            line=dict(color='slategray', width=2),
            name='Fit'
        )

    return r_squared, coef, pval, scatter_trace, line_trace

def plot_regression(scatter_trace, line_trace, r_squared, coef, pval, metric, x_col):

    fig = go.Figure()
    fig.add_trace(scatter_trace)
    fig.add_trace(line_trace)

    # Add annotations
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.98, showarrow=False,
        text=f"R² = {r_squared:.3f}<br>β = {coef:.3f}<br>pval: {pval:.2e}",
        align='left',
        font=dict(size=12)
    )

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=metric.replace('_', ' ').title(),
        width=600,
        height=400
    )

    return fig

def plot_regression_subplots(metrics, 
                             new_df, 
                             variable='Grade_Ordinal', 
                             cols=1, 
                             title=''):
    
    num_plots = len(metrics)
    rows = math.ceil(num_plots / cols)

    # Create subplot layout
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in metrics],
        #shared_xaxes=True,
    )

    # Add traces to each subplot
    for i, metric in enumerate(metrics):
        row = (i // cols) + 1
        col = (i % cols) + 1

        r_squared, coef, pval, scatter_trace, line_trace = get_linear_regression_traces(new_df, metric, variable)
        fig.add_trace(scatter_trace, row=row, col=col)

        if line_trace is not None:
            # Only add line trace if it exists
            fig.add_trace(line_trace, row=row, col=col)

        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.98, showarrow=False,
            text=f"R² = {r_squared:.3f}<br>β = {coef:.3f}<br>pval: {pval:.2e}",
            align='left',
            font=dict(size=12),
            row=row, col=col
        )
        fig.update_xaxes(title_text=variable, row=row, col=col)
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)

    fig.update_layout(
        title=title,
        title_x=0.5,
        height=400 * rows,
        width=1000,
        showlegend=False,
        template='simple_white'
    )

    return fig


def compute_cohens_d(df, metric, group_col='Grade Level', grade_order=['Kindergarten', 'Elementary', 'Teen', 'College', 'Adult']):
    
    d_matrix = pd.DataFrame(index=grade_order, columns=grade_order, dtype=float)
    for i, g1 in enumerate(grade_order):
        for j, g2 in enumerate(grade_order):
            if i < j:
                group1 = df[df[group_col]==g1][metric].dropna()
                group2 = df[df[group_col]==g2][metric].dropna()
                mean1, mean2 = group1.mean(), group2.mean()
                sd1, sd2 = group1.std(), group2.std()
                n1, n2 = group1.count(), group2.count()
                pooled_sd = np.sqrt(((n1 - 1)*sd1**2 + (n2 - 1)*sd2**2) / (n1+n2-2))
                d = (mean1 - mean2) / pooled_sd if pooled_sd != 0 else 0
                d_matrix.loc[g1, g2] = d
                d_matrix.loc[g2, g1] = -d  # symmetric differences
            elif i == j:
                d_matrix.loc[g1, g2] = 0
    return d_matrix

def plot_posthoc_heatmap(df, metric, group_col='Grade Level', grade_order=None, alpha=0.05):
    grade_order = grade_order or ['Kindergarten', 'Elementary', 'Teen', 'College', 'Adult']
    # Compute Dunn's test p-values:
    pvals = sp.posthoc_dunn(df, val_col=metric, group_col=group_col, p_adjust='bonferroni')
    pvals = pvals.loc[grade_order, grade_order]
    
    # Compute pairwise effect sizes (Cohen's d)
    d_matrix = compute_cohens_d(df, metric, group_col, grade_order)
    
    # Prepare annotations: effect size value and stars for significance
    annotations = []
    for i, g1 in enumerate(grade_order):
        for j, g2 in enumerate(grade_order):
            d_val = d_matrix.loc[g1, g2]
            p_val = pvals.loc[g1, g2]
            stars = ""
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            annotations.append(dict(
                x=g2,
                y=g1,
                text=f"{d_val:.2f}{stars}",
                showarrow=False,
                font=dict(color='black', size=10)
            ))
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=d_matrix.values,
        x=grade_order,
        y=grade_order,
        colorscale='RdBu',
        reversescale=True,
        colorbar=dict(title="Cohen's d"),
        zmid=0
    ))
    fig.update_layout(
        title=f"Pairwise Effect Sizes for {metric.replace('_',' ').title()}<br>(Stars: p<0.05)",
        annotations=annotations,
        template='plotly_white',
        width=500,
        height=500
    )
    #fig.show()
    return fig


def plot_length_dependence_by_grade(df, metric, grade_order = ['Kindergarten', 'Elementary', 'Teen', 'College', 'Adult']):

    """
    Creates a row of scatter plots showing the relationship between excerpt length and a given metric, 
    faceted by grade level.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'Excerpt_Length', the metric of interest, and 'Grade Level' columns.
    metric : str
        The name of the column in `df` representing the metric to plot on the y-axis.
    grade_order : list of str, optional
        List of grade level labels specifying the order in which plots should appear.

    Returns:
    -------
    plotly.graph_objects.Figure
        A Plotly figure with subplots showing metric vs. excerpt length per grade level.
    """

    rows = 1
    cols = 5

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=grade_order,
        shared_yaxes=True,
        horizontal_spacing=0.03
    )

    for i, grade in enumerate(grade_order):
        subset = df[df['Grade Level'] == grade]
        fig.add_trace(
            go.Scatter(
                x=subset['Excerpt_Length'],
                y=subset[metric],
                mode='markers',
                marker=dict(size=5, opacity=0.5, color='royalblue'),
                name=grade,
                showlegend=False
            ),
            row=1, col=i + 1
        )

    fig.update_layout(
        height=350, width=250*5,
        title_text=f'{metric.replace("_", " ").title()} vs Excerpt Length by Grade Level',
        template='plotly_white'
    )
    return fig
