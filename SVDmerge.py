"""
SVDmerge: a spectral method to remove batch effects.

Francesc Font-Clos
Nov 2018
"""
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


def check_data_integrity(
    expr=None,
    meta=None,
    column=None,
    verbose=False,
    plot_only=False
):
    """
    Check data integrity.

    This function performs some basic checks on the provided data.
    In particular, it checks that:

    - expression and metadata dataframes have been supplied
    - a column from metadata has been specified
    - the indices of expr and meta coincide
    - meta[column] has exactly two unique labels (if not plotting)


    In addition, it gives warnings (when verbose is True) if
    indices contain duplicate labels.

    Parameters
    ----------
    expr : pandas.DataFrame
        gene-expression (samples in rows, genes in columns)
    meta : pandas.DataFrame
        metadata
    column : str
        column of meta used to group samples
    verbose : bool
        print warnings or not
    plot_only : bool
        if True, does not check that exactly two labels exist

    """
    # Check if all data was supplied
    if expr is None:
        raise ValueError(
            "You must supply the expression matrix as a pandas Dataframe")

    if meta is None:
        raise ValueError(
            "You must supply the clinical data as a pandas Dataframe")

    if column is None:
        raise ValueError(
            "You must indicate which column of meta sohuld be used to"
            "differentiate samples")

    # Check that indices of expr and meta coincide
    if set(expr.index) != set(meta.index) or expr.shape[0] != meta.shape[0]:
        raise ValueError("Indices of expr and meta must coincide")

    # Check unique labels in indices
    if max(Counter(expr.index).values()) > 1:
        if verbose:
            print("Warning: there are non-unique labels in expr index")
    if max(Counter(meta.index).values()) > 1:
        if verbose:
            print("Warning: there are non-unique labels in meta index")

    # Check that column exists in meta df
    if column not in meta.columns:
        raise ValueError("Column '%s' not found in 'meta' dataframe" % column)

    # If not plotting, check that column has only two unique labels
    if plot_only is False:
        if np.unique(meta[column]).shape[0] != 2:
            raise ValueError(
                "Column '%s' in 'meta' must contain exactly two unique labels" % column)


def pca_df(df):
    """

    """
    return pd.DataFrame(
        data=PCA(whiten=True).fit_transform(df),
        index=df.index,
        columns=["pca" + str(i) for i in range(min(df.shape))]
    )


def plot_pca_2d(
    expr=None,
    meta=None,
    column=None,
    figsize=(6, 5),
    x_axis="pca0",
    y_axis="pca1",
    colors=None,
    two_colors=["green", "red"],
    legend=True,
    ax=None,
    verbose=False,
    **kwargs
):
    """
    Plot a 2D representation via PCA.

    This function plots a 2-dimensional projection of a dataset
    using PCA. Samples are grouped via meta[column] and colored
    accordingly. By default, the first two principal components
    are plotted. To plot further components, change the value
    of x_axis='pca0' and y_axis='pca1'.

    Additional keyword arguments are passed onto plt.scatter().

    Parameters
    ----------
    expr : pandas.DataFrame
        gene-expression matrix (samples in rows, genes in columns).
    meta : pandas.DataFrame
        metadata matrix (clinical data, samples in rows, features in columns).
    column : str
        column of 'meta' used to group samples.
    figsize : (int,int)
        Size of the figure.
        Defaults to (6,5).
    x_axis : str
        Principal component to plot on x axis.
        Defaults to "pca0".
    y_axis : str
        Principal component to plot on x axis.
        Defaults to "pca1".
    colors : list of RGB colors
        Colors to be used if there are more than two groups.
        Defaults to sns.color_palette().
    two_colors : [str,str]
        Colors to be used if there are two groups.
        Defaults to ["green","red"].
    legend : bool
        Plot the legend.
    ax : matplotlib.axes.Axes
        Axes object to do the plot. If not given,
        a new axes is generated.
    verbose : bool
        Be verbose or not

    """

    # Checks
    check_data_integrity(expr=expr, meta=meta, column=column,
                         verbose=verbose, plot_only=True)

    # PCA dataframe
    pca = PCA(whiten=True).fit(expr)
    expr_pca = pd.DataFrame(
        data=pca.transform(expr),
        index=expr.index,
        columns=["pca" + str(i) for i in range(min(expr.shape))],
    )

    # Colors
    n_groups = np.unique(meta[column]).shape[0]
    if n_groups == 2:
        colors = two_colors
    elif colors is None:
        if n_groups <= 6:
            colors = sns.color_palette(n_colors=n_groups)
        else:
            colors = sns.color_palette(palette="Dark2", n_colors=n_groups + 1)
    else:
        assert len(colors) == n_groups

    # Create axis if not given
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter plots
    for i, (lab, df) in enumerate(expr_pca.groupby(meta[column])):
        ax.scatter(df[x_axis], df[y_axis], label=lab,
                   color=colors[i], **kwargs)

    # Legend
    if legend:
        ax.legend(loc=(1, 0), fontsize=14)

    # Big capital labels
    ax.set_xlabel(x_axis.upper(), fontsize=14)
    ax.set_ylabel(y_axis.upper(), fontsize=14)


def onestep_filter(expr=None, meta=None, column=None, verbose=False):
    """
    Perform a single SVD filtering step.

    Parameters
    ----------
    expr : pandas.DataFrame
        gene-expression matrix (samples in rows, genes in columns).
    meta: pandas.DataFrame
        metadata matrix (clinical data, samples in rows, features in columns).
    colum: str
        column of 'meta' used to group samples.

    """
    # Check data integrity
    check_data_integrity(expr=expr, meta=meta, column=column, verbose=verbose)

    # Create groups
    lab1, lab2 = np.unique(meta[column].values)

    # Create PCA dataframe
    pca = PCA(whiten=True).fit(expr)
    expr_pca = pd.DataFrame(
        data=pca.transform(expr),
        index=expr.index,
        columns=["pca" + str(i) for i in range(min(expr.shape))],
    )

    # Compute Kolmogorov-Smirnov tests
    pvals = ([ks_2samp(
        expr_pca.loc[meta[column] == lab1, pca_col],
        expr_pca.loc[meta[column] == lab2, pca_col]
    ).pvalue for pca_col in expr_pca.columns])

    min_pval_col = np.argmin(np.array(pvals))

    if verbose:
        if min_pval_col > 1:
            print("The first %d principal components where set to zero" %
                  min_pval_col)
        if min_pval_col == 1:
            print("The first principal component was set to zero")
        if min_pval_col == 0:
            print("No principal components where set to zero")
        print("")

    # set first min_pval_col components to zero
    expr_pca.iloc[:, :min_pval_col] = 0

    # return values in dataframe
    return pd.DataFrame(
        data=pca.inverse_transform(expr_pca.values),
        index=expr.index,
        columns=expr.columns
    )


def twostep_filter(expr_list=None, meta_list=None, column=None, verbose=False):
    """
    Two step filter.

    This function merges a collection of datasets,
    removing batch effects via the two-step SVD merging method.

    Parameters
    ----------
    expr_list : list of pandas.DataFrame 
        List of dataframes containing gene-expression values,
        where each dataframe corresponds to a batch.
    meta_list : list of pandas.DataFrame
        List of dataframes containing clinical data, ordered
        as in expr_list.
    column : str
        A column common to all dataframes in meta_list,
        to be used to define two groups (usually healthy/disease).

    Return
    ------
    expr : pandas.DataFrame
        The merged gene-expression values.


    Example
    -------

    ``` 
    > import SVDmerge
    > import pandas as pd
    > 
    > # merge the expression data
    > expr = SVDmerge.twostep_filter(
    >   expr_list = [expr1,expr2,expr3],
    >   meta_list = [meta1,meta2,meta3],
    >   column = "state" )
    >
    > # concatenate the metadata
    > meta = pd.concat([meta1,meta2,meta3])
    >
    > # plot
    > SVDmerge.plot_pca_2d(expr=expr, meta=meta, column="state")
    >
    ```
    """
    # First filter batch by batch
    exprs_1f = []
    for i, (expr, meta) in enumerate(zip(expr_list, meta_list)):
        if verbose:
            print("Processing batch %d..." % i)
        g = onestep_filter(expr=expr, meta=meta,
                           column=column, verbose=verbose)
        exprs_1f.append(g)

    # merge batches
    if verbose:
        print("Merging batches...")
    expr_1f = pd.concat(exprs_1f, join="inner")
    meta = pd.concat(meta_list, join="inner")

    return onestep_filter(
        expr=expr_1f,
        meta=meta,
        column=column,
        verbose=verbose)
