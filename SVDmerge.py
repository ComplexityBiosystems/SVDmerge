import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

def check_data_integrity(geno=None,pheno=None,column=None,verbose=False,plot_only=False):

    # Check if all data was supplied
    if geno is None:
        raise NameError("You must supply the expression matrix as a pandas Dataframe") 
    
    if pheno is None:
        raise NameError("You must supply the clinical data as a pandas Dataframe") 
    
    if column is None:
        raise NameError("You must indicate which column of pheno sohuld be used to differentiate samples") 
    
    # Check that indices of geno and pheno coincide
    if set(geno.index)!=set(pheno.index) or geno.shape[0]!=pheno.shape[0]:
        raise NameError("Indices of geno and pheno must coincide")

    # Check unique labels in indices
    if max(Counter(geno.index).values())>1:
        if verbose:print "Warning: there are non-unique labels in geno index"
    if max(Counter(pheno.index).values())>1:
        if verbose:print "Warning: there are non-unique labels in pheno index"
    
    # Check that column exists in pheno df
    if column not in pheno.columns:        
        raise NameError("%s not found in pheno dataframe"%column) 
    
    # If not plotting, check that column has only two unique labels
    if plot_only is False:
        if np.unique(pheno[column]).shape[0]!=2:
            raise NameError("Column %s in pheno must contain exactly two unique labels")



def onestep_filter(geno=None,pheno=None,column=None,verbose=False):
    """
    Performs a single SVD filtering step.
    geno: expression matrix
    pheno: clinical data
    colum: column of pheno to be used to define groups
    """

    # Check data integrity
    check_data_integrity(geno=geno,pheno=pheno,column=column,verbose=verbose)    

    # Create groups
    lab1,lab2 = np.unique(pheno[column].values)

    # Create PCA dataframe
    pca = PCA(whiten=False).fit(geno)
    geno_pca = pd.DataFrame(
        data = pca.transform(geno),
        index = geno.index,
        columns = ["pca"+str(i) for i in range(min(geno.shape))],
        )
    
    # Compute Kolmogorov-Smirnov tests
    pvals =  ([ks_2samp(
        geno_pca.loc[pheno[column]==lab1,pca_col],
        geno_pca.loc[pheno[column]==lab2,pca_col]
        ).pvalue for pca_col in geno_pca.columns ])

    min_pval_col = np.argmin(np.array(pvals))    


    if verbose:
        if min_pval_col>1: print "The first %d principal components where set to zero"%min_pval_col
        if min_pval_col==1: print "The first principal component was set to zero"
        if min_pval_col==0: print "No principal components where set to zero"

    # set first min_pval_col components to zero
    geno_pca.iloc[:,:min_pval_col] = 0

    
    # return values in dataframe
    return pd.DataFrame(
        data = pca.inverse_transform(geno_pca.values),
        index = geno.index,
        columns = geno.columns 
        )



def plot_pca_2d(geno=None,pheno=None,column=None,figsize=(6,5),x_axis="pca0",y_axis="pca1",two_colors=["green","red"],verbose=False):
    """
    Plot the projection onto two principal components,colored by column.
    """
    # CHECKS
    check_data_integrity(geno=geno,pheno=pheno,column=column,verbose=verbose,plot_only=True)    

    # create pca dataframe
    pca = PCA(whiten=True).fit(geno)
    geno_pca = pd.DataFrame(
        data = pca.transform(geno),
        index = geno.index,
        columns = ["pca"+str(i) for i in range(min(geno.shape))],
        )
 
    n_groups = np.unique(pheno[column]).shape[0]
    if n_groups ==2:
        colors = two_colors
    else:
        if n_groups<=6:
            colors = sns.color_palette(n_colors=n_groups)
        else:
            colors = sns.color_palette(palette="Dark2",n_colors=n_groups+1)
    #plt.figure(figsize=figsize)
    
    for i,(lab,df) in enumerate(geno_pca.groupby(pheno[column])):
        plt.scatter(df[x_axis],df[y_axis],label=lab,color=colors[i])
    plt.legend(loc=(1,0),fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel(x_axis.upper(),fontsize=14)
    plt.ylabel(y_axis.upper(),fontsize=14)

def pca_df(df):
    """
    I AM NOT USING THIS FUNCTION AT ALL ...
    """
    return pd.DataFrame(
        data = PCA(whiten=True).fit_transform(df),
        index = df.index,
        columns = ["pca"+str(i) for i in range(min(df.shape))]
    )

def twostep_filter(geno_list=None,pheno_list=None,column=None,verbose=False):
    """

    Merges a set of datasets, removing batch effects via the two-step SVD merging method.

    geno_list: list of dataframes containing gene-expression values (each item corresponds to a batch).
    pheno_list: associated list of dataframes containing clinical data (ordered as in geno_list).
    column: name of a column in pheno_list to be used to define two groups (usually healthy/disease).

    
    """
    # First filter batch by batch
    genos_1f = []
    for geno,pheno in zip(geno_list,pheno_list):
        g =onestep_filter(geno=geno,pheno=pheno,column=column,verbose=verbose)
        genos_1f.append(g)

    # merge batches
    geno_1f = pd.concat(genos_1f,join="inner")
    pheno = pd.concat(pheno_list,join="inner")

    return onestep_filter( geno = geno_1f , pheno = pheno , column = column , verbose = verbose)


