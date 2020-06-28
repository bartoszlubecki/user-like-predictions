import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from scipy import sparse, io
from scipy.stats import pearsonr


def load_user_like_matrix():
    """
    This function loads a sparse User-Like matrix created in R according 
    to instructions provided by https://www.michalkosinski.com/data-mining-tutorial
    It then converts it to a Pandas dataframe and returns it as a result.
    Args:
        none
    Returns:
        (pandas.DataFrame) dataframe with 
    """
    sparsematrix = io.mmread('data/sparsematrix.txt')
    m_dense = sparsematrix.toarray()
    var_names = np.genfromtxt('data/rownames.txt', dtype=str,
                              delimiter='\n', encoding="utf8")
    col_names = np.genfromtxt('data/colnames.txt', dtype=str,
                              delimiter='\n', encoding="utf8")
    df = pd.DataFrame(m_dense, columns=col_names, index=var_names)
    
    return df


def plot_roc_curve(y_true, scores):
    """
    This function plots a ROC curve.
    Args:
        y_true: True y values.
        scores: Predicted probabilities to compare with true values.
    Returns:
        none
    """    
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=scores)

    plt.plot(fpr, tpr)
    plt.scatter(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], linestyle='--')
    auc = roc_auc_score(y_true=y_true,
                        y_score=scores)
    plt.title('AUC: {}'.format(auc))
    plt.show()
    
    
def pearson_cor(y_true, y_pred, **kwargs):
    return pearsonr(y_true, y_pred)[0]


def test_linreg_targets(X: pd.DataFrame, 
                        users: pd.DataFrame,
                        label: str,
                        model: object = LinearRegression(n_jobs=-1)):  
    """
    This function takes X data and a model, trains it and evaluates scores by 10-fold cross validation. 
    Custom pearson_scorer is used to match Authors' results. Scores are evaluated for continuous values
    specified in linreg_targets.
    Args:
        X: (pd.DataFrame)
        users: (pd.DataFrame)
        label: (str)
        model: (object)
    Returns:
        (pandas.DataFrame) dataframe with 
    """

    results = {}
    results['name'] = label
    pearson_scorer = make_scorer(pearson_cor)
    linreg_targets = ['age', 'ope', 'con', 'ext', 'agr', 'neu']
    
    for target in linreg_targets:

        y = users[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)

        model = model
        model.fit(X_train, y_train)

        cv_score = round(cross_val_score(model, X, y, scoring=pearson_scorer, cv=10, n_jobs=-1).mean(), 2)
        
        # Further metrics for each model may be implemented here in the future
        # y_pred = model.predict(X_test)
        # chosen_metric(y_test, y_pred)        
        
        results[target] = cv_score
        results = pd.DataFrame(results, index=[0])
        
    return results


def test_logreg_target(X: pd.DataFrame,
                       y: pd.Series,
                       label: str,
                       model: object = LogisticRegression(random_state=42, n_jobs=-1)):  
    """
    This function takes X data and a model, trains it and evaluates scores by 10-fold cross validation. 
    ROC_AUC scoring is used to match Authors' results. 
    Args:
        X: (pd.DataFrame)
        users: (pd.DataFrame)
        label: (str)
        model: (object)
    Returns:
        (pandas.DataFrame) dataframe with 
    """
    
    result = {}
    result['name'] = label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    model = model
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_pred_proba)

    result[y.name] = round(cross_val_score(
        model, X, y, scoring='roc_auc', cv=10, n_jobs=-1).mean(), 2)
    result = pd.DataFrame(result, index=[0])

    return result


def plot_results(title: str, 
                 labels: pd.Series, 
                 scores: pd.Series, 
                 axis_label: str):
    """
    This function draws barplots to visualize results. 
    Args:
        title: (str) Title of the plot
        labels: (pd.Series) Bar labels
        scores: (pd.Series) Bar values
        axis_label: (str) Y axis label
    Returns:
        none
    """
    # Plot Bars
    plt.figure(figsize=(8,4), dpi= 80)
    plt.bar(labels, scores, color='tab:blue', width=.5)
    for i, val in enumerate(scores.values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

    # Decoration
    plt.gca().set_xticklabels(labels, rotation=60, horizontalalignment= 'right')
    plt.title(title, fontsize=22)
    plt.ylabel(axis_label)
    plt.ylim(0, 1.1)
    plt.show()