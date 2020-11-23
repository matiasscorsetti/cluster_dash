import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from yellowbrick.style import set_palette
from yellowbrick.cluster.silhouette import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance
from yellowbrick.features import RadViz, ParallelCoordinates, Rank1D, Rank2D, PCA
from yellowbrick.target import ClassBalance, FeatureCorrelation
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clusters(X,
             y,
             model,
             features,
             ):
    
    X = pd.DataFrame(X, columns=features)
    y = pd.Series(y)
    
    visualizer = ClassBalance()
    visualizer.fit(y)
    visualizer.show()
    
    visualizer = FeatureCorrelation()
    visualizer.fit(X, y)
    visualizer.show()
    
    visualizer = Rank2D(algorithm='pearson')
    visualizer.fit(X, y)
    visualizer.transform(X)
    visualizer.show()
    
    visualizer = Rank1D(algorithm='shapiro')
    visualizer.fit(X, y)
    visualizer.transform(X)
    visualizer.show()
    
    visualizer = PCA(
                    scale=False, projection=2, proj_features=True,
                )
    visualizer.fit_transform(X, y)
    visualizer.show()
    
    visualizer = PCA(
                    scale=False, projection=3, proj_features=True,
                )
    visualizer.fit_transform(X, y)
    visualizer.show()
    
    visualizer = ParallelCoordinates(
                               sample=0.02, shuffle=True,
                                )
    visualizer.fit_transform(X, y)
    visualizer.show()
    
    print('standardize features by removing the mean and scaling to unit variance')
    visualizer = ParallelCoordinates(
                               sample=0.02, shuffle=True, normalize='standard',
                                )
    visualizer.fit_transform(X, y)
    visualizer.show()
    
    visualizer = RadViz(alphafloat=0.1)
    visualizer.fit(X, y.values)
    visualizer.show()
    
    visualizer = SilhouetteVisualizer(model=model,
                                  is_fitted=True
                                 )
    visualizer.fit(X)
    visualizer.show()
    
    print('mean silhouette score:', visualizer.silhouette_score_)
    
    visualizer = InterclusterDistance(model=model,
                                  is_fitted=True,
                                 )
    visualizer.fit(X)
    visualizer.show()


def pairgrid(X,
             y,
            ):
    
    data = X
    data['cluster'] = y
    
    g = sns.PairGrid(data=data, hue='cluster', hue_kws={'alpha':  [0.03 for cluster in data['cluster'].unique()]})
    g = g.map_diag(plt.hist, alpha = 0.8)
    g = g.map_offdiag(plt.scatter)
    g = g.add_legend()
    
    return g


def var_by_result(df, result_col_name, col_name=None):
    
    if col_name:
        pivot = pd.pivot_table(data=df,
                               index=[col_name, result_col_name],
                               aggfunc="size")
        
    else:
        pivot = pd.pivot_table(data=df,
                               index=result_col_name,
                               aggfunc="size")
    
    pivot = pivot.reset_index().rename(columns={0:"size"}).sort_values(result_col_name)
    
    return pivot


def stratify_sample(df, result_col_name,  columns_list=None, test_size=0.05):
    
    if columns_list:
        columns = [result_col_name] + columns_list
        
    else:
        columns = [result_col_name]
        
    X_train, X_test = train_test_split(df[columns],
                                        stratify=df[result_col_name], 
                                        test_size=test_size)
    return pd.DataFrame(X_test, columns=columns)


def box_plot(df, result_col_name, col_name):

    color_discrete_map = {value: color for value, color in zip(sorted(df[result_col_name].unique()), px.colors.qualitative.Plotly)}
    sample = stratify_sample(df=df, result_col_name=result_col_name, columns_list=[col_name]).sort_values(result_col_name)
    fig = px.box(sample,
             x=col_name,
             y=result_col_name,
             color=result_col_name, 
             color_discrete_map=color_discrete_map, 
             points="suspectedoutliers",
             title="Box plott satistics of the variable {} for each class".format(col_name))
    fig['layout']['yaxis']['autorange'] = "reversed"

    return fig


def bar_plot(df, result_col_name):

    color_discrete_map = {value: color for value, color in zip(sorted(df[result_col_name].unique()), px.colors.qualitative.Plotly)}
    pivot_result = var_by_result(df=df, result_col_name=result_col_name)
    fig = px.bar(pivot_result, x=result_col_name, y="size", color=result_col_name, color_discrete_map=color_discrete_map, 
                 title="Bar graph of the distribution of each class")
    fig.update_layout(xaxis={'categoryorder':'total descending'})

    return fig


def pie_plot(df, result_col_name):

    color_discrete_map = {value: color for value, color in zip(sorted(df[result_col_name].unique()), px.colors.qualitative.Plotly)}
    pivot_result = var_by_result(df=df, result_col_name=result_col_name)
    fig = go.Figure(data=[go.Pie(labels=pivot_result[result_col_name], values=pivot_result["size"],
                                 marker={"colors": list(color_discrete_map.values())}, 
                                  sort=False)])
    fig.update_layout(title="Pie chart of the percentage  of each class")

    return fig


def histogram_plot(df, result_col_name, col_name):

    color_discrete_map = {value: color for value, color in zip(sorted(df[result_col_name].unique()), px.colors.qualitative.Plotly)}
    pivot_result = var_by_result(df=df, result_col_name=result_col_name, col_name=col_name)
    fig = px.histogram(pivot_result, x=col_name, y="size", color=result_col_name, facet_row=result_col_name, histnorm="percent",
                   color_discrete_map=color_discrete_map,  title="Histogram of the istribution of the variable {} for each class".format(col_name))
    
    return fig


def feature_importance_for_each_class(X, y):
    
    df = pd.DataFrame(X)
    features_cols = df.columns
    df['result'] = y
    df_features_importance = pd.DataFrame(index=features_cols)
    
    for class_ in df['result'].unique():
               
        X = df[features_cols].copy()
        y = pd.Series(np.where(df['result']==class_, 1, 0))
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = LogisticRegression()
        model.fit(X, y)
        df_features_importance.loc[features_cols, class_] = model.coef_
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_features_importance.loc[:, class_] = scaler.fit_transform(df_features_importance[[class_]])
    
    df_features_importance.index = features_cols
    df_features_importance = df_features_importance.T
    df_features_importance.index.name = 'class'
    df_features_importance = df_features_importance.sort_index()
    
    return df_features_importance


def heatmap_plot(df, result_col_name, features):

    sample_df = stratify_sample(df=df, result_col_name=result_col_name, columns_list=features)
    result_1 = feature_importance_for_each_class(sample_df[features], sample_df[result_col_name])
    sample_df = stratify_sample(df=df, result_col_name=result_col_name, columns_list=features)
    result_2 = feature_importance_for_each_class(sample_df[features], sample_df[result_col_name])
    sample_df = stratify_sample(df=df, result_col_name=result_col_name, columns_list=features)
    result_3 = feature_importance_for_each_class(sample_df[features], sample_df[result_col_name])
    result = (result_1 + result_2 + result_3) / 3

    fig = px.imshow(result, color_continuous_scale=['#FF9900', '#f7e0bc', '#f7e0bc', '#FCFCFC', '#ceeaf2', '#ceeaf2', '#0099C6'], 
                    title="Heat map with the importance of characteristics by class")
    fig.update_yaxes(type='category')
    fig.update_xaxes(type='category')
    
    return fig


def bar_importance_plot(df, result_col_name, features):

    color_discrete_map = {value: color for value, color in zip(sorted(df[result_col_name].unique()), px.colors.qualitative.Plotly)}
    sample_df = stratify_sample(df=df, result_col_name=result_col_name, columns_list=features)
    result = feature_importance_for_each_class(sample_df[features], sample_df[result_col_name])
    result = result.stack().reset_index().rename(columns={"level_1":"features", 0:"importance"})
    fig = px.bar(result,
             y="features", x="importance", color="class", 
             title="Bar graph with the importance of characteristics by class",
             color_discrete_map=color_discrete_map)
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    
    return fig


def describe_by_result(df, result_col_name, col_name):
    
    pivot = pd.pivot_table(data=df,
                           index=result_col_name,
                           values=col_name,
                           aggfunc={col_name:["mean", "std", "max", "min",
                                              lambda x: np.percentile(x, 25),
                                              lambda x: np.percentile(x, 50),
                                              lambda x: np.percentile(x, 75),
                                              "count"]
                                    })
    pivot = pivot.rename(columns={"<lambda_0>": "q1",
                                  "<lambda_1>": "median",
                                  "<lambda_2>": "q3",
                                  })
    pivot = pivot[["mean", "std", "max", "min", "q1", "median", "q3", "count"]]
    pivot = pivot.reset_index()
    
    return pivot

def simple_histogram(df, col_name):

    fig = px.histogram(df.sample(frac=0.03), x=col_name, color_discrete_sequence=["rgb(102,102,102)"], marginal="box",
                      title="Distribution general of the variable {}".format(col_name))

    return fig

def bar_from_dict(dictionary):

    dictionary = {str(key): value for key, value in sorted(dictionary.items())}
    marker_color = [color for keys, color in zip(dictionary.keys(), px.colors.qualitative.Plotly)]
    fig = go.Figure(data=go.Bar(x=list(dictionary.keys()), y=list(dictionary.values()), marker_color=marker_color))
    fig.update_layout(title="Bar graph of the silhouette coefficient of each class",
                      xaxis={"title":"Class", "type": "category"},
                      yaxis={"title":"silhouette coef."})

    return fig

def pie_from_dict(dictionary):

    dictionary = {str(key): value for key, value in sorted(dictionary.items())}
    marker_color = [color for keys, color in zip(dictionary.keys(), px.colors.qualitative.Plotly)]
    fig = go.Figure(data=go.Pie(labels=list(dictionary.keys()), values=list(dictionary.values()),
                                marker={"colors": marker_color}, sort=False))
    fig.update_layout(title="Pie graph of the silhouette coefficient of each class")

    return fig