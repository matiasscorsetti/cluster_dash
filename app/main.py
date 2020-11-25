import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
import re
from Graphics import graphics
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

########## tatusa
import io
import base64
import matplotlib.pyplot as plt
#############


variables_dir = "data/variables/"
variables_df_name = "_variables.parquet.gzip"
process_dir =  "data/process/"
process_df_name = "_cluster_result.parquet.gzip"
periods = 30
result_col_name = "cluster"
metrics_dir = "metrics/clusters/"
metrics_plckle_name = "_metrics_cluster_result.pickle"
not_cluster_list = ["0", "1"]

path = "/dashboard/"
external_stylesheets = [dbc.themes.BOOTSTRAP]

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets, requests_pathname_prefix=path)

dash_app.layout = html.Div(children=[

    # load dataset
    html.H1(children='Clustering Dashboard', style={'color': 'gray', 'text-align': 'center'}),
    html.Br(),
    html.Div(
        dbc.Row([
            dbc.Col(html.H5(children='Select a date to load the analysis dataset', style={'color': 'gray', 'text-align': 'left'}), 
                    width={"size": "auto", "offset": 1},),
            dbc.Col(dcc.DatePickerSingle(
                    id='my-date-picker-single',
                    placeholder = 'Select date',
                    display_format="DD/MM/YYYY",
                    with_portal=True
                    ),
                    width="auto"),
            dbc.Col(html.Button(id='submit-button', n_clicks=0, children='Load Results'),
                    width="auto"),
            dbc.Col(dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=html.Div(id='container-button-timestamp')),
                    width="auto", lg=3),
            ]),
        ),
    html.Br(),  

    # tabs
    dbc.Tabs([

        # data describe
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(id='dataset-size', width={"size": "auto", "offset": 1}),
                dbc.Col(id='model-name', width={"size": "auto", "offset": 0.5}),
                ]),
            dbc.Row([
                dbc.Col(id='title-features', width={"size": "auto", "offset": 1}),
                dbc.Col(id='tab-features', width={"size": "auto", "offset": 0.5}),
                ]),
            html.Br(),
            dbc.Row(dbc.Col(id='dataset-table', width={"size": "auto", "offset": 1})),
            html.Br(),
            ],label="Data Describe"),

        # data drift
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button(id='drift-load-button', n_clicks=0, children='Load Drift'), width={"size": "auto", "offset": 1}),
                dbc.Col(dcc.Loading(
                            id="loading-2",
                            type="default",
                            children=html.Div(id='container-drift')), 
                        width={"size": "auto"}),
                ]),
            html.Br(),
            dbc.Row(dbc.Col(id='drift-average-predict', width={"size": "auto", "offset": 1})),
            dbc.Row(dbc.Col(id='drift-percentage-predict', width={"size": "auto", "offset": 1})),
            dbc.Row(dbc.Col(id='drift-totals-predict', width={"size": "auto", "offset": 1})),
            html.Br(),
            dbc.Row(dbc.Col(id='drift-plot-predict', width=12),),
            dbc.Row(dbc.Col(id='drift-plot-describe', width=12),),
            html.Br(),
            ],label="Data Drift"),

        # data results
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button(id='results-button', n_clicks=0, children='Load Results'), width={"size": "auto", "offset": 1}),
                dbc.Col(dcc.Loading(
                            id="loading-3",
                            type="default",
                            children=html.Div(id='container-results')), 
                        width={"size": "auto"})
                    ]),
            html.Br(),
            dbc.Row(dbc.Col(id='distribution-results-title', width={"size": "auto"}), justify="center"),
            html.Br(),
            dbc.Row(dbc.Col(id='distribution-data', width={"size": "auto", "offset": 1})),
            dbc.Row([
                dbc.Col(id='bar-distribution-results', width=6),
                dbc.Col(id='pie-distribution', width=6),
                ]),
            dbc.Row(dbc.Col(id='silhouette-results-title', width={"size": "auto"}), justify="center"),
            html.Br(),
            dbc.Row(dbc.Col(id='silhouette-data', width={"size": "auto", "offset": 1})),
            dbc.Row([
                dbc.Col(id='bar-silhouette-coefficient-class', width=6),
                dbc.Col(id='pie-silhouette-coefficient-class', width=6),
                ]),
            ], label="Results Metrics"),

        # features importance
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button(id='features-importance-button', n_clicks=0, children='Calculate'), width={"size": "auto", "offset": 1}),
                dbc.Col(dcc.Loading(
                            id="loading-4",
                            type="default",
                            children=html.Div(id='container-features-importance')), 
                        width={"size": "auto"})
                    ]),
            html.Br(),
            dbc.Row([
                dbc.Col(id='bar-features-importance', align="start", width=6),
                dbc.Col(id='heatmap-features-importance', align="end", width=6),
                ]),
            html.Br(),
            ], label="Features Importance"),

        # describe features
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='describe-column-dropdown', 
                                     options=[],
                                     ), width={"size": 3, "offset": 1}),
                dbc.Col(dcc.Loading(
                            id="loading-5",
                            type="default",
                            children=html.Div(id='container-describe-column')), 
                        width={"size": "auto"})
                    ]),
            html.Br(),
            dbc.Row([
                dbc.Col(id='histogram-feature', align="start", width=6),
                dbc.Col(id='boxplot-feature', align="end", width=6),
                ]),
            html.Br(),
            dbc.Row(dbc.Col(id='describe-feature-for-result-title', width={"size": "auto", "offset": 1})),
            dbc.Row(dbc.Col(id='describe-feature-for-result', width={"size": 10, "offset": 1})),
            html.Br(),
            html.Br(),
            dbc.Row(dbc.Col(id='describe-feature-general-title', width={"size": "auto", "offset": 1})),
            dbc.Row([
                dbc.Col(id='histogram-feature-general', width={"size": 8, "offset": 1}),
                dbc.Col(id='describe-feature-general', width={"size": 2}),
                ]),
            html.Br(),
            ], label="Cluster Decribe"),
        
        # warning
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(html.Button(id='cluster-evaluation-button', n_clicks=0, children='Calculate'), width={"size": "auto", "offset": 1}),
                dbc.Col(dcc.Loading(
                            id="loading-6",
                            type="default",
                            children=html.Div(id='container-cluster-evaluation')), 
                        width={"size": "auto"})
                    ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Img(id='elbow_fig'), width={"size": "auto"}),
                ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Img(id='silhouette_fig'), width={"size": "auto"}),
                ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Img(id='distance_fig'), width={"size": "auto"}),
                ]),
            html.Br(),
            ], label="Cluster Evaluation"),
        ])
    ])

# load dataset in date range
@dash_app.callback(
    [Output('container-button-timestamp', 'children'),
     Output('dataset-size', 'children'),
     Output('model-name', 'children'),
     Output('title-features', 'children'),
     Output('tab-features', 'children'),
     Output('dataset-table', 'children'),
     Output('describe-column-dropdown', 'options'),
     ],
    [Input('submit-button', 'n_clicks'),
     Input('my-date-picker-single', 'date'),
     ],
    )
def displayClick(submitbtn, date=None):

    global df
    global df_variables
    global columns_labels
    global metrics
    global model_name
    global features
    df = None
    df_variables = None
    columns_labels = []
    metrics = None
    model_name = None
    features = []
    dataset_table = None
    size = None
    str_model_name = None
    title_features = None
    tab_features = None

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'submit-button' in changed_id:
        if date:
            end_date = pd.to_datetime(date).strftime('%Y%m%d')
            start_date = (pd.to_datetime(date) + pd.DateOffset(days= - periods + 1)).strftime('%Y%m%d')

            variables_file_path = variables_dir + end_date + "_" + start_date + variables_df_name
            process_file_path = process_dir + end_date + "_" + start_date + process_df_name
            metrics_file_path = metrics_dir + end_date + "_" + start_date + metrics_plckle_name

            try:
                df_variables = pd.read_parquet(variables_file_path)
                df_process = pd.read_parquet(process_file_path)
                with open(metrics_file_path, 'rb') as handle:
                    metrics = pickle.load(handle)

                model_name = metrics.get("model_name")
                features = metrics.get("features")

                df = pd.concat([df_variables, df_process], axis=1)
                clusters = ~df[result_col_name].isin(not_cluster_list)
                df = df[clusters]
                df_variables = df_variables[clusters]
                df_process = df_process[clusters]
                msg = dcc.Markdown("**Dataset**: *start date* {start_date}, *end date* {end_date}".format(start_date=pd.to_datetime(start_date).strftime('%d/%m/%Y'), 
                                                                                 end_date=pd.to_datetime(end_date).strftime('%d/%m/%Y'),
                                                                                 ))
                size = dcc.Markdown("**Dataset size:** {} rows".format(df.shape[0]))
                str_model_name = dcc.Markdown("**Model name:** {}".format(model_name))
                title_features = dcc.Markdown("**Features:** ")
                tab_features = dcc.Markdown(" - ".join([str(i) for i in features]))
                df_describe = df.describe().round(4).reset_index()
                df_describe = df_describe.rename(columns={"index":"statistics"})
                dataset_table = dbc.Table.from_dataframe(df_describe, striped=True, bordered=True, hover=True)

                columns_labels = [{'label': str(col), 'value': col} for col in features]

            except FileNotFoundError:
                msg = "a file with the specified date range could not be found !! \
                       paths: {variables}, {process}, {metrics}".format(variables=variables_file_path, 
                                                                        process=process_file_path,
                                                                        metrics=metrics_file_path)

        else:
            msg = "No date range selected"

    else:
        msg = "Select a date to load the dataset"

    return msg, size, str_model_name, title_features, tab_features, dataset_table, columns_labels


# drift
@dash_app.callback(
    [Output('container-drift', 'children'),
     Output('drift-average-predict', 'children'),
     Output('drift-percentage-predict', 'children'),
     Output('drift-totals-predict', 'children'),
     Output('drift-plot-predict', 'children'),
     Output('drift-plot-describe', 'children'),
     ],
    [Input('drift-load-button', 'n_clicks'),
     ],
    )
def displayDrift(driftloadbutton):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    str_drift_average_predict = None
    str_drift_percentage_predict = None
    str_drift_totals_predict = None
    plot_predict_drift = None
    plot_describe_drift = None
    msg = "click to load drift results into input data"

    if 'drift-load-button' in changed_id:
        if all(v is not None for v in [metrics]):
            str_drift_average_predict = dcc.Markdown("""###### Drift averege predict: {}""".format(round(metrics.get("drift_average_predict"), 4)))
            str_drift_percentage_predict = dcc.Markdown("###### Drift percentage predict:\n {}".format({key: round(value, 4) for key, value in metrics.get("drift_percentage_predict").items()}))
            str_drift_totals_predict = dcc.Markdown("###### Drift totals predict:\n {}".format({key: int(value) for key, value in metrics.get("drift_totals_predict").items()}))
            plot_predict_drift = dcc.Graph(figure=metrics.get("drift_plot_predict").update_layout(autosize=True, width=1500, height=int(len(features)*250)))
            plot_describe_drift = dcc.Graph(figure=metrics.get("drift_plot_describe").update_layout(autosize=True, width=1500, height=int(len(features)*250)))
            msg = "drift results have been loaded"

        else:
            msg = "a dataset has not been loaded"

    return msg, str_drift_average_predict, str_drift_percentage_predict, str_drift_totals_predict, plot_predict_drift, plot_describe_drift


# results
@dash_app.callback([
    Output('container-results', 'children'),
    Output('distribution-results-title', 'children'),
    Output('distribution-data', 'children'),
    Output('bar-distribution-results', 'children'),
    Output('pie-distribution', 'children'),
    Output('silhouette-results-title', 'children'),
    Output('silhouette-data', 'children'),
    Output('bar-silhouette-coefficient-class', 'children'),
    Output('pie-silhouette-coefficient-class', 'children'),
    ],
    [Input('results-button', 'n_clicks'),
     ],
     )
def displayResultst(resultsbutton):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    msg = "click to load results"
    distribution_title = None
    distribution_data = None
    bar_distribution = None
    pie_distribution = None
    silhouette_title = None
    silhoutte_data = None
    bar_silhoutte_class = None
    pie_silhouette_class = None

    if 'results-button' in changed_id:
        if all(v is not None for v in [df, metrics]):
            distribution_title =  dcc.Markdown("## Class Distribution")
            distribution_data = df[result_col_name].value_counts(normalize=True).to_dict()
            bar_distribution = dcc.Graph(figure=graphics.bar_plot(df=df, result_col_name=result_col_name))
            pie_distribution = dcc.Graph(figure=graphics.pie_plot(df=df, result_col_name=result_col_name))
            silhouette_title = dcc.Markdown("## Silhouette Results")
            silhoutte_data = metrics.get("cluster_silhouette")
            bar_silhoutte_class = dcc.Graph(figure=graphics.bar_from_dict(dictionary=silhoutte_data))
            pie_silhouette_class = dcc.Graph(figure=graphics.pie_from_dict(dictionary=silhoutte_data))
            distribution_data = dcc.Markdown("**Values:** {}".format(str({key: round(value, 4) for key, value in distribution_data.items()})))
            silhoutte_data = dcc.Markdown("**Values:** {}".format(str({key: round(value, 4) for key, value in silhoutte_data.items()})))

        else:
            msg = "a dataset has not been loaded"

    return msg, distribution_title, distribution_data, bar_distribution, pie_distribution, silhouette_title, silhoutte_data, bar_silhoutte_class, pie_silhouette_class



# feature importance
@dash_app.callback(
    [Output('container-features-importance', 'children'),
     Output('heatmap-features-importance', 'children'),
     Output('bar-features-importance', 'children'),
     ],
    [Input('features-importance-button', 'n_clicks'),
     ],
    )
def displayFeatureImportance(featuresimportancebutton):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    heatmap_features_importance = None
    bar_features_importance = None
    msg = "click to calculate the importance of each feature"

    if 'features-importance-button' in changed_id:
        if all(v is not None for v in [df, df_variables,]):
            heatmap_features_importance = dcc.Graph(figure=graphics.heatmap_plot(df=df, result_col_name=result_col_name, features=features))
            bar_features_importance = dcc.Graph(figure=graphics.bar_importance_plot(df=df, result_col_name=result_col_name, features=features))
            msg = "importance of features calculated"

        else:
            msg = "a dataset has not been loaded"
    
    return msg, heatmap_features_importance, bar_features_importance


# describe column
@dash_app.callback(
    [Output('container-describe-column', 'children'),
     Output('histogram-feature', 'children'),
     Output('boxplot-feature', 'children'),
     Output('describe-feature-for-result-title', 'children'),
     Output('describe-feature-for-result', 'children'),
     Output('describe-feature-general-title', 'children'),
     Output('histogram-feature-general', 'children'),
     Output('describe-feature-general', 'children'),
     ],
    [Input('describe-column-dropdown', 'value'),
     ],
    )
def displayDescribeFeature(value):

    histogram_clusters = None
    boxplot_clusters = None
    describe_cluster_title = None
    describe_clusters = None
    describe_general_title = None
    histogram_general = None
    describe_general = None

    if all(v is not None for v in [df, df_variables]):
        histogram_clusters = dcc.Graph(figure=graphics.histogram_plot(df=df, result_col_name=result_col_name, col_name=value))
        boxplot_clusters = dcc.Graph(figure=graphics.box_plot(df=df, result_col_name=result_col_name, col_name=value))
        describe_cluster_title = dcc.Markdown("**Feature statistics for each class:** {}".format(value))
        describe_clusters_data = graphics.describe_by_result(df=df, result_col_name=result_col_name, col_name=value).round(4)
        describe_clusters = dbc.Table.from_dataframe(describe_clusters_data, striped=True, bordered=True, hover=True)
        describe_general_title = dcc.Markdown("**General feature statistics:** {}".format(value))
        histogram_general = dcc.Graph(figure=graphics.simple_histogram(df=df, col_name=value))
        describe_general_data = df[value].describe().to_frame().round(4).reset_index()
        describe_general_data = describe_general_data.rename(columns={"index":"statistics"})
        describe_general = dbc.Table.from_dataframe(describe_general_data, striped=True, bordered=True, hover=True)

    return value, histogram_clusters, boxplot_clusters, describe_cluster_title, describe_clusters, describe_general_title, histogram_general, describe_general


# warning
@dash_app.callback(
    [Output('container-cluster-evaluation', 'children'),
     Output('elbow_fig', 'src'),
     Output('silhouette_fig', 'src'),
     Output('distance_fig', 'src'),
     ],
    [Input('cluster-evaluation-button', 'n_clicks'),
     ],
    )
def displayResultsMetrics(resultsmetricsbutton):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    fig_elbow = None
    fig_silhoutte = None
    fig_distance = None
    msg = "click to calculate the resulting metrics"

    if 'cluster-evaluation-button' in changed_id:
        if all(v is not None for v in [df]):
            fig_elbow = graphics.elbow_yellowbrick(X=df, y=df[result_col_name], features=features)
            me =  fig_elbow.show(outpath="fig_elbow.png")
            fig_elbow = 'fig_elbow.png'
            fig_elbow = base64.b64encode(open(fig_elbow, 'rb').read())
            fig_elbow = 'data:image/png;base64,{}'.format(fig_elbow.decode())
            me = me.cla()

            fig_distance = graphics.distance_yellowbrick(X=df, y=df[result_col_name], features=features)
            me =  fig_distance.show(outpath="fig_distance.png")
            fig_distance = 'fig_distance.png'
            fig_distance = base64.b64encode(open(fig_distance, 'rb').read())
            fig_distance = 'data:image/png;base64,{}'.format(fig_distance.decode())
            me = me.cla()
            
            fig_silhoutte = graphics.silhoutte_yellowbrick(X=df, y=df[result_col_name], features=features)
            me =  fig_silhoutte.show(outpath="fig_silhoutte.png")
            fig_silhoutte = 'fig_silhoutte.png'
            fig_silhoutte = base64.b64encode(open(fig_silhoutte, 'rb').read())
            fig_silhoutte = 'data:image/png;base64,{}'.format(fig_silhoutte.decode())
            me = me.cla()

            msg = "the results metrics have been calculated"

        else:
            msg = "a dataset has not been loaded"
            
    return msg, fig_elbow, fig_silhoutte, fig_distance

app = FastAPI()

@app.get("/")
def read_root():
    return {"Cluster Dashboard": "Wyleex"}

app.mount(path, WSGIMiddleware(dash_app.server), name="dashboard")