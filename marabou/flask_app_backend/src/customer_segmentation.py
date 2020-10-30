##################################################################
#                      Import packages                           #
##################################################################
import pandas as pd
import numpy as np
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pycountry_convert import country_name_to_country_alpha3
from yellowbrick.cluster.elbow import kelbow_visualizer


##################################################################
#                      Functions Area                            #
##################################################################

# ***************************************************
# ******            normalizer Function        ******
# ***************************************************
def normalizer(df, norm_type='StandardScaler'):
    if norm_type == 'StandardScaler':
        normalizer = StandardScaler()
    else:
        normalizer = MinMaxScaler()
    # fit and transform the data table df
    normalizer.fit(df)
    df_normalized = pd.DataFrame(normalizer.transform(df))
    df_normalized.columns = df.columns
    df_normalized.index = df.index
    return df_normalized

##################################################################
#                      Main Area                                 #
##################################################################

# ***************************************************
# ******            Load Data                  ******
# ***************************************************
# -------------------------------
# 1. Load data
data_url= 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
data_table = pd.read_excel(data_url)

# -------------------------------
# 2. Sample of the data table
print("data sample {0}".format(data_table.shape))
print(data_table.head())

# -------------------------------
# 3. Summary of the numeric data_table's columns
print("dataset description")
print(data_table.describe())

# ***************************************************
# ******  Data Cleaning & Feature Engineering  ******
# ***************************************************
# -------------------------------
# 1. Convert type
data_table['InvoiceDate'] = data_table['InvoiceDate'].dt.date

# -------------------------------
# 2. Create new feature: total_price
data_table['total_price'] = data_table['Quantity'] * data_table['UnitPrice']

# -------------------------------
# 3. Calculate RFM parameters
snapshot_date = max(data_table.InvoiceDate) + datetime.timedelta(days=1)
# 3.A. Aggregate data by each customer
customers_data = data_table.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'total_price': 'sum'})
# 3.B. Rename columns
customers_data.rename(columns={'InvoiceDate': 'recency',
                               'InvoiceNo': 'frequency',
                               'total_price': 'monetary'}, inplace=True)
print("customer data shape {0}".format(customers_data.shape))

# ***************************************************
# ******                Visualization          ******
# ***************************************************
count_country = pd.DataFrame(data_table.Country.value_counts())
count_country = count_country.reset_index()
count_country.columns = ['country', 'count']
count_country['country'].replace({'EIRE': 'Ireland', 'Channel Islands': 'United Kingdom', 'RSA': 'South Africa'},
                                 inplace=True)
count_country = count_country.loc[~count_country.country.isin(['Unspecified', 'European Community']), :]
count_country['country_alpha_3'] = count_country.country.apply(lambda x: country_name_to_country_alpha3(x))
print("count country dataset {0}".format(count_country.shape))

#fig = px.choropleth(count_country, locations='country_alpha_3', color='count',
#                    hover_name='country', color_continuous_scale=px.colors.sequential.Plasma)
#plot(fig)


# ***************************************************
# ******  Data Preparation For Modeling        ******
# ***************************************************
"""
# -------------------------------
# 1. Skewness check
fig = make_subplots(rows=1, cols=3)
fig.add_trace(go.Box(y=customers_data.recency, name='recency'), row=1, col=1)
fig.add_trace(go.Box(y=customers_data.frequency, name='frequency'), row=1, col=2)
fig.add_trace(go.Box(y=customers_data.monetary, name='monetary'), row=1, col=3)
fig.update_layout(yaxis_title='Value')
plot(fig)

customers_data_cleaned = pd.DataFrame()
customers_data_cleaned['recency'] = stats.boxcox(customers_data['recency'])[0]
customers_data_cleaned['frequency'] = stats.boxcox(customers_data['frequency'])[0]
customers_data_cleaned['monetary'] = pd.Series(np.cbrt(customers_data['monetary'])).values


fig = make_subplots(rows=1, cols=3)
fig.add_trace(go.Box(y=customers_data_cleaned.recency, name='recency'), row=1, col=1)
fig.add_trace(go.Box(y=customers_data_cleaned.frequency, name='frequency'), row=1, col=2)
fig.add_trace(go.Box(y=customers_data_cleaned.monetary, name='monetary'), row=1, col=3)
fig.update_layout(yaxis_title='Value')
plot(fig)

# -------------------------------
# 2. Normalization
customers_data_norm = normalizer(df=customers_data_cleaned, norm_type='StandardScaler')

# ***************************************************
# ******            Modelling                  ******
# ***************************************************

# -------------------------------
# 1. Extract optimize number of clusters based on the Elbow Method
kelbow_visualizer(KMeans(random_state=4), customers_data_norm, k=(2,10))

# -------------------------------
# 2. Apply clustering model
clustering_model = KMeans(n_clusters=5, random_state=42)
clustering_model.fit(customers_data_norm)

customers_data_norm['cluster'] = clustering_model.labels_
customers_data_norm['cluster_str'] = 'cluster' + customers_data_norm['cluster'].astype(str)

customers_data['cluster'] = clustering_model.labels_
customers_data['cluster_str'] = 'cluster' + customers_data['cluster'].astype(str)

# ***************************************************
# ******    Post-processing & Visualization    ******
# ***************************************************
# -------------------------------
# 1. Count per class/ rate of class
cluster_count = customers_data_norm.cluster.value_counts()
cluster_count = cluster_count.sort_index()

cluster_rate = cluster_count.sort_index() / customers_data_norm.shape[0]
cluster_rate = cluster_rate.sort_index()

fig_bar_plot = make_subplots(rows=1, cols=2)
fig_bar_plot.add_trace(
    go.Bar(name='clusters_count', x=cluster_count.index, y=cluster_count),
    row=1, col=1
)
fig_bar_plot.add_trace(
    go.Bar(name='cluster_rate', x=cluster_rate.index, y=cluster_rate),
    row=1, col=2
)
plot(fig_bar_plot)

# -------------------------------
# 2. Scatter plot of clusters and features, 2D and 3D
# 2.A. 2D scatter plot
fig_2D = go.Figure(data=go.Scatter(
    x=customers_data_norm.recency,
    y=customers_data_norm.monetary,
    mode='markers',
    marker=dict(color=customers_data_norm.cluster, opacity=0.7, size=10)))
fig_2D.update_layout(
    title="2D Scatter plot of the Classes ",
    xaxis_title="recency",
    yaxis_title="monetary"

)

plot(fig_2D)

# 2.B. 3D scatter plot
fig_3D = go.Figure(data=[go.Scatter3d(
    x=customers_data_norm.frequency,
    y=customers_data_norm.recency,
    z=customers_data_norm.monetary,
    mode='markers',
    marker=dict(
        size=12,
        color=customers_data_norm.cluster,
        colorscale='Jet',
        opacity=0.6
    )
)])
fig_3D.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                     scene=dict(xaxis_title='norm frequency',
                                yaxis_title='norm recency',
                                zaxis_title='norm monetary'))
plot(fig_3D)

# -------------------------------
# 1. Calculating average of each feature of the classes in normalized data and original data
# 1.A. Calculation average from normalized data
features_avg_per_class_norm = customers_data_norm.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'}).round(2)
# 1.B. Calculation average from original data
features_avg_per_class_orig = customers_data.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'}).round(2)

# 1.C. Plot features_avg_per_class_norm
fig_bar_norm = go.Figure(data=[
    go.Bar(name='recency', x=features_avg_per_class_norm.index, y=features_avg_per_class_norm.recency),
    go.Bar(name='frequency', x=features_avg_per_class_norm.index, y=features_avg_per_class_norm.frequency),
    go.Bar(name='monetary', x=features_avg_per_class_norm.index, y=features_avg_per_class_norm.monetary)
])
fig_bar_norm.update_layout(barmode='group')
plot(fig_bar_norm)

# 1.D. Plot features_avg_per_class_orig
fig_bar_orig = go.Figure(data=[
    go.Bar(name='recency', x=features_avg_per_class_orig.index, y=features_avg_per_class_orig.recency),
    go.Bar(name='frequency', x=features_avg_per_class_orig.index, y=features_avg_per_class_orig.frequency),
    go.Bar(name='monetary', x=features_avg_per_class_orig.index, y=features_avg_per_class_orig.monetary)
])
fig_bar_orig.update_layout(barmode='group')
plot(fig_bar_orig)

# -------------------------------
# 2. Calculating average of each feature of the classes in normalized data and original data
# 2.A. Melting data
customers_data_norm_melt = pd.melt(customers_data_norm.reset_index(),
                                   id_vars=['index', 'cluster'],
                                   value_vars=['recency', 'frequency', 'monetary'],
                                   var_name='feature',
                                   value_name='value')
# 2.B. Visualize melted data
sns.lineplot('feature', 'value', hue='cluster', data=customers_data_norm_melt)
"""