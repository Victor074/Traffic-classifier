import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from sklearn.preprocessing import LabelEncoder

sns.set(style = "darkgrid" )
#Lectura de la Imagen
path = './Jupyter notebook/Traffic.csv'
path_results = './models_compare.csv'
path_results_testing = './models_comparison_testing.csv'
data = pd.read_csv(path)
all_results = pd.read_csv(path_results)
results_testing = pd.read_csv(path_results_testing)
Logo = io.imread("./Pictures/traffic_pic.jpg")
tree_pic = io.imread("./Pictures/tree.png")
tree_pic2 = io.imread("./Pictures/descarga.png")
st.set_page_config(page_title="Traffic classification", page_icon=":tada:", layout="wide")
le = LabelEncoder()
correlation_data = data.copy()
correlation_data['Time'] = le.fit_transform(correlation_data['Time'])
correlation_data['Traffic Situation'] = correlation_data['Traffic Situation'].replace({'low': 0,'normal': 1,'high': 2, 'heavy':3})
correlation_data['Day of the week'] = correlation_data['Day of the week'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday': 6,'Sunday':7})
correlation_matrix = correlation_data.corr()
correlation_matrix = np.trunc(1000 * correlation_matrix) / 1000

names = pd.DataFrame({
    'Classifier Algorithm':[
                    'CatBoost',
                    'RandomForestClassifier',
                    'LogisticRegression',
                    'GradientBoostingClassifier',
                    'DecisionTreeClassifier',
                    'KNeighborsClassifier',
                 ]
})
all_results['Classifier Algorithm'] = names

# Header
with st.container():
    st.image(Logo, width=800)
    st.title("Traffic classifier")
    st.markdown(":blue[This project classifies incoming traffic on 4 categories, [low, normal, heavy, high]]")
    st.subheader("Introduction")
    st.write(
            """
            My project compares 6 different Algorithms for classification, for selecting the right hyper parameters the Algorithms go through GridSearch
        , then with the best hyper parameters for each algorithm i will rank the scores for the testing phase.
        Previous to training the models i applied some label encoders for the categorical features and got rid off features of less importance such as time
            """
        )
    st.markdown("**Traffic Data**")
    st.markdown(":blue[This **DataSet** is updated every 15 minutes, providing a comprehensive view of traffic patterns over the course of one month. Additionally, the dataset includes a column indicating the traffic situation categorized into four classes: 1-Heavy, 2-High, 3-Normal, and 4-Low. This information can help assess the severity of congestion and monitor traffic conditions at different times and days of the week.]")
    st.dataframe(data)

st.sidebar.markdown("## PARAMETERS")
vars_feature = ['Time','Date','Day of the week','CarCount','BikeCount','BusCount','TruckCount','Total']
default_hist = vars_feature.index('CarCount')
histo_selected = st.sidebar.selectbox('Histogram variable:', vars_feature, index = default_hist)

vars_algorithm = ['LogisticRegression','RandomForestClassifier','CatBoost','GradientBoostingClassifier','DecisionTreeClassifier', 'KNeighborsClassifier']
default_pers = vars_algorithm.index('LogisticRegression')
algorithm_selected = st.sidebar.selectbox('Algorithm for classification:', vars_algorithm, index = default_pers)
st.write("[Github repo](https://github.com/Victor074/Traffic-classifier)")

with st.container():
    st.subheader(":blue[Correlation Matrix]")
    fig3, ax3 = plt.subplots()
    sns.heatmap(correlation_matrix, annot = True)
    st.pyplot(fig3)

#histograms

with st.container():
    st.subheader(":blue[Histogram]")
    fig1, ax1 = plt.subplots()
    sns.histplot(data, x=histo_selected, hue='Traffic Situation', kde=True)
    ax1.set_title(f'Traffic Situation vs {histo_selected}')
    ax1.set_xlabel(histo_selected)
    ax1.set_ylabel('Traffic Situation')
    st.pyplot(fig1)

with st.container():
    st.subheader(":blue[Imbalanced dataset]")
    fig7, ax7 = plt.subplots()
    plt.close()
    
    plt.figure(figsize=(10, 10))
    cvalues = data["Traffic Situation"].value_counts()
    colors = sns.color_palette('pastel')[0:len(cvalues)]
    plt.pie(cvalues, labels = cvalues.index, colors = colors, autopct='%.0f%%')
    st.pyplot(plt)
    


with st.container():
    st.subheader(":blue[Results for training]")
    st.dataframe(all_results)
    fig2, ax2 = plt.subplots()
    sns.set(style="darkgrid")
    sns.scatterplot(data = all_results, x='Classifier Algorithm', y='Accuracy score')
    plt.xticks(rotation='vertical')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    st.pyplot(fig2)
    
hyper = {'LogisticRegression':"""
C=268.2695795279727, penalty=l1, solver=saga""",
'RandomForestClassifier':"""
bootstrap=False, max_depth=10, max_features='log2', min_samples_split=5, n_estimators=500
""",
'CatBoost':"""
iterations=200, learning_rate=0.1
""",
'GradientBoostingClassifier':"""
learning_rate=0.5, max_depth=4
""",
'DecisionTreeClassifier':"""
max_features='auto', min_samples_split=5
""",
'KNeighborsClassifier':"""
metric='manhattan'
"""

}
hyper_explanation = {'LogisticRegression':"""
C: This is the inverse of regularization strength.

penalty: This specifies the norm used in the penalization.

solver: This is the algorithm to use in the optimization problem.

                     """,
'RandomForestClassifier':"""
bootstrap: This parameter is used to control whether bootstrap samples are used when building trees. If `False`, the whole dataset is used to build each tree.

max_depth: This parameter controls the maximum depth of the tree.

max_features:  This parameter determines the number of features to consider when looking for the best split.

min_samples_split: This parameter determines the minimum number of samples required to split an internal node.

n_estimators: This parameter specifies the number of trees in the forest.
""",
'CatBoost':"""

iterations: This parameter specifies the maximum number of trees that can be built when solving machine learning problems.

learning_rate: This parameter is used for reducing the gradient step.
""",
'GradientBoostingClassifier':"""

learning_rate: This parameter shrinks the contribution of each tree by the learning rate.

max_depth: This parameter controls the maximum depth of the individual regression estimators.
""",
'DecisionTreeClassifier':"""

max_features: This parameter determines the number of features to consider when looking for the best split. If ‘auto’, then max_features=sqrt(n_features). In other words, it will consider the square root of the total number of features at each split1.

min_samples_split: This parameter determines the minimum number of samples required to split an internal node.
""",
'KNeighborsClassifier':"""

metric: This parameter determines the distance metric used for the tree.
"""

}

with st.container():
    st.subheader(":blue[Results for testing]")
    st.dataframe(results_testing)
    fig4, ax4 = plt.subplots()
    sns.set(style="darkgrid")
    sns.scatterplot(data = results_testing, x='Classifier Algorithm', y='Accuracy score')
    plt.xticks(rotation='vertical')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    st.pyplot(fig4)

with st.container():
    st.subheader(f":blue[Results for {algorithm_selected}]")
    text_column1, text_column2 = st.columns((1,2))
    test_score = results_testing[results_testing['Classifier Algorithm']==algorithm_selected]["Accuracy score"]
    training_score =all_results[all_results['Classifier Algorithm']==algorithm_selected]["Accuracy score"]
    with text_column1:
        st.subheader(f"Training score: {training_score.values[0]}")
        st.subheader(f"Testing score: {test_score.values[0]}")
        
        st.write(f"Best found hyper parameters: {str(hyper[algorithm_selected])}")
    with text_column2:
        st.write(f"{str(hyper_explanation[algorithm_selected])}")
with st.container():
    st.subheader(":blue[WINNER is Random Forest]")
    
    st.image(tree_pic2, width=1500)
