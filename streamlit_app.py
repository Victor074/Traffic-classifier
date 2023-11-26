import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#from catboost import CatBoostClassifier
sns.set(style = "darkgrid" )
#Lectura de la Imagen
path = './Jupyter notebook/Traffic.csv'
data = pd.read_csv(path)
Logo = io.imread("./Pictures/traffic_pic.jpg")
st.set_page_config(page_title="Traffic classification", page_icon=":tada:", layout="wide")
le = LabelEncoder()
correlation_data = data.copy()
correlation_data['Time'] = le.fit_transform(correlation_data['Time'])
correlation_data['Traffic Situation'] = correlation_data['Traffic Situation'].replace({'low': 0,'normal': 1,'high': 2, 'heavy':3})
correlation_data['Day of the week'] = correlation_data['Day of the week'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday': 6,'Sunday':7})
correlation_matrix = correlation_data.corr()
correlation_matrix = np.trunc(1000 * correlation_matrix) / 1000
#sns.heatmap(correlation_matrix, annot = True)

# Header
with st.container():
    st.image(Logo, width=800)
    st.title("Traffic classifier")
    st.subheader(":blue[This project classifies incoming traffic with a logistic... and the classes are: 1/2/3/4]")

    st.markdown("**Traffic Data**")
    st.markdown(":blue[This **DataFrame** contains information of traffic on ... on a 15 minute interval, the way the did this .....]")
    st.dataframe(data)
    image_column, text_column = st.columns((1,2))
    with text_column:
        st.subheader("Explanation")
        st.write(
            """
            Ok so this is the quick explanation to my project bla bla blabalaljkla
            """
        )
    with image_column:
        st.image(Logo, width=200)

#st.sidebar.image(Logo, width = 200)
st.sidebar.markdown("## CONFIGURACIÓN")
vars_feature = ['Time','Date','Day of the week','CarCount','BikeCount','BusCount','TruckCount','Total']
default_hist = vars_feature.index('CarCount')
histo_selected = st.sidebar.selectbox('Histogram variable:', vars_feature, index = default_hist)

vars_algorithm = ['Logistic Regression','Naive Vayes','KNN','Decision Trees','Random Forest','CatBoost']
default_pers = vars_algorithm.index('Logistic Regression')
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

train_x, test_x = train_test_split(data,test_size=0.2,random_state=123)
train_x.shape, test_x.shape
train_y = train_x["Traffic Situation"].to_numpy()
test_y = test_x["Traffic Situation"].to_numpy()
train_x = train_x.drop(columns=["Traffic Situation"],axis=1).to_numpy()
test_x = test_x.drop(columns=["Traffic Situation"],axis=1).to_numpy()
pipeline_traffic= Pipeline([
    ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
    ("onehot",OneHotEncoder(handle_unknown='ignore')),
    ("std_scaler", StandardScaler(with_mean=False)),
    ('',''),
])
models = [
    RandomForestClassifier(random_state=123),
    #GradientBoostingClassifier(random_state=123),
    LogisticRegression(random_state=123),
    DecisionTreeClassifier(random_state=123),
    KNeighborsClassifier(),
    #XGBClassifier(),
    #AdaBoostClassifier(random_state=123),
    #CatBoostClassifier(verbose=False)
]

rf_param = {
    'max_features': [1, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [False],
    'n_estimators': [100, 300],
    'criterion': ['gini']
}
gradient_param = {
    'loss': ['log_loss'],
    'n_estimators': [100, 200, 300],

}
logreg_param = {
    'C': np.linspace(-3, 3, 7),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [100, 300]
}
dt_param = {
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': np.arange(1, 100, 20),
    "min_samples_split": range(10, 500, 20)

}
knn_param = {
    'n_neighbors': np.arange(1, 20, 2),
    'metric': ['euclidean', 'manhattan'],
    'n_jobs': np.arange(1, 100, 20),

}
classifier_param = [
    rf_param,
    gradient_param,
    logreg_param,
    dt_param,
    knn_param,
]
results = []
clf_best_score = []
clf_best_estimators = []
for i in range(len(models)):
    clf = GridSearchCV(models[i], classifier_param[i], scoring='accuracy', verbose=1,
                       cv=10, n_jobs=-1)
    clf.fit(train_x, train_y)
    clf_best_score.append(clf.best_score_)
    clf_best_estimators.append(clf.best_estimator_)
    results.append({"Algorithm": models[i],"Best Estimators": clf_best_estimators[i], "Accuracy": clf_best_score[i]})
results = pd.DataFrame(results)
results.sort_values(by='Accuracy', ascending=True)
with st.container():
    st.subheader(":blue[Results]")
    fig4, ax4 = plt.subplots()
    sns.heatmap(results, annot = True)
    st.pyplot(fig4)
