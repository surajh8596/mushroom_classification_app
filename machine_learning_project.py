import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
sns.set_style('dark')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

#Background Image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://images8.alphacoders.com/959/959012.jpg");
background-size: cover;
background-position: top center;
background-repeat: no-repeat;
background-attachment: local;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

#script to print text on webapp
def main():
    st.header("Mushroom Classification Web App")
    st.sidebar.title("Content")
    st.sidebar.markdown("Mushroom Classification ML Model")
if __name__=='__main__':
    main()

#load dataset
@st.cache_data(persist=True)
def load():
    data=pd.read_csv("data/mushrooms.csv")
    label=LabelEncoder()
    for col in data.columns:
        data[col]=label.fit_transform(data[col])
    return data
df=load()

#display data
if st.sidebar.checkbox("Disaply Data", False):
    st.subheader("Show Mushroom Dataset")
    st.write(df)

#Train test split
@st.cache_data(persist=True)
def split(df):
    x=df.drop('class', axis=1)
    y=df['class']
    x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)
    return x_train,x_test,y_train, y_test
x_train,x_test,y_train,y_test=split(df)

#select perticular metrics plot
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        cm=confusion_matrix(y_test, y_pred, labels=model.classes_)
        plt.figure(figsize=(4,4))
        fig=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig.plot()
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        fpr, tpr, _=roc_curve(y_test, y_pred)
        roc_display=RocCurveDisplay(fpr=fpr, tpr=tpr)
        roc_display.plot()
        st.pyplot()
    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        prec, recall, _=precision_recall_curve(y_test, y_pred)
        pr_display=PrecisionRecallDisplay(precision=prec, recall=recall)
        pr_display.plot()
        st.pyplot()

class_names=['edible','poisnous']

#Select training model
st.sidebar.subheader("Choose Classifier")
classifier=st.sidebar.selectbox("Classifier",("Logistic Regression", "Support Vector Machine", "Naive Bayes", "Decision Tree Classifier", "Random Forest Classifier", "Ada Boost Classifier", "Gradient Boosting Classifier", "XGBoost Classifier"))

#Train Logistic Regression
if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#Train Support Vector Machine Classifier
if classifier == "Support Vector Machine":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Classifier Results")
        model = SVC(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#Train Naive Bayes Classifier
if classifier == "Naive Bayes":
    st.sidebar.subheader("No Hyperparameters to tune")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Naive Bayes Results")
        model = GaussianNB()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#Train Decision Tree Classifier
if classifier == "Decision Tree Classifier":
    st.sidebar.subheader("Hyperparameters")
    criterion_dt = st.sidebar.selectbox("Criterion", ('gini', 'entropy', 'log_loss'))
    max_features_dt = st.sidebar.number_input("Max Features", 50, 100000)
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Decision Tree Classifier Results")
        model = DecisionTreeClassifier(criterion=criterion_dt, max_features=max_features_dt)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#Random Forest Classifier
if classifier == "Random Forest Classifier":
    st.sidebar.subheader("Hyperparameters")
    estimators_rf=st.sidebar.number_input("Number of estimators", 1, 10000)
    criterion_rf= st.sidebar.selectbox("Criterion", ('gini', 'entropy', 'log_loss'))
    max_features_rf = st.sidebar.selectbox("Max Features", (None, 'sqrt', 'log2'))
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Classifier Results")
        model = RandomForestClassifier(n_estimators=estimators_rf, criterion=criterion_rf, max_features=max_features_rf)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#Gradient Boosting Classifier
if classifier == "Gradient Boosting Classifier":
    st.sidebar.subheader("Hyperparameters")
    estimators_gb=st.sidebar.number_input("Number of estimators", 1, 10000)
    learning_rate_gb=st.sidebar.slider("Learning Rate", 0.0, 1.0)
    criterion_gb= st.sidebar.selectbox("Criterion", ('friedman_mse', 'squared_error'))
    max_features_gb= st.sidebar.selectbox("Max Features", ('auto', 'sqrt', 'log2'))
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Gradient Boosting Classifier Results")
        model = GradientBoostingClassifier(n_estimators=estimators_gb, criterion=criterion_gb, max_features=max_features_gb, learning_rate=learning_rate_gb)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#Ada Boost Classifier
if classifier == "Ada Boost Classifier":
    st.sidebar.subheader("Hyperparameters")
    estimators_ada=st.sidebar.number_input("Number of estimators", 1, 100000)
    learning_rate_ada=st.sidebar.slider("Learning Rate", 0.0, 1.0)
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Ada Boost Classifier Results")
        model = AdaBoostClassifier(n_estimators=estimators_ada, learning_rate=learning_rate_ada)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)


#XGBoost Classifier
if classifier == "XGBoost Classifier":
    st.sidebar.subheader("Hyperparameters")
    estimators_xgb=st.sidebar.number_input("Number of estimators", 1, 100000)
    learning_rate_xgb=st.sidebar.slider("Learning Rate", 0.0, 1.0)
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("XGBoost Classifier Results")
        model = XGBClassifier(n_estimators=estimators_xgb, learning_rate=learning_rate_xgb)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        precision=precision_score(y_test, y_pred, labels=class_names)
        recall=recall_score(y_test, y_pred, labels=class_names)
        st.success("Accuracy: {}".format(accuracy))
        st.success("Precision: {}".format(precision))
        st.success("Recall: {}".format(recall))
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        plot_metrics(metrics)

#Connect with me
col1, col2, col3, col4=st.columns(4, gap='small')
with col1:
    st.subheader("[LinkedIn](https://www.linkedin.com/in/surajhonkamble/)")
with col2:
    st.subheader("[GitHub](https://github.com/surajh8596)")
with col3:
    st.subheader("[Instagram](https://www.instagram.com/surajking6958/)")
with col4:
    st.subheader("[Tableau](https://public.tableau.com/app/profile/suraj.honkamble)")