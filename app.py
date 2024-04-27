import streamlit as st
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

import os

#st.write(os.getcwd())


@st.cache_data
def load_data():
    df = pd.read_csv("./Country-data.csv")
    df["log_income"] = np.log10(df["income"])
    df["log_gdpp"] = np.log10(df["gdpp"])
    return df



#######################################
# Load and process data
#######################################

df = load_data()
#col1, col2 = st.columns(2)

colnames = df.columns.to_list()
colnames.remove("country")
colnames.remove("life_expec")
#selected_col = st.sidebar.selectbox("Field to compare", colnames, index=None, placeholder="Select a field")



target_name = "country"
target = df[target_name]
data = df.drop(columns=target_name)




#######################################
#  display
#######################################

st.title("Countries clustering")
st.write(
    """The date of the dataset is not specified. Information on the source page let believe it could be from 2020 or even before"""
)



st.header("Data Exploration")



with st.container(border=True):

    st.write(df)
    
    with st.expander("Statistiques"):
        st.write( df.describe() )
    
    with st.expander("Code book"):
        file_path = "./data-dictionary.csv"
        df_code = pd.read_csv(file_path)
        st.table(df_code)

        
    st.write(
        "<a href='https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data/' target='_blank'>Data source (on Kaggle)</a>",
        unsafe_allow_html=True
    )



with st.container(border=True):

    st.subheader("Life expectancy vs other columns")
    
    selected_col = st.radio("Select a field", options=colnames, horizontal = True)
    st.caption("Note: income and gdpp have a log relationship with life expectancy")
    if selected_col:
        st.write(f"Life expectancy vs {selected_col}") # investments (in % of GDP per capita)")
        fig2 = plt.figure(figsize=(10, 4))
        sns.regplot(data=df, x=selected_col, y="life_expec");    
        st.pyplot(fig2)
    

    st.write("Life expectancy vs log of GDP per capita and log of income ")
    #fig1 = sns.relplot(data=df, x="gdpp",  y="health", hue="life_expec")
    fig1 = sns.relplot(data=df, x="log_gdpp",  y="log_income", hue="life_expec")
    st.pyplot(fig1)
    #plt.show()

    

#################
# clustering
#################
st.header("Data Clustering")


with st.container(border=True):

    cols = ", ".join(list(data.columns.values))
    st.markdown(f"""
        ##### Clustering is done using columns: 
        {cols}
    """)
    
    # execute clustering
    n_cluster = st.slider('Number of clusters (2 to 10)', 2, 10, 2)

    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto")
    model = make_pipeline(StandardScaler(), kmeans)
    #st.write(model)
    estimator = model.fit(data)    
    #st.write(kmeans.labels_)

    
    # display  Number of countries per cluster
    values, counts = np.unique(kmeans.labels_, return_counts=True)
    df_vc =  pd.DataFrame( {"value": values, "count": counts} )

    st.subheader("Number of countries per cluster")
    st.bar_chart(df_vc, x="value", y="count")
    
    
    
    
    # assigner les countries aux groupes
    df_clustered = df[ ["country", "life_expec"] ]
    df_clustered["group"] = kmeans.labels_

    # calculer le mean life expectancy by group
    grouped = df_clustered[ ["group", "life_expec"] ].groupby('group')         
    grouped = grouped.mean()  
    grouped["life_expec"] = grouped["life_expec"].apply(lambda x: round(x, 2))

    
    # afficher les moyennes
    st.subheader("Mean life expectancy by cluster")
    st.write("Unsorted")
    st.dataframe(grouped.T)
    st.write("Sorted")
    df_sorted = grouped.sort_values("life_expec", ascending=False)
    st.dataframe(df_sorted.T)
    #st.line_chart(df_sorted.reset_index(), x="group", y="life_expec")
    
    # choisir le groupe
    st.subheader("Clusters details")
    
    unique_labels = np.unique(kmeans.labels_)
    selected_group = st.radio("Select a cluster", 
        options=unique_labels, 
        horizontal = True,
        format_func=lambda x: f"Cluster #{x}"
    )
    
    # afficher le groupe choisi
    if (selected_group != None):
        
        df_group = df_clustered.query(f"group == {selected_group}")
    
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("##### Statistics", unsafe_allow_html=True)
            st.write(df_group.describe())
        with col2:
        
            st.markdown("##### Countries in this cluster", unsafe_allow_html=True)
            st.write(df_group.style.background_gradient(subset=["life_expec"], cmap="RdYlGn"))
     

    #df_clustered
   
  
    
    
  
st.write(  
'''
    -----------
    JP Tanguay (2024)
'''
)