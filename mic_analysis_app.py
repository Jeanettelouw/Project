#!/usr/bin/env python
# coding: utf-8

# In[46]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go


# In[47]:


def plot_data(data, year, observed_MIC, ECOFF_value, start_year, end_year):
    # Remove any asterisks from the year column and convert to integers
    data[year] = data[year].astype(str).str.replace('*', '').astype(int)
    
    # Filter out year
    data = data[(data[year] >= start_year) & (data[year] <= end_year)]
    
    # Classify isolates based on ECOFF value
    data['classify_ECOFF'] = data[observed_MIC].apply(lambda x: 'R' if x > ECOFF_value else 'NR')
    
    # Proportion of resistant and non-resistant isolates for each year
    resistant_prop = data.groupby(year)['classify_ECOFF'].apply(lambda x: (x == 'R').mean()).reset_index(name='resistant_proportions')
    non_resistant_prop = data.groupby(year)['classify_ECOFF'].apply(lambda x: (x != 'R').sum()).reset_index(name='non_resistant_count')
    non_resistant_prop['non_resistant_proportions'] = non_resistant_prop['non_resistant_count'] / non_resistant_prop['non_resistant_count'].sum()
    
    # Average for log2(MIC) of resistant and non-resistant isolates for each year
    log2_mic_non_resistant = data[data['classify_ECOFF'] != 'R'].groupby(year)[observed_MIC].apply(lambda x: np.mean(np.log2(x))).reset_index(name='log2_mic_mean')
    log2_mic_resistant = data[data['classify_ECOFF'] == 'R'].groupby(year)[observed_MIC].apply(lambda x: np.mean(np.log2(x))).reset_index(name='log2_mic_mean')

    # Plot for non-resistant isolates
    fig, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(non_resistant_prop[year], non_resistant_prop['non_resistant_proportions'], color='lightsteelblue', label='Proportion')
    ax2.set_ylabel("Proportion on categorical scale")
    ax2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 1.015), loc='upper left')
    ax2.set_yticks([i*0.02 for i in range(int(max(non_resistant_prop['non_resistant_proportions'] + 0.02)/0.02)+1)])
    ax2_2 = ax2.twinx()
    ax2_2.plot(log2_mic_non_resistant[year], log2_mic_non_resistant['log2_mic_mean'], color='steelblue', linestyle='-', label='Average of log2(MIC) levels', linewidth=1.2)
    ax2_2.set_ylabel("Average of log2(MIC) levels")
    ax2_2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 0.9), loc='lower left')
    ax2.set_title("Non-resistant Isolates")
    ax2.set_xlabel("Year")
    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this
    
    # Plot for resistant isolates
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(resistant_prop[year], resistant_prop['resistant_proportions'], color='salmon', label='Proportion')
    ax1.set_ylabel("Proportion on categorical scale")
    ax1.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 1.015), loc='upper left')
    ax1.set_yticks([i*0.02 for i in range(int(max(resistant_prop['resistant_proportions'] + 0.02)/0.02)+1)])
    ax1_2 = ax1.twinx()
    ax1_2.plot(log2_mic_resistant[year], log2_mic_resistant['log2_mic_mean'], color='brown', linestyle='-', label='Average of log2(MIC) levels', linewidth=1.2)
    ax1_2.set_ylabel("Average of log2(MIC) levels")
    ax1_2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 0.9), loc='lower left')
    ax1.set_title("Resistant Isolates")
    ax1.set_xlabel("Year")
    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this

# Title of the app
st.title("Antimicrobial Susceptibility Testing")

# Load the data directly from the file
data_path = "C:\\Users\\Jeanette\\Desktop\\UNI\\Y4 Hons\\Semester 1\\STK 795\\Code\\mic_analysis_app\\IsolateData28Jun.xlsx"
data = pd.read_excel(data_path)

year = st.selectbox("Select the year column", data.columns)
observed_MIC = st.selectbox("Select the MIC value column of specific AM agent", data.columns)
ECOFF_value = st.number_input("Enter ECOFF value of AM-bacteria combination", min_value=0.0, value=1.0, step=0.1)
start_year = st.slider("Start Year", min_value=2000, max_value=2025, value=2020)#st.number_input("Start Year", min_value=2000, max_value=2024, value=2020)
end_year = st.slider("End Year", min_value=2000, max_value=2025, value=2020)#st.number_input("End Year", min_value=2000, max_value=2024, value=2020)

if st.button("Generate Temporal Plot"):
    plot_data(data, year, observed_MIC, ECOFF_value, start_year, end_year)


# In[44]:


#Command prompt: Powershell Prompt in Anaconda
#conda info --envs
#conda activate C:\Users\Jeanette\anaconda3
#cd "C:\Users\Jeanette\Desktop\UNI\Y4 Hons\Semester 1\STK 795\Code\mic_analysis_app"
#streamlit run mic_analysis_app.py

# Enter -> opens tab to upload 


# In[45]:


# To see temporal trends
#data = pd.read_excel("C:\\Users\\Jeanette\\Desktop\\UNI\\Y4 Hons\\Semester 1\\STK 795\\Code\\mic_analysis_app\\IsolateData28Jun.xlsx")
#year = 'Data Year'
#observed_MIC = 'CHL Rslt'
#ECOFF_value = 16
#start_year = 2005
#end_year = 2024

#plot_data(data, year, observed_MIC, ECOFF_value, start_year, end_year)

