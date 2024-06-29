#!/usr/bin/env python
# coding: utf-8

# In[46]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go


# In[49]:


def plot_data(data, observed_MIC, ECOFF_value, start_year, end_year):
    
    # Remove any asterisks from the year column and convert to integers
    #data[Year] = data[Year].astype(str).str.replace('*', '').astype(int)
    
    # Filter out year
    data = data[(data[Year] >= start_year) & (data[Year] <= end_year)]
    
    # Classify isolates based on ECOFF value
    data['classify_ECOFF'] = data[observed_MIC].apply(lambda x: 'R' if x > ECOFF_value else 'NR')
    
    # Proportion of resistant and non-resistant isolates for each year
    resistant_prop = data.groupby(Year)['classify_ECOFF'].apply(lambda x: (x == 'R').mean()).reset_index(name='resistant_proportions')
    non_resistant_prop = data.groupby(Year)['classify_ECOFF'].apply(lambda x: (x != 'R').sum()).reset_index(name='non_resistant_count')
    non_resistant_prop['non_resistant_proportions'] = non_resistant_prop['non_resistant_count'] / non_resistant_prop['non_resistant_count'].sum()
    
    # Average for log2(MIC) of resistant and non-resistant isolates for each year
    log2_mic_non_resistant = data[data['classify_ECOFF'] != 'R'].groupby(year)[observed_MIC].apply(lambda x: np.mean(np.log2(x))).reset_index(name='log2_mic_mean')
    log2_mic_resistant = data[data['classify_ECOFF'] == 'R'].groupby(year)[observed_MIC].apply(lambda x: np.mean(np.log2(x))).reset_index(name='log2_mic_mean')

    # Plot for non-resistant isolates
    fig, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(non_resistant_prop[Year], non_resistant_prop['non_resistant_proportions'], color='lightsteelblue', label='Proportion')
    ax2.set_ylabel("Proportion on categorical scale")
    ax2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 1.015), loc='upper left')
    ax2.set_yticks([i*0.02 for i in range(int(max(non_resistant_prop['non_resistant_proportions'] + 0.02)/0.02)+1)])
    ax2_2 = ax2.twinx()
    ax2_2.plot(log2_mic_non_resistant[Year], log2_mic_non_resistant['log2_mic_mean'], color='steelblue', linestyle='-', label='Average of log2(MIC) levels', linewidth=1.2)
    ax2_2.set_ylabel("Average of log2(MIC) levels")
    ax2_2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 0.9), loc='lower left')
    ax2.set_title("Non-resistant Isolates")
    ax2.set_xlabel("Year")
    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this
    
    # Plot for resistant isolates
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(resistant_prop[Year], resistant_prop['resistant_proportions'], color='salmon', label='Proportion')
    ax1.set_ylabel("Proportion on categorical scale")
    ax1.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 1.015), loc='upper left')
    ax1.set_yticks([i*0.02 for i in range(int(max(resistant_prop['resistant_proportions'] + 0.02)/0.02)+1)])
    ax1_2 = ax1.twinx()
    ax1_2.plot(log2_mic_resistant[Year], log2_mic_resistant['log2_mic_mean'], color='brown', linestyle='-', label='Average of log2(MIC) levels', linewidth=1.2)
    ax1_2.set_ylabel("Average of log2(MIC) levels")
    ax1_2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 0.9), loc='lower left')
    ax1.set_title("Resistant Isolates")
    ax1.set_xlabel("Year")
    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this


# In[51]:


# Title of the app
st.title("Antimicrobial Susceptibility Testing")

# Description of the app
st.markdown("""
This app analyses antimicrobial resistance data from uploaded files. It visualises the 
proportions of resistant and non-resistant isolates over the years for a specific antimicrobial-bacteria combination using histograms. 
The histogram analysis does not account for the ordinal levels of MIC (minimum inhibitory concentration) data. 
Improved insights are provided by plotting overlapping curves representing the average log2(MIC) levels. 
However, these curves may not accurately reflect the data due to the neglect of important characteristics such as left- and right-censoring.""")

# Display a warning message
st.warning("""
Please ensure the required format to proceed. This app supports only XLSX file format. 
Ensure the following column names: 'Year', 'Genus', 'Species', 'Serotype'.
Moreover, 'Year' column must be four digital format. While there must be at least one column of MIC values, in real number format, for a specific antimicrobial agent. 
A unique ECOFF value must be predetermined for each AM-bacteria combination, or can alternatively be found 
[here](https://mic.eucast.org/search/).
""")

# File uploader to allow users to upload their own data file
uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    min_year = 2000
    max_year = 2024
    
    # Select boxes for Genus, Species, and Serotype
    genus = st.selectbox("Select the Genus", data['Genus'].unique())
    species = st.selectbox("Select the Species", data['Species'].unique())
    serotype = st.selectbox("Select the Serotype", data['Serotype'].unique())
    
    # Filter data based on selected Genus, Species, and Serotype
    filtered_data = data[(data['Genus'] == genus) & (data['Species'] == species) & (data['Serotype'] == serotype)]
    
    observed_MIC = st.selectbox("Select the MIC value column", data.columns)
    ECOFF_value = st.number_input("Enter ECOFF value", min_value=0.0, value=1.0, step=0.1)
    start_year, end_year = st.slider("Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    #start_year = st.slider("Start Year", min_value=2000, max_value=2024, value=2020)
    #end_year = st.slider("End Year", min_value=2000, max_value=2024, value=2020)
    
    if st.button("Generate temporal plot"):
        plot_data(data, observed_MIC, ECOFF_value, start_year, end_year)


# In[ ]:





# In[ ]:




