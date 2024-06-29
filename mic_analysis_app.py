#!/usr/bin/env python
# coding: utf-8

# In[53]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import re


# In[54]:


# Year based plot
def create_yearly_mic_table(data, selected_year, observed_MIC, ECOFF_value, equiv_column):
    # Get all unique MIC values from the entire dataset
    all_mic_values = sorted(
        set((data[equiv_column] + data[observed_MIC].astype(str)).unique()),
        key=lambda x: int(re.search(r'\d+', x).group(0))  
    )
    
    # Filter data for the specific year
    data_year = data[data['Year'].astype(str).str.replace('*', '').astype(int) == selected_year]
    
    # Combine the equivalence and MIC result into a single column
    data_year['Adjusted MIC'] = data_year[equiv_column] + data_year[observed_MIC].astype(str)
    
    # Calculate counts and percentages
    mic_counts = data_year['Adjusted MIC'].value_counts().reindex(all_mic_values, fill_value=0)
    mic_table = pd.DataFrame({'MIC': mic_counts.index, 'Count': mic_counts.values})
    
    total_count = mic_table['Count'].sum()
    mic_table['Total'] = total_count
    mic_table['Percent'] = (mic_table['Count'] / total_count) * 100
    
    # Function to clean MIC values and extract integers
    def clean_mic_value(mic_str):
        return int(re.search(r'\d+', mic_str).group(0))
    
    # Determine Conclusion and sort table
    mic_table['Conclusion'] = mic_table['MIC'].apply(lambda x: 'R' if clean_mic_value(x) > ECOFF_value else 'NR')
    mic_table['MIC_clean'] = mic_table['MIC'].apply(clean_mic_value)
    mic_table = mic_table.sort_values(by='MIC_clean').drop(columns=['MIC_clean'])

    # Plot by year
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with different colors for R and NR
    bar_width = 0.8  # Bar width can be adjusted if needed
    nr_bars = ax.bar(mic_table[mic_table['Conclusion'] == 'NR']['MIC'], 
                     mic_table[mic_table['Conclusion'] == 'NR']['Percent'], 
                     color='lightsteelblue', label='Non-resistant', width=bar_width)
    r_bars = ax.bar(mic_table[mic_table['Conclusion'] == 'R']['MIC'], 
                    mic_table[mic_table['Conclusion'] == 'R']['Percent'], 
                    color='salmon', label='Resistant', width=bar_width)
    
    # Add labels
    for bars in [nr_bars, r_bars]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', 
                 va='bottom', ha='center')
    
    # Add a red broken line for the ECOFF value
    ecoff_str = f'CHL{ECOFF_value}'  # Convert ECOFF value to match the MIC format in the x-axis
    if ecoff_str in mic_table['MIC'].values:
        ecoff_position = mic_table[mic_table['MIC'] == ecoff_str].index[0]
        ax.axvline(x=ecoff_position + 0.5 * bar_width, color='red', linestyle='--', label='ECOFF')
    else:
        closest_mic = min(all_mic_values, key=lambda x: abs(clean_mic_value(x) - ECOFF_value))
        closest_position = mic_table[mic_table['MIC'] == closest_mic].index[0]
        ax.axvline(x=closest_position + 0.5 * bar_width, color='red', linestyle='--', label='ECOFF')

    # Create legend
    ax.legend(loc='upper left', facecolor='white', edgecolor='none')
    
    ax.set_xlabel('MIC (µg/mL)')
    ax.set_ylabel('Percentage')
    ax.set_title(f'MIC Distribution for {selected_year}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    #return mic_table # can comment this out


# In[55]:


# Temporal plot
def plot_data(data, observed_MIC, ECOFF_value, start_year, end_year):
    
    # Remove any asterisks from the year column and convert to integers
    #data[Year] = data[Year].astype(str).str.replace('*', '').astype(int)
    
    # Filter out year
    data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
    
    # Classify isolates based on ECOFF value
    data['classify_ECOFF'] = data[observed_MIC].apply(lambda x: 'R' if x > ECOFF_value else 'NR')
    
    # Proportion of resistant and non-resistant isolates for each year
    resistant_prop = data.groupby('Year')['classify_ECOFF'].apply(lambda x: (x == 'R').mean()).reset_index(name='resistant_proportions')
    non_resistant_prop = data.groupby('Year')['classify_ECOFF'].apply(lambda x: (x != 'R').sum()).reset_index(name='non_resistant_count')
    non_resistant_prop['non_resistant_proportions'] = non_resistant_prop['non_resistant_count'] / non_resistant_prop['non_resistant_count'].sum()
    
    # Average for log2(MIC) of resistant and non-resistant isolates for each year
    log2_mic_non_resistant = data[data['classify_ECOFF'] != 'R'].groupby('Year')[observed_MIC].apply(lambda x: np.mean(np.log2(x))).reset_index(name='log2_mic_mean')
    log2_mic_resistant = data[data['classify_ECOFF'] == 'R'].groupby('Year')[observed_MIC].apply(lambda x: np.mean(np.log2(x))).reset_index(name='log2_mic_mean')

    # Plot for non-resistant isolates
    fig, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(non_resistant_prop['Year'], non_resistant_prop['non_resistant_proportions'], color='lightsteelblue', label='Proportion')
    ax2.set_ylabel("Proportion on categorical scale")
    ax2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 1.015), loc='upper left')
    ax2.set_yticks([i*0.02 for i in range(int(max(non_resistant_prop['non_resistant_proportions'] + 0.02)/0.02)+1)])
    ax2_2 = ax2.twinx()
    ax2_2.plot(log2_mic_non_resistant['Year'], log2_mic_non_resistant['log2_mic_mean'], color='steelblue', linestyle='-', label='Average of log2(MIC) levels', linewidth=1.2)
    ax2_2.set_ylabel("Average of log2(MIC) levels")
    ax2_2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 0.9), loc='lower left')
    ax2.set_title("Non-resistant Isolates")
    ax2.set_xlabel("Year")
    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this
    
    # Plot for resistant isolates
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(resistant_prop['Year'], resistant_prop['resistant_proportions'], color='salmon', label='Proportion')
    ax1.set_ylabel("Proportion on categorical scale")
    ax1.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 1.015), loc='upper left')
    ax1.set_yticks([i*0.02 for i in range(int(max(resistant_prop['resistant_proportions'] + 0.02)/0.02)+1)])
    ax1_2 = ax1.twinx()
    ax1_2.plot(log2_mic_resistant['Year'], log2_mic_resistant['log2_mic_mean'], color='brown', linestyle='-', label='Average of log2(MIC) levels', linewidth=1.2)
    ax1_2.set_ylabel("Average of log2(MIC) levels")
    ax1_2.legend(title="", title_fontsize='11', fontsize='9', bbox_to_anchor=(0.01, 0.9), loc='lower left')
    ax1.set_title("Resistant Isolates")
    ax1.set_xlabel("Year")
    plt.tight_layout()
    st.pyplot(fig)  # Replace plt.show() with this


# In[56]:


# Title of the app
st.title("Antimicrobial Susceptibility Testing")

# Description of the app
st.markdown("""
This app analyses temporal trends antimicrobial resistance data from uploaded files.""")

# Display a warning message
st.warning("""
Please ensure the required format to proceed. This app supports only XLSX file format. 
Ensure the following column names: 'Year', 'Genus', 'Species', 'Serotype'.
Moreover, 'Year' column must be four digital format. While there must be at least one column of MIC values (real number format) 
for a specific antimicrobial agent which corresponds to a seperate column of a sign column (symbols of inequality and equality format).
A unique ECOFF value must be predetermined for each AM-bacteria combination, or can alternatively be found 
[here](https://mic.eucast.org/search/).
""")

# File uploader to allow users to upload their own data file
uploaded_file = st.file_uploader("Choose an XLSX file", type="xlsx")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    min_year = int(data['Year'].min())
    max_year = int(data['Year'].max())
    
    # Select boxes for Genus, Species, and Serotype
    genus = st.selectbox("Select the Genus", data['Genus'].unique())
    species = st.selectbox("Select the Species", data['Species'].unique())
    serotype = st.selectbox("Select the Serotype", data['Serotype'].unique())
    filtered_data = data[(data['Genus'] == genus) & (data['Species'] == species) & (data['Serotype'] == serotype)]
        
    ECOFF_value = st.number_input("Enter unique ECOFF value", min_value=0.0, value=1.0, step=0.001)
    observed_MIC = st.selectbox("Select the MIC value column", data.columns)
    equiv_column = st.selectbox("Select sign column", data.columns)    
        
        
    st.markdown("""
    A yearly pecentage histogram of resistant and non-resistant for 
    MIC (minimum inhibitory concentration) values of a specific antimicrobial-bacteria combination can be visualised, called MIC distribution.""")
    selected_year = st.slider("Select year", min_year, max_year, 2020)
    if st.button("Generate year plot"):
        create_yearly_mic_table(filtered_data, selected_year, observed_MIC, ECOFF_value, equiv_column)
    
    
    st.markdown(""" When the non-resistant and resistant bars are both stacked seperately for the MIC distribution, a temporal trend can be achieved. 
    This is seen by the histogram proportions of resistant and non-resistant isolates over the years for a specific antimicrobial-bacteria combination.
    However, the histogram does not account for the ordinal nature of MIC values. 
    Essentially, the closer the bars are to the ECOFF line in the MIC distribution, the more dangerously close they are to becoming resistant over time.
    Enhanced insights can be gained by assigning weights to the MIC values. One approach is to take the logarithm (base 2) of the observed MIC values 
    and average these values for each year, represented by overlapping curves.
    Despite these improvements, the results still do not fully capture the nature of the data. This is because they overlook important characteristics,
    such as modeling the observed MIC values rather than the true ones within those bounds, and neglect the left- and right-censoring of the observed MIC values""")
    start_year, end_year = st.slider("Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    if st.button("Generate temporal plot"):
        plot_data(filtered_data, observed_MIC, ECOFF_value, start_year, end_year)


# In[ ]:





# In[ ]:




