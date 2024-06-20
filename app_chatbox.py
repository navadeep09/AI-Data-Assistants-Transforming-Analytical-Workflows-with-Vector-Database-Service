#Import required libraries
import os 

import streamlit as st
import pandas as pd
import librosa

from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)
from streamlit_chat import message
from auxiliary_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from skimage import exposure
from scipy.io import wavfile
from scipy.fftpack import fft, ifft







#OpenAIKey
os.environ['OPENAI_API_KEY'] = ""
load_dotenv(find_dotenv())

#Title
st.title('AI Data Assistants🤖:')
st.subheader('Transforming Analytical Workflows with vector database service')
#Welcoming message
st.write("Hello, This is an Artifical Intelligence System, helps you with your in understanding data science Concpts over you own dataset.")

#Explanation sidebar
with st.sidebar:
    st.title('AI Data Assistants🤖:')
    st.write('Vamsi Kakani, Navadeep Thotakura, DR. G. Arul Elango')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> made  by Vamsi.K, Navadeep.T</p>",unsafe_allow_html=True)
    


#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    tab1, tab2 = st.tabs(["Data Analysis and Data Science","Multimodal Analysis"])
    with tab1:
        user_csv = st.file_uploader("Upload your file here", type="csv")
        if user_csv is not None:
            user_csv.seek(0)
            df = pd.read_csv(user_csv, low_memory=False)

            #llm model
            llm = OpenAI(temperature = 0)

            #Function sidebar
            @st.cache_data
            def steps_eda():
                steps_eda = llm('What are the steps of EDA')
                return steps_eda
            def pinecone_call(query):
                if query:
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, query)
                    st.subheader("Refined Query:")
                    st.write(refined_query)
                    context = find_match(refined_query)
                    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                    return response
                                                    
            @st.cache_data
            def data_science_framing():
                data_science_framing = llm("Write a couple of paragraphs about the importance of framing a data science problem approriately")
                return data_science_framing
            
            @st.cache_data
            def algorithm_selection():
                data_science_framing = llm("Write a couple of paragraphs about the importance of considering more than one algorithm when trying to solve a data science problem")
                return data_science_framing

            #Pandas agent
            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

            #Functions main
            @st.cache_data
            def function_agent():
                st.write("**Data Overview**")
                st.write("The first rows of your dataset look like this:")
                st.write(df.head())
                st.write("**Data Cleaning**")
                columns_df = pandas_agent.run("What are the meaning of the columns?")
                st.write(columns_df)
                missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                st.write(missing_values)
                duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                st.write(duplicates)
                # Fill missing values with mean
                # Get all column names except 'SL'
                columns_except_SL = [col for col in df.columns if col != 'SL']

                # Fill missing values with mean for all columns except 'SL'
                # df.replace('ab', np.nan, inplace=True)
                # df[columns_except_SL] = df[columns_except_SL].fillna(df[columns_except_SL].mean())
                # st.write("Missing values have been replaced with the mean of the respective feature.")
                st.write("**Data Summarisation**")
                st.write(df.describe())
                correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
                st.write(correlation_analysis)
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
                new_features = pandas_agent.run("What new features would be interesting to create?.")
                st.write(new_features)
                return

            @st.cache_data
            def function_question_variable():
                if option == 'Line Chart':
                    st.line_chart(df, y =[user_question_variable])
                    summary = pandas_agent.run(f"Could you explain the key insights and trends depicted in the  graph chart titled 'The Line graph Chart'? Please summarize the data represented in the graph, including any notable patterns, comparisons, or anomalies. the data feature is {user_question_variable}")
                elif option == 'Bar Chart':
                    st.bar_chart(df[user_question_variable])
                    summary = pandas_agent.run(f"Could you explain the key insights and trends depicted in the  graph chart titled 'The Bar graph Chart'? Please summarize the data represented in the graph, including any notable patterns, comparisons, or anomalies. the data feature is {user_question_variable}")
                elif option == 'Area Chart':
                    st.area_chart(df[user_question_variable])
                    summary = pandas_agent.run(f"Could you explain the key insights and trends depicted in the  graph chart titled 'The Area graph Chart'? Please summarize the data represented in the graph, including any notable patterns, comparisons, or anomalies. the data feature is{user_question_variable}")
                elif option == 'Pie Chart':
                    pie_data = df[user_question_variable].value_counts()
                    plt.figure(figsize=(6,5))
                    plt.pie(pie_data, labels = pie_data.index,autopct='%1.1f%%')
                    plt.title('Pie Chart')
                    st.pyplot(plt.gcf()) 
                    summary = pandas_agent.run(f"Could you explain the key insights and trends depicted in the  graph chart titled 'The Pie graph Chart'? Please summarize the data represented in the graph, including any notable patterns, comparisons, or anomalies. the data feature is {user_question_variable}")
                elif option == 'histogram':
                    plt.figure(figsize=(6,5))
                    sns.histplot(data=df, x=user_question_variable)
                    plt.title(f'Histogram of {user_question_variable}')
                    st.pyplot(plt.gcf())
                    summary = pandas_agent.run(f"Could you explain the key insights and trends depicted in the  graph chart titled 'The histogram Chart'? Please summarize the data represented in the graph, including any notable patterns, comparisons, or anomalies. the data feature is {user_question_variable}")
                elif option == 'correlation_analysis':
                    var1,var2 = user_question_variable.split(',')
                    plt.figure(figsize=(6,5))
                    sns.scatterplot(data=df, x=var1, y=var2)
                    plt.title(f'Correlation between {var1} and {var2}')
                    st.pyplot(plt.gcf())
                    df_subset = df[[var1, var2]]
                    # Calculate the correlation matrix
                    corr = df_subset.corr()
                    # Plot the heatmap
                    sns.heatmap(corr, annot=True, cmap='coolwarm')
                    plt.title(f'Correlation between {var1} and {var2}')
                    st.pyplot(plt.gcf())
                    summary = pandas_agent.run(f"Could you explain the key insights and trends depicted in the  graph chart titled 'The correlation_analysis Chart'? Please summarize the data represented in the graph, including any notable patterns, comparisons, or anomalies. the data feature is {var1} and {var2}")                
                st.write(summary)
                normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
                st.write(normality)
                outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
                st.write(outliers)
                trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
                st.write(trends)
                missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
                st.write(missing_values)
                return
            
            
            def function_question_dataframe():
                dataframe_info = pandas_agent.run(user_question_dataframe)
                st.write(dataframe_info)
                return

            
            
            #Main

            st.header('Exploratory data analysis')
            st.subheader('General information about the dataset')

            with st.sidebar:
                with st.expander('What are the steps of EDA'):
                    st.write(steps_eda())

            function_agent()

            st.subheader('Variable of study')
            option = st.selectbox(
                'Select an Type of plot you want to visualize the variable with:',
                ('Line Chart', 'Bar Chart', 'Area Chart', 'Pie Chart', 'histogram', 'correlation_analysis')
            )

            # Display the selected option.
            st.write('You selected:', option)
            user_question_variable = st.text_input('What variable are you interested in')
            
            if user_question_variable is not None and user_question_variable !="":
                function_question_variable()

                st.subheader('Further study')

            if user_question_variable:
                user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
                if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                    function_question_dataframe()
                if user_question_dataframe in ("no", "No"):
                    st.write("")
    with tab2:
        uploaded_file = st.file_uploader("Upload a file", type=['png', 'jpg', 'jpeg', 'mp3', 'wav'])
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            st.write(file_details)

            # Check the file type and handle the file upload
            if uploaded_file.type == "audio/mp3" or uploaded_file.type == "audio/wav":
                st.audio(uploaded_file, format='audio/wav')
                with open('saved_audio.wav', 'wb') as f:
                    f.write(uploaded_file.getvalue())
                audio, sr = librosa.load(uploaded_file, sr=None)
                # Generate spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                plt.figure(figsize=(12, 4))
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Spectrogram')
                st.pyplot(plt.gcf())
                sample_rate, data = wavfile.read('saved_audio.wav')

                # Take the Fourier transform of the data
                fft_data = fft(data)

                # Take the logarithm of the absolute value of the Fourier transform
                log_fft_data = np.log(np.abs(fft_data))

                # Take the inverse Fourier transform of the result
                cepstrum = np.abs(ifft(log_fft_data))

                # Plot the cepstrum
                plt.figure(figsize=(10, 4))
                plt.plot(cepstrum[:len(cepstrum)//2])
                plt.title('Cepstrum')
                plt.xlabel('Quefrency')
                plt.ylabel('Amplitude')
                st.pyplot(plt.gcf())
            elif uploaded_file.type == "image/png" or uploaded_file.type == "image/jpeg":
                image = Image.open(uploaded_file)
                
                # Convert image to grayscale
                image = image.convert('L')
                image = np.array(image)

                # Calculate histogram and PDF
                hist, bins_center = exposure.histogram(image)

                plt.figure(figsize=(9, 4))
                plt.subplot(1, 2, 1)
                plt.title('Histogram')
                plt.plot(bins_center, hist, lw=2)
                plt.subplot(1, 2, 2)
                plt.title('PDF')
                pdf = hist / np.sum(hist)
                plt.plot(bins_center, pdf, lw=2)
                st.pyplot(plt.gcf())

                