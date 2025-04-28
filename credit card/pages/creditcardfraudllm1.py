import timeit
import pandas as pd
#import matplotlib.pyplot as plt
#import plotly.express as px
import numpy as np
import warnings
import pandas as pd
import pickle
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
warnings.filterwarnings("ignore")
from openai import OpenAI
import streamlit as st
import subprocess
import json

def display_gen_ai_fraud_form():
    
    if st.session_state.transaction_details2 is  None: 
        st.title('Credit Card Fraud Detection!')
    else:
        initialize_session_state()
    

    client = OpenAI(api_key='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')
    # Initialize the language model, use text-davinci-003's successor: 'text-davinci-002'"])
    f1 = open("credit card/pages/Fraud_llm", "rb")
    model = pickle.load(f1)
    os.environ["OPENAI_API_KEY"] = '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'
    # Initialize the language model, use text-davinci-003's successor: 'text-davinci-002'
    llm = ChatOpenAI(model_name="gpt-3.5-turbo") # Changed the model name to a supported model
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Define the prompt template
    prompt_template = PromptTemplate(
    input_variables=["transaction_details"],
    template="Given the following transaction details, explain why it might be considered fraudulent:\n{transaction_details}"
    )
    transaction_details = """
        Transaction ID: 123456
        Amount: $5000
        Time: 02:30 AM
        Location: Unknown
        Previous Transactions: None
        """
    response=""



    # Create the LLMChain
    fraud_explainer = LLMChain(llm=llm, prompt=prompt_template)
   # st.write("outside session state")
    if st.session_state.transaction_details2 is not None:  # check if not null
           # st.write("inside session state")
            #st.write(st.session_state.transaction_details2)
            explanation = fraud_explainer.run(st.session_state.transaction_details2)
            return explanation
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    def new_scores():
                
                #st.session_state.messages.append({"role": "assistant", "content": transaction_ID})
            #st.session_state.messages.append({"role": "assistant", "content": st.session_state.transaction_ID})
            #st.session_state.messages.append({"role": "assistant", "content": st.session_state.amount})
            #st.session_state.messages.append({"role": "assistant", "content": st.session_state.time})
            #st.session_state.messages.append({"role": "assistant", "content": st.session_state.location})
                #st.session_state.messages.append({"role": "assistant", "content": st.session_state.previous_Transactions})
                #response = st.write("processing new record....") 
                #st.session_state.messages.append({"role": "assistant", "content": "processing. new record...."})
            
            transaction_details1 = ""+ "Transaction ID:" + str(st.session_state.transaction_ID ) + "   Amount:"  + str(st.session_state.amount)  + "  Time:" + st.session_state.time  + "   Location:" +st.session_state.location +  "   Previous Transactions: 100" + ""
            #response =   st.write(transaction_details1)
            st.session_state.messages.append({"role": "assistant", "content":transaction_details1})
            try:
                explanation = fraud_explainer.run(transaction_details1)
                st.write(explanation)
                st.session_state.messages.append({"role": "assistant", "content": explanation})
            except Exception as e:
                st.error(f"An error occurred: {e}")
    st.write("Welcome!!!,Please Enter Yes to proceed with Fraud Detection")
    if prompt := st.chat_input("Welcome!!!,Please Enter Yes to proceed with Fraud Detection"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        #st.session_state.messages.append({"role": "assistant", "content": "Welcome!!!,Please Enter Yes to proceed with Fraud Detection"})

        score_df = pd.DataFrame(st.session_state.scores)
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
        
            # Check the user's message using an if statement
        if "yes" in prompt.lower():
            response = st.write("Transaction ID: 123456,Amount: $5000,Time: 02:30 AM,Location: Unknown,Previous Transactions: None") 
            st.session_state.messages.append({"role": "assistant", "content": "Transaction ID: 123456,Amount: $5000,Time: 02:30 AM,Location: Unknown,Previous Transactions: None"})
            response =st.write("Shall i Proceed with these sample Transaction Details(Type proceed else Type new record)")
            st.session_state.messages.append({"role": "assistant", "content": "Shall i Proceed with these sample Transaction Details(Type proceed else new record)"})
            
        elif "new record" in prompt.lower():
            st.write("# Sample Transction for Fraud Detection")

            score_df = pd.DataFrame(st.session_state.scores)
        
            st.write(score_df)
            
            
            
            st.write("# Add a New Transaction Details for checking Fraudulent")
            with st.form("new_score", clear_on_submit=False):   
                transaction_ID= st.text_input("Transaction_ID", key="transaction_ID")
                amount = st.number_input("Amount", key="amount", step=1, value=1, min_value=1)
                time= st.text_input("Time", key="time")
                location= st.text_input("Location", key="location")
                previous_Transactions= st.text_input("Previous_Transactions", key="previous_transactions")
                st.form_submit_button("Submit", on_click=new_scores)
                
                
        elif "no" in prompt.lower():
            
            response = st.write("no Response") 
            st.session_state.messages.append({"role": "assistant", "content": "no Response"})
            st.write("Welcome!!!,Please Enter Yes to proceed with Fraud Detection")
        
        elif "proceed" in prompt.lower():
            #st.session_state.messages.append({"role": "assistant", "content": "proceed"})
            explanation = fraud_explainer.run(transaction_details)
            st.write(explanation)
            st.session_state.messages.append({"role": "assistant", "content": explanation})
            st.write("Welcome!!!,Please Enter Yes to proceed with Fraud Detection")
        
        elif st.session_state.transaction_details2 is not None:  # check if not null
                
            explanation = fraud_explainer.run(st.session_state.transaction_details2)
            return
        
        else:
            response = st.write("I didn't Understand") 
            st.session_state.messages.append({"role": "assistant", "content": "I didn't Understand"})
            st.write("Welcome!!!,Please Enter Yes to proceed with Fraud Detection")
    
        

    f1.close()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Add other session state variables here
    if "Transaction_ID" not in st.session_state:
        transaction_ID = 123457
    if "Amount" not in st.session_state:
        amount = 500
    if "Time" not in st.session_state:
        time = "2:30 PM" 
    if "Location" not in st.session_state:
        location = "london"
    if "previous_Transactions" not in st.session_state:
        previous_Transactions = "no" 
    if "scores" not in st.session_state:
        st.session_state["scores"] = []
        st.session_state.scores = [
            {"Transaction_ID": "123456", "Amount": "$5000", "Time": "02:30 AM","Location": "Unknown","Previous_Transactions": "None"}
        ]
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
        

    if "messages" not in st.session_state:
        st.session_state.messages = []

initialize_session_state()

if __name__ == "__main__":
    initialize_session_state()
    display_gen_ai_fraud_form()
