import streamlit as st
import pandas as pd
import time  # for simulating real-time updates
#import plotly.express as px  # for interactive charts

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# Sidebar menu for navigation
st.sidebar.title("Navigation")
menu_option = st.sidebar.radio(
    "Go to",
    options=["Dashboard", "Predict Fraud", "Gen AI Fraud"]
)

# Initialize session state for monitoring, index tracking, and metrics
if "monitoring" not in st.session_state:
    st.session_state.monitoring = True  # Default to monitoring mode
if "current_index" not in st.session_state:
    st.session_state.current_index = 0  # Start from the first record
if "genuine_count" not in st.session_state:
    st.session_state.genuine_count = 0  # Initialize genuine transaction count
if "genuine_5_AccountAge_Count" not in st.session_state:
    st.session_state.genuine_5_AccountAge_Count = 0  # Initialize genuine transaction count
if "genuine_10_AccountAge_Count" not in st.session_state:
    st.session_state.genuine_10_AccountAge_Count = 0  # Initialize genuine transaction count
if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0  # Initialize fraud transaction count
if "fraud_Online_purchase_count" not in st.session_state:
    st.session_state.fraud_Online_purchase_count = 0  # Initialize fraud transaction count
if "fraud_first_purchase_count" not in st.session_state:
    st.session_state.fraud_first_purchase_count = 0  # Initialize fraud transaction count
st.session_state.transaction_details2 = ""  # Initialize transaction details for Gen AI Fraud page

# Dashboard Page
if menu_option == "Dashboard":
    st.title("Real-Time Fraud Detection Metrics")

    # Read CSV from a URL
    @st.cache_data
    def get_data() -> pd.DataFrame:
        dataset_url = "https://drive.google.com/uc?export=download&id=1IwtwzwhEQApAZJJX8XmEWtJh3RUWxVvw"
        return pd.read_csv(dataset_url)

    df = get_data()

    # Create placeholders for live metrics
    genuine_placeholder = st.empty()
    fraud_placeholder = st.empty()
    geunie_M1_placeholder = st.empty()
    geunie_M2_placeholder = st.empty()
    fraud_M1_placeholder = st.empty()
    fraud_M2_placeholder = st.empty()
    myKey = 'my_key'
    if myKey not in st.session_state:
        st.session_state[myKey] = False

    if  st.session_state[myKey]:
        myBtn = st.button('Monitor Transactions')
        st.session_state.monitoring = False
        st.session_state[myKey] = False
    else:
        myBtn = st.button('Stop and Generate Report')
        st.session_state[myKey] = True
        st.session_state.monitoring = True
    # Toggle button for monitoring using session state
   # if "stop_button_clicked" not in st.session_state:
    #    st.session_state.stop_button_clicked = False

    #   if st.button("Stop and Generate Report"):
    #        st.session_state.monitoring = False
    #       st.session_state.stop_button_clicked = True
    #else:
        
    #        st.session_state.monitoring = True
    #        st.session_state.stop_button_clicked = False

    # Display metrics
    placeholder=st.empty()

    # Process records one by one if monitoring is active
    if st.session_state.monitoring:
        st.subheader("Processing Transactions...")
        for index, row in df.iloc[st.session_state.current_index:].iterrows():
            time.sleep(0.10)  # Simulate real-time processing delay

            # Update metrics based on the "Fraudulent" column
            if row["Fraudulent"] == 0:
                st.session_state.genuine_count += 1
                # Count if the account age is less than or equal to 5 years and increment the session state variable genuine_5_AccountAge_Count
                if row["Account Age"] > 0 and row["Account Age"] <= 5:
                    st.session_state.genuine_5_AccountAge_Count += 1
                if row["Account Age"] >6 and row["Account Age"] <= 10:
                        st.session_state.genuine_10_AccountAge_Count += 1
                    
            elif row["Fraudulent"] == 1:
                st.session_state.fraud_count += 1
                # Count if the online purchase is 1 and increment the session state variable fraud_Online_purchase_count
                if row["Online Purchase"] == 1:
                    st.session_state.fraud_Online_purchase_count += 1
                # Count if the first purchase is 1 and increment the session state variable fraud_first_purchase_count
                if row["First Purchase"] == 1:
                    st.session_state.fraud_first_purchase_count += 1

                # Display the current fraudulent transaction in a scrollable container
                #with st.container():
                 #   st.write(f"**Processing Fraudulent Transaction:**")
                 #   st.write(f"Transaction Amount: {row['Transaction Amount']}")
                 #   st.write(f"Customer Age: {row['Customer Age']}")
                 #   st.write(f"Account Age: {row['Account Age']}")
                 #   st.write(f"Online Purchase: {row['Online Purchase']}")
                 #   st.write(f"First Purchase: {row['First Purchase']}")

            # Update the placeholders with the new metrics
            with placeholder.container():
            #st.subheader("Live Metrics")
                col1, col2 = st.columns(2)# top and bottom columns in a grid layout with border color blue  
                col3, col5 = st.columns(2)
                col4, col6 = st.columns(2)
                with col1:
                    col1.metric(label="Genuine Transactions", value=st.session_state.genuine_count, delta=st.session_state.genuine_count - 1,delta_color="normal",border=True)
                with col2:
                    col2.metric(label="Fraud Transactions", value=st.session_state.fraud_count,delta=st.session_state.fraud_count - 1,delta_color="inverse",border=True)
                with col3:
                    col3.metric(label="Account Age 0-5", value=st.session_state.genuine_5_AccountAge_Count,delta=st.session_state.genuine_5_AccountAge_Count - 1,delta_color="normal",border=True)     
                with col4:
                    col4.metric(label="Account Age 6-10", value=st.session_state.genuine_10_AccountAge_Count,delta=st.session_state.genuine_10_AccountAge_Count - 1,delta_color="normal",border=True)   
                with col5:
                    col5.metric(label="No of Online Purchase ", value=st.session_state.fraud_Online_purchase_count,delta=st.session_state.fraud_Online_purchase_count - 1,delta_color="inverse",border=True) 
                with col6:
                    col6.metric(label="No of First Purchase", value=st.session_state.fraud_first_purchase_count,delta=st.session_state.fraud_first_purchase_count - 1,delta_color="inverse",border=True)   
                    
            # Update the current index in session state
            st.session_state.current_index = index + 1

    # Generate report if monitoring is stopped
    if not st.session_state.monitoring:
        st.subheader("Fraud & Genuine Transactions Report & Anaylsis")
        #st.write("### Genuine Transactions")
        genuine_df = df[df["Fraudulent"] == 0]
        #st.dataframe(genuine_df)

        st.write("### Fraudulent Transactions")
        fraudulent_df = df[df["Fraudulent"] == 1]
        # for each row in the dataframe fraudulent_df, display the transaction amount, customer age, account age, online purchase, and first purchase  
        for index, row in fraudulent_df.head(5).iterrows():
        
            st.write(f"**Transaction Amount:** {row['Transaction Amount']}")
            st.write(f"**Customer Age:** {row['Customer Age']}")
            st.write(f"**Account Age:** {row['Account Age']}")
            st.write(f"**Online Purchase:** {row['Online Purchase']}")
            st.write(f"**First Purchase:** {row['First Purchase']}")
            transaction_details2 = ""+ "Transaction ID:" + str(index) + "   Amount: "  +str(row['Transaction Amount'])   + "  Time: 02:30PM" + "   Location: unknown  " +    "   Previous Transactions: None" + ""
            # make this transaction details2 as global variable to be used in the Gen AI Fraud page
            st.session_state.transaction_details2 = transaction_details2
            st.write(st.session_state.transaction_details2)
            from pages.creditcardfraudllm1 import display_gen_ai_fraud_form
            st.write(display_gen_ai_fraud_form())
       #


        # Create charts
        st.subheader("Charts")
        st.write("Genuine Transactions vs Transaction Amount and Customer Age, Account Age 0-5 and 6-10")
        
        # Chart 1: Genuine Transactions vs Transaction Amount and Customer Age, Account Age 0-5 and 6-10
        genuine_df = df[df["Fraudulent"] == 0]
        genuine_df["Account Age Group"] = genuine_df["Account Age"].apply(
            lambda x: "0-5" if x <= 5 else "6-10" if x <= 10 else "10+"
        )
        fig1_data = genuine_df.groupby("Account Age Group").agg(
            {"Transaction Amount": "mean", "Customer Age": "mean"}
        ).reset_index()

        st.bar_chart(
            data=fig1_data,
            x="Account Age Group",
            y=["Transaction Amount", "Customer Age"],
            use_container_width=True,
        )
        
        
        st.write("Fraudulent Transactions vs Transaction Amount, Online Purchase, and First Purchase")
        # Chart 2: Fraudulent Transactions vs Transaction Amount, Online Purchase, and First Purchase
        fraudulent_df = df[df["Fraudulent"] == 1]
        fraudulent_df["Purchase Type"] = fraudulent_df.apply(
            lambda row: "Online" if row["Online Purchase"] == 1 else "In-Store", axis=1
        )
        fig2_data = fraudulent_df.groupby("Purchase Type").agg(
            {"Transaction Amount": "sum", "First Purchase": "sum"}
        ).reset_index()

        st.bar_chart(
            data=fig2_data,
            x="Purchase Type",
            y=["Transaction Amount", "First Purchase"],
            use_container_width=True,
        )
        
    # Display the DataFrame with a download button   
   
# Predict Fraud Page
elif menu_option == "Predict Fraud":
    st.title("Predict Fraud")
    st.write("This page uses the `FindFraud.py` functionality.")
    # Import and call the function from FindFraud.py
    from pages.FindFraud import display_fraud_detection_form
    display_fraud_detection_form()
   
# Gen AI Fraud Page
elif menu_option == "Gen AI Fraud":
    st.title("Gen AI Fraud Detection")
    st.write("This page uses the `creditcardfraudllm1.py` functionality.")
  
    # Set the transaction details variable to None
    st.session_state.transaction_details2 = None

    # Import and call the function from creditcardfraudllm1.py
    from pages.creditcardfraudllm1 import display_gen_ai_fraud_form
    display_gen_ai_fraud_form()
