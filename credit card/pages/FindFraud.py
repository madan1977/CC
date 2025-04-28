import streamlit as st
import subprocess
import json

def display_fraud_detection_form():
    st.title("Fraud Detection Input Form")

    transaction_amount = st.number_input("Transaction Amount", min_value=0)
    transaction_country = st.selectbox("Transaction Country", ["US", "Canada", "UK", "Germany", "France", "Other"])
    if transaction_country == "US":
         transaction_currency = st.selectbox("USD ", ["USD", "CAD", "EUR", "GBP", "AUD"])
    elif transaction_country == "Canada":
         transaction_currency = st.selectbox("CAD ", ["CAD", "USD", "EUR", "GBP", "AUD"])   
    elif transaction_country == "UK":
            transaction_currency = st.selectbox("GBP ", ["GBP", "USD", "CAD", "EUR", "AUD"])    
    elif transaction_country == "Germany":
            transaction_currency = st.selectbox("EUR ", ["EUR", "USD", "CAD", "GBP", "AUD"])
    elif transaction_country == "France":
            transaction_currency = st.selectbox("EUR ", ["EUR", "USD", "CAD", "GBP", "AUD"])
    else:
            transaction_currency = st.selectbox("Other ", ["USD", "CAD", "EUR", "GBP", "AUD"])
    # transaction_currency = st.selectbox("Transaction Currency", ["USD", "CAD", "EUR", "GBP", "AUD"])

    transaction_time = st.text_input("Transaction Time", value="1",min_value=1, max_value=23) 
    transaction_day = st.text_input("Transaction Day", value="Weekday")
    card_issuer = st.selectbox("Card Issuer", ["Visa", "MasterCard", "American Express", "Discover", "Other"])
    card_type = st.selectbox("Card Type", ["Credit", "Debit", "Prepaid"])
    online_purchase = st.radio("Online Purchase", ["Yes", "No"])
    first_purchase = st.radio("First Purchase", ["Yes", "No"])
    customer_age = st.number_input("Customer Age", min_value=1, max_value=99, value=30)
    account_age = st.number_input("Account Age (in years)", min_value=1, max_value=100, value=5)

    if st.button("Submit"):
        # Prepare the data as a JSON object
        data = [
            {
                "Transaction Amount": transaction_amount,
                "Transaction Country": transaction_country,
                "Transaction Currency": transaction_currency,
                "Transaction Time": transaction_time,
                "Transaction Day": transaction_day,
                "Card Issuer": card_issuer,
                "Card Type": card_type,
                "Online Purchase": "1" if online_purchase == "Yes" else "0",
                "First Purchase": "1" if first_purchase == "Yes" else "0",
                "Customer Age": customer_age,
                "Account Age": account_age
            }
        ]

        # Convert the data to a JSON string
        json_data = json.dumps(data)

        # Execute the curl command
        try:
            result = subprocess.run(
                [
                    "curl", "-G", "https://api.akkio.com/api",
                    "-d", "project_key=kgohxpDTneovWEzpeeam/1",
                    "-d", "api_key=3a73b54f-2a42-4077-af9c-ba5c9f83af3f",
                    "--data-urlencode", f"data={json_data}",
                    "-d", "deploy-transforms-only=false"
                ],
                capture_output=True, text=True
            )
            # Display the response
            st.text("Response from server:")
            # Parse the response JSON
            try:
                response_json = json.loads(result.stdout)
                if response_json and "Probability Fraudulent" in response_json[0]:
                    probability_fraudulent = response_json[0]["Probability Fraudulent"]
                    if probability_fraudulent > 0.90:
                        st.warning(f"High probability of fraud detected: {probability_fraudulent:.2f}")
                    else:
                        st.success(f"Transaction seems safe: {probability_fraudulent:.2f}")
                else:
                    st.error("Unexpected response format or missing 'Probability Fraudulent' key.")
            except json.JSONDecodeError:
                st.error("Failed to parse server response as JSON.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    display_fraud_detection_form()
