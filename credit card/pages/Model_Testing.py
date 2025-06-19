
def model_testing_app():
    import streamlit as st
    import pandas as pd
    import pickle
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from bilstm_model import AttentionLayer # Assuming you have a function to build your Bi-LSTM model
    st.title("Model Testing: Traditional vs Bi-LSTM")
    # Upload test data
    uploaded_file = st.file_uploader("Upload Excel file with test data", type=["xlsx"])
    if uploaded_file:
        test_df = pd.read_excel(uploaded_file)
        st.write("Test Data Preview:", test_df.head())

        # Select target column
        target_col = st.selectbox("Select target column", test_df.columns)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Upload traditional model
        trad_model_file = st.file_uploader("Upload Traditional Model (.pkl)", type=["pkl"])
        # Upload Bi-LSTM model
        #bilstm_model_file = st.file_uploader("Upload Bi-LSTM Model (.h5)", type=["h5"])
        #trad_model_file = "pages/classical_model.pkl"
        bilstm_model_file = "credit card/pages/bilstm_model.h5" 
        if trad_model_file and bilstm_model_file:
            # Load traditional model
            trad_model = pickle.load(trad_model_file)
            # Align columns
            expected_features = trad_model.feature_names_in_
            X_test = X_test.reindex(columns=expected_features, fill_value=0)

            # Ensure all columns are numeric
            X_test = X_test.apply(pd.to_numeric, errors='coerce')
            #assert not X_test.isnull().any().any(), "X_test contains NaNs after conversion!"
            # After all preprocessing and before prediction:
            X_test = X_test.fillna(0)  # or use another strategy if 0 is not appropriate

            # Now the assertion should pass
            assert not X_test.isnull().any().any(), "X_test still contains NaNs!"

            # Predict
            y_pred_trad = trad_model.predict(X_test)
            st.subheader("Traditional Model Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_trad))
            st.write("Precision:", precision_score(y_test, y_pred_trad, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred_trad, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred_trad, average='weighted'))
            st.text(classification_report(y_test, y_pred_trad))

            # Load Bi-LSTM model
            bilstm_model = tf.keras.models.load_model(bilstm_model_file)
            # Reshape X_test for LSTM if needed
            X_bilstm = X_test.values
            if len(X_bilstm.shape) == 2:
                X_bilstm = X_bilstm.reshape((X_bilstm.shape[0], 1, X_bilstm.shape[1]))
            y_pred_bilstm = bilstm_model.predict(X_bilstm)
            # If output is probabilities, get class labels
            if y_pred_bilstm.shape[1] == 1:
                y_pred_bilstm = (y_pred_bilstm > 0.5).astype(int).flatten()
            else:
                y_pred_bilstm = y_pred_bilstm.argmax(axis=1)
            st.subheader("Bi-LSTM Model Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_bilstm))
            st.write("Precision:", precision_score(y_test, y_pred_bilstm, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred_bilstm, average='weighted'))
            st.write("F1 Score:", f1_score(y_test, y_pred_bilstm, average='weighted'))
            st.text(classification_report(y_test, y_pred_bilstm))

            # Compare models
            st.subheader("Model Comparison")
            trad_acc = accuracy_score(y_test, y_pred_trad)
            bilstm_acc = accuracy_score(y_test, y_pred_bilstm)
            better = "Traditional Model" if trad_acc > bilstm_acc else "Bi-LSTM Model"
            st.write(f"Better Model: **{better}**")


            # Upload test data
            uploaded_file = st.file_uploader("Upload Excel file with test data", type=["xlsx"])
            if uploaded_file:
                test_df = pd.read_excel(uploaded_file)
                st.write("Test Data Preview:", test_df.head())

                # Select target column
                target_col = st.selectbox("Select target column", test_df.columns)
                X_test = test_df.drop(columns=[target_col])
                y_test = test_df[target_col]

                #trad_model_file = "pages/classical_model.pkl" 
                trad_model_file = st.file_uploader("Upload Traditional Model (.pkl)", type=["pkl"])
                bilstm_model_file = "credit card/pages/bilstm_model.h5" 
                #bilstm_model_file = st.file_uploader("Upload sBi-LSTM Model (.h5)", type=["h5"])

                if trad_model_file and bilstm_model_file:
                    # Load traditional model
                    trad_model = pickle.load(trad_model_file)
                    y_pred_trad = trad_model.predict(X_test)
                    st.subheader("Traditional Model Results")
                    st.write("Accuracy:", accuracy_score(y_test, y_pred_trad))
                    st.write("Precision:", precision_score(y_test, y_pred_trad, average='weighted'))
                    st.write("Recall:", recall_score(y_test, y_pred_trad, average='weighted'))
                    st.write("F1 Score:", f1_score(y_test, y_pred_trad, average='weighted'))
                    st.text(classification_report(y_test, y_pred_trad))

                    # Load Bi-LSTM model
                    bilstm_model = tf.keras.models.load_model(bilstm_model_file,custom_objects={'AttentionLayer': AttentionLayer})
                    # Reshape X_test for LSTM if needed
                    X_bilstm = X_test.values
                    if len(X_bilstm.shape) == 2:
                        X_bilstm = X_bilstm.reshape((X_bilstm.shape[0], 1, X_bilstm.shape[1]))
                    y_pred_bilstm = bilstm_model.predict(X_bilstm)
                    # If output is probabilities, get class labels
                    if y_pred_bilstm.shape[1] == 1:
                        y_pred_bilstm = (y_pred_bilstm > 0.5).astype(int).flatten()
                    else:
                        y_pred_bilstm = y_pred_bilstm.argmax(axis=1)
                    st.subheader("Bi-LSTM Model Results")
                    st.write("Accuracy:", accuracy_score(y_test, y_pred_bilstm))
                    st.write("Precision:", precision_score(y_test, y_pred_bilstm, average='weighted'))
                    st.write("Recall:", recall_score(y_test, y_pred_bilstm, average='weighted'))
                    st.write("F1 Score:", f1_score(y_test, y_pred_bilstm, average='weighted'))
                    st.text(classification_report(y_test, y_pred_bilstm))

                    # Compare models
                    st.subheader("Model Comparison")
                    trad_acc = accuracy_score(y_test, y_pred_trad)
                    bilstm_acc = accuracy_score(y_test, y_pred_bilstm)
                    better = "Traditional Model" if trad_acc > bilstm_acc else "Bi-LSTM Model"
                    st.write(f"Better Model: **{better}**")
if __name__ == "__main__":
    model_testing_app()
