import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from catboost import Pool
from joblib import load


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv">Download csv file</a>'
    return href

@st.cache
def get_model():
    clf = load(Path('nbs') / 'clf.joblib')
    return clf

@st.cache
def get_explainer():
    explainer = load(Path('nbs') / 'explainer.joblib')
    return explainer


clf = get_model()
explainer = get_explainer()

st.write('## Perform bulk predictions from a CSV file')
file_upload = st.file_uploader('', type="csv")
cat_cols = ['Geography', 'Gender']
num_cols = ['CreditScore', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary']
if file_upload and st.button('Get predictions'):
    df_uploaded = pd.read_csv(file_upload)
    cols_uploaded = set(df_uploaded.columns)
    if not set(cat_cols + num_cols).issubset(cols_uploaded):
        st.error(f"""
Uploaded CSV file must contain all of the following columns:  
\n{cat_cols + num_cols}.  
\nThe columns that are missing:  
\n{set(cat_cols + num_cols) - cols_uploaded}
""")
        st.stop()
    X = df_uploaded[cat_cols + num_cols]
    y_prob = clf.predict_proba(X)
    y_pred = clf.predict(X)
    df_uploaded['Churn Probability'] = y_prob[:, 1]
    df_uploaded['Churn Prediction'] = y_pred
    # reorder columns (not required)
    df_uploaded = df_uploaded[['Churn Prediction', 'Churn Probability'] + list(cols_uploaded)]
    st.dataframe(df_uploaded)
    st.markdown(get_table_download_link(df_uploaded), unsafe_allow_html=True)

df = pd.read_csv(Path('data') / 'Churn.csv')
st.write(f"## Or enter customer's info to predict's their chances of churning:")
customer_data_dict = {}
for col in cat_cols:
    val = st.selectbox(f'Select {col}', options=df[col].unique())
    customer_data_dict[col] = val

for col in num_cols:
    val = st.number_input(f'Select {col} (min: {int(df[col].min())}, max: {int(df[col].max())})',
                          value=int(df[col].mean()),
                          min_value=int(df[col].min()),
                          max_value=int(df[col].max()))
    customer_data_dict[col] = val

customer_data = pd.Series(customer_data_dict)
customer_data = pd.DataFrame(customer_data).transpose()
if st.button('Get customer prediction'):
    y_prob = clf.predict_proba(customer_data)
    y_pred = y_prob[:, 1] >= 0.5
    st.write(f'## Probability of churn for this customer: {int(100 * y_prob[0][1])}%')
    st.write('### Shapley values')
    shap_values = explainer.shap_values(Pool(customer_data, [y_pred], cat_features=['Geography', 'Gender']))
    fig = shap.force_plot(explainer.expected_value,
                          shap_values[0, :],
                          customer_data.iloc[0, :],
                          matplotlib=True,
                          show=False,
                          figsize=(16, 5))
    st.pyplot(fig=fig, bbox_inches='tight')
    plt.clf()
