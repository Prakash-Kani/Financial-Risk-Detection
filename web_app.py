import streamlit as st
import datetime as dt
import numpy as np
import json
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(page_title = "Financial Risk Detection",
                   page_icon = "💸",
                   layout = "wide",
                   initial_sidebar_state = "expanded",
                   menu_items = None)

@st.cache_resource()

def Defaulter_model():

    model = joblib.load('Defaulter_prediction.joblib.gz')

    return model

clf_model = Defaulter_model()


with open(r'src\Category_Columns_Encoded_Data.json', 'r') as file:
    data = json.load(file)


st.title(":blue[Financial] :red[Risk] :green[Detection]")

col1, col2, col3, col4, col5 = st.columns(5, gap= 'medium')

# Created the required input fields
with col1:
    sk_id_curr = st.number_input('Enter the **CURRENT SK ID**', value =223374.00, min_value= 100000.00, max_value= 456255.00)
    name_education_type = st.selectbox('Select the **Education Type**', data['name_education_type'].keys(), index = 1)
    name_contract_type_app = st.selectbox('Select the **Contract Type** in app', data['name_contract_type_app'].keys(), index = 0)
    name_family_status = st.selectbox('Select the **Family Status**', data['name_family_status'].keys(), index = 3)
    name_portfolio = st.selectbox('Select the **Portfolio**', data['name_portfolio'].keys(), index = 1)
    days_employed = st.number_input('Enter the **Days Employed**', value =1564.0, min_value= 0.00, max_value= 365243.00)
    reg_region_not_live_region = st.selectbox('Select the **reg_region_not_live_region**', [0, 1], index = 0)
    days_last_phone_change = st.number_input('Enter the **Last Phone Changed** in Days', value =1768.0, min_value= 0.00, max_value= 4500.00)
    region_population_relative = st.number_input('Enter the **Region Population Relative**', value =0.046220, min_value= 0.00, max_value= 1.00)

    amt_req_credit_bureau_hour = st.selectbox('Select the **Amount Required to Credit Bureau Hour**', [float(i)  for i in range(5)], index = 0)
    weekday_appr_process_start_prev = st.selectbox('Select the **Weekday approximate process start** in previous application', data['weekday_appr_process_start_prev'].keys(), index = 6)

    

with col2:
    code_gender = st.selectbox('Select the **Gender**', ['M', 'F'], index = 0)
    organization_type = st.selectbox('Select the **Organization Type**', data['organization_type'].keys(), index = 0)
    name_contract_type_prev = st.selectbox('Select the **Contract Type** in prev_app', data['name_contract_type_prev'].keys(), index = 1)
    name_housing_type = st.selectbox('Select the **Housing Type**', data['name_housing_type'].keys(), index = 3)
    channel_type = st.selectbox('Select the **Channel Type**', data['channel_type'].keys(), index = 1)
    days_registration  = st.number_input('Enter the **Days Registration**', value =5353.0, min_value= 2.00, max_value= 25500.00)
    reg_region_not_work_region = st.selectbox('Select the **reg_region_not_work_region**', [0, 1], index = 1)
    ext_source_2 = st.number_input('Enter the **Ext Source 2**', value =0.611304, min_value= 0.00, max_value= 1.00)
    region_rating_client_w_city = st.selectbox('Select the **region rating client w city**', [1, 2, 3], index = 0)
    amt_req_credit_bureau_day = st.selectbox('Select the **Amount Required to Credit Bureau Week**', [float(i)  for i in range(8)], index = 0)
    weekday_appr_process_start_app = st.selectbox('Select the **Weekday approximate process start** in application', data['weekday_appr_process_start_app'].keys(), index =6)


with col3:
    age = st.number_input('Enter Your **Age**', value= 38.0, min_value=15.0, max_value=100.0)
    occupation_type = st.selectbox('Select the **Occupation Type**', data['occupation_type'].keys(), index = 3)
    flag_last_appl_per_contract = st.selectbox('Select the **flag for last app**', data['flag_last_appl_per_contract'].keys(), index =0)
    name_type_suite_app = st.selectbox('Select the **Type Suite**', data['name_type_suite_app'].keys(), index = 0)
    name_yield_group = st.selectbox('Select the **Yield Group**', data['name_yield_group'].keys(), index = 1)
    days_id_publish = st.number_input('Enter the **ID Publish** in Days', value =3225.0, min_value= 2.00, max_value= 8000.00)
    live_region_not_work_region = st.selectbox('Select the **live_region_not_work_region**', [0, 1], index = 1)
    ext_source_3= st.number_input('Enter the **Ext Source 3**', value =0.089044, min_value= 0.00, max_value= 1.00)
    obs_30_cnt_social_circle = st.selectbox('Select the **obs_30_cnt_social_circle**', [5.,  2.,   1.,   0.,   4.,   8.,   7.,   3.,   6.,  12.,   9.,  10.,  13.,  11.,  14.,  22.,  16.,  15.,  17.,  20.,  25.,  19., 18.,  21.,  24.,  23.,  28.,  26.,  29.,  27.,  47., 348.,  30.], index = 0)

    amt_req_credit_bureau_week = st.selectbox('Select the **Amount Required to Credit Bureau Week**', [float(i)  for i in range(9)], index = 0)
    hour_appr_process_start_app = st.selectbox('Select the **Hours approximate process start** in application', [i for i in range(23)], index = 10)

    

with col4:
    cnt_children = st.number_input('Enter the **Childern Count**', value =1.00, min_value= 0.00, max_value= 20.00)
    name_income_type = st.selectbox('Select the **Income Type**', data['name_income_type'].keys(), index = 0)
    amt_application = st.number_input('Enter the **Amount of the Application**', value = 450000.0, min_value = 0.0)
    name_payment_type = st.selectbox('Select the **Payment Type**', data['name_payment_type'].keys(), index =0)
    product_combination = st.selectbox('Select the **Product Combination**', data['product_combination'].keys(), index = 15)
    days_decision = st.number_input('Enter the **Days Decsion**', value =1127.0, min_value= 1.00, max_value= 1900.00)
    reg_city_not_live_city = st.selectbox('Select the **reg_city_not_live_city**', [0, 1], index =0)
    sellerplace_area = st.number_input('Enter the **Sellerplace Area**', value =-1.0, min_value= -1.00, max_value= 4000000.0)
    def_30_cnt_social_circle = st.selectbox('Select the **def_30_cnt_social_circle**',[0.,  1., 2.,  3.,  4.,  5.,  6.,  7., 8., 34.], index = 0)

    amt_req_credit_bureau_mon = st.selectbox('Select the **Amount Required to Credit Bureau Month**', [float(i)  for i in range(33)], index = 1)
    hour_appr_process_start_prev = st.selectbox('Select the **Hours approximate process start** in previous application', [i for i in range(23)], index = 15)
#   

with col5:
    flag_own_car = st.selectbox('Select the **Flag for Own Car**', [0, 1], index = 0)
    amt_income_total = st.number_input('Enter the **Aunnal Income**', value =360000.00, min_value= 20000.00, max_value= 117000000.00)
    name_contract_status = st.selectbox('Select the **Contract Status**', data['name_contract_status'].keys(), index = 2)
    name_client_type = st.selectbox('Select the **Client Type**', data['name_client_type'].keys(), index = 1)
    nflag_last_appl_in_day = st.selectbox('Select the **nflag_last_appl_in_day**', [0, 1], index =1)
    amt_goods_price = st.number_input('Enter the **Goods Price**', value =900000.00, min_value= 40500.00, max_value= 4050000.00)
    reg_city_not_work_city =st.selectbox('Select the **reg_city_not_work_city**', [0, 1], index = 1)
    obs_30_cnt_social_circle = st.selectbox('Select the **obs_30_cnt_social_circle**', [  2.,   1.,   0.,   4.,   8.,   7.,   3.,   6.,   5.,  12.,   9.,  10.,  13.,  11.,  14.,  22.,  16.,  15.,  17.,  20.,  25.,  19., 18.,  21.,  24.,  23.,  28.,  26.,  29.,  27.,  47., 348.,  30.], index = 8)
    amt_req_credit_bureau_qrt = st.selectbox('Select the **Amount Required to Credit Bureau Quarter**', [float(i)  for i in range(9)], index = 0)
  
    amt_req_credit_bureau_year = st.selectbox('Select the **Amount Required to Credit Bureau Year**', [float(i)  for i in range(26)], index = 0)



# Converting categorical data into numerical data with the help Category_columns Encoded data 

test_data = [sk_id_curr, data['name_contract_type_app'][name_contract_type_app], data['code_gender'][code_gender],
            flag_own_car, cnt_children, amt_income_total, amt_goods_price,
            data['name_type_suite_app'][name_type_suite_app], data['name_income_type'][name_income_type], 
            data['name_education_type'][name_education_type], data['name_family_status'][name_family_status],
            data['name_housing_type'][name_housing_type],
            region_population_relative,
            days_employed, days_registration, days_id_publish,
            data['occupation_type'][occupation_type], region_rating_client_w_city,
            data['weekday_appr_process_start_app'][weekday_appr_process_start_app], hour_appr_process_start_app,
            reg_region_not_live_region, reg_region_not_work_region,
            live_region_not_work_region, reg_city_not_live_city,
            reg_city_not_work_city, data['organization_type'][organization_type], ext_source_2,
            ext_source_3, obs_30_cnt_social_circle, def_30_cnt_social_circle,
            days_last_phone_change, amt_req_credit_bureau_hour,
            amt_req_credit_bureau_day, amt_req_credit_bureau_week,
            amt_req_credit_bureau_mon, amt_req_credit_bureau_qrt,
            amt_req_credit_bureau_year, age, data['name_contract_type_prev'][name_contract_type_prev],
            amt_application, data['weekday_appr_process_start_prev'][weekday_appr_process_start_prev],
            hour_appr_process_start_prev, data['flag_last_appl_per_contract'][flag_last_appl_per_contract],
            nflag_last_appl_in_day, data['name_contract_status'][name_contract_status], days_decision,
            data['name_payment_type'][name_payment_type], 
            data['name_client_type'][name_client_type], data['name_portfolio'][name_portfolio],
            data['channel_type'][channel_type], sellerplace_area, data['name_yield_group'][name_yield_group],
            data['product_combination'][product_combination]]
st.markdown('Click below button to predict the :red[Defaluter] or :green[Repayer]')

if st.button('**Predict**'):
    pred = clf_model.predict([test_data])

    if pred[0]==1:
        st.markdown("### :bule[Prediction :] :red[**Defaluter**]")
    else:
        st.markdown("### :bule[Prediction :] :green[**Repayer**]")
