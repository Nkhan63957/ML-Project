
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide",
    page_icon="🏠"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .price-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get an instant price estimate using our advanced machine learning model</p>', unsafe_allow_html=True)

# Load pre-trained model and preprocessing artifacts
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        onehot_encoder = joblib.load('onehot_encoder.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        selected_features = joblib.load('selected_features.pkl')

        try:
            numeric_medians = joblib.load('numeric_medians.pkl')
        except FileNotFoundError:
            numeric_medians = {
                'Lot Area': 9478.5, 'Overall Qual': 6.0, 'Overall Cond': 5.0, 'Year Built': 1973.0,
                'Year Remod/Add': 1994.0, 'BsmtFin SF 1': 383.5, 'Total Bsmt SF': 991.5,
                '1st Flr SF': 1087.0, '2nd Flr SF': 0.0, 'Gr Liv Area': 1464.0, 'Garage Area': 472.0,
                'Yr Sold': 2008.0, '3Ssn Porch': 0.0, 'Bedroom AbvGr': 3.0, 'TotalSF': 2500.0,
                'Age': 35.0, 'Qual_LivArea': 8760.0, 'Remodeled': 0.0,
                'Lot Frontage': 69.0, 'Mas Vnr Area': 0.0, 'Bsmt Unf SF': 0.0,
                'Bsmt Full Bath': 0.0, 'Full Bath': 2.0, 'TotRms AbvGrd': 6.0,
                'Fireplaces': 0.0, 'Garage Yr Blt': 1973.0, 'Wood Deck SF': 0.0,
                'Open Porch SF': 0.0, 'Enclosed Porch': 0.0, 'Mo Sold': 6.0,
                'Order': 1.0, 'PID': 0.0, 'MS SubClass': 20.0
            }

        return model, scaler, onehot_encoder, target_encoder, selected_features, numeric_medians
    except FileNotFoundError as e:
        st.error(f"❌ Required model file missing: {str(e)}")
        st.info("Please ensure all .pkl files are in the current directory.")
        st.stop()

model, scaler, onehot_encoder, target_encoder, selected_features, numeric_medians = load_models()

# Define categorical options
neighborhood_options = [
    'Names', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt',
    'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge',
    'Timber', 'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn', 'MeadowV',
    'BrDale', 'Veenker', 'NPkVill', 'Blueste'
]
exterior_options = [
    'VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd',
    'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock'
]
alley_options = ['Grvl', 'Pave', 'NA']
bldg_type_options = ['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE']
ms_zoning_options = ['A (agr)', 'C (all)', 'FV', 'I (all)', 'RH', 'RL', 'RM']
street_options = ['Grvl', 'Pave']
bsmt_cond_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
bsmt_exposure_options = ['Gd', 'Av', 'Mn', 'No', 'NA']
bsmt_qual_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
bsmtfin_type1_options = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
bsmtfin_type2_options = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
central_air_options = ['N', 'Y']
condition1_options = ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe', 'RRNn']
condition2_options = ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAn', 'RRNn']
electrical_options = ['FuseA', 'FuseF', 'FuseP', 'Mix', 'SBrkr']
exter_cond_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
exter_qual_options = ['Ex', 'Gd', 'TA', 'Fa']
fence_options = ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']
fireplace_qu_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
foundation_options = ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood']
functional_options = ['Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'Sal', 'Typ']
garage_cond_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
garage_finish_options = ['Fin', 'RFn', 'Unf', 'NA']
garage_qual_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
garage_type_options = ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA']
heating_options = ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall']
heating_qc_options = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
house_style_options = ['1.5Fin', '1.5Unf', '1Story', '2.5Unf', '2Story', 'SFoyer', 'SLvl']
kitchen_qual_options = ['Ex', 'Gd', 'TA', 'Fa']
land_contour_options = ['Bnk', 'HLS', 'Low', 'Lvl']
land_slope_options = ['Gtl', 'Mod', 'Sev']
lot_config_options = ['Corner', 'CulDSac', 'FR2', 'FR3', 'Inside']
lot_shape_options = ['IR1', 'IR2', 'IR3', 'Reg']
mas_vnr_type_options = ['BrkCmn', 'BrkFace', 'CBlock', 'Stone']
misc_feature_options = ['Gar2', 'Othr', 'Shed', 'NA']
paved_drive_options = ['N', 'P', 'Y']
pool_qc_options = ['Ex', 'Gd', 'TA', 'Fa', 'NA']
roof_matl_options = ['CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl']
roof_style_options = ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed']
sale_condition_options = ['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Normal', 'Partial']
sale_type_options = ['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'VWD', 'WD ']
utilities_options = ['AllPub', 'NoSewr']

# Sidebar input form
st.sidebar.markdown("### 🏡 Property Details")
st.sidebar.markdown("Fill in the details below to get your price prediction:")

with st.sidebar.form("house_form"):
    st.markdown("#### 📏 **Basic Information**")
    lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=9600, step=100)
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=6)
    overall_cond = st.slider("Overall Condition (1-10)", min_value=1, max_value=10, value=5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    year_remod_add = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000)

    st.markdown("#### 🏠 **Living Spaces**")
    bsmtfin_sf_1 = st.number_input("Basement Finished Area (sq ft)", min_value=0.0, value=0.0, step=10.0)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0.0, value=1000.0, step=10.0)
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=0, value=1200, step=10)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, value=0, step=10)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=0, value=1200, step=10)
    bedroom_abvgr = st.number_input("Bedrooms Above Ground", min_value=0, value=3, step=1)

    st.markdown("#### 🚗 **Garage & Outdoor**")
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0.0, value=400.0, step=10.0)
    ssn_porch = st.number_input("3-Season Porch Area (sq ft)", min_value=0, value=0, step=10)

    st.markdown("#### 🏘️ **Location & Type**")
    neighborhood = st.selectbox("Neighborhood", options=neighborhood_options, index=0)
    yr_sold = st.number_input("Year Sold", min_value=2000, max_value=2025, value=2010)

    with st.expander("🔧 Advanced Options"):
        st.markdown("#### **Exterior**")
        exterior_1st = st.selectbox("Exterior 1st", options=exterior_options, index=0)
        exterior_2nd = st.selectbox("Exterior 2nd", options=exterior_options, index=0)
        exter_cond = st.selectbox("Exterior Condition", options=exter_cond_options, index=2)
        exter_qual = st.selectbox("Exterior Quality", options=exter_qual_options, index=2)

        st.markdown("#### **Property Details**")
        alley = st.selectbox("Alley Access", options=alley_options, index=2)
        bldg_type = st.selectbox("Building Type", options=bldg_type_options, index=0)
        ms_zoning = st.selectbox("MS Zoning", options=ms_zoning_options, index=5)
        street = st.selectbox("Street", options=street_options, index=1)

        st.markdown("#### **Basement**")
        bsmt_cond = st.selectbox("Basement Condition", options=bsmt_cond_options, index=5)
        bsmt_exposure = st.selectbox("Basement Exposure", options=bsmt_exposure_options, index=4)
        bsmt_qual = st.selectbox("Basement Quality", options=bsmt_qual_options, index=5)
        bsmtfin_type1 = st.selectbox("Basement Finish Type 1", options=bsmtfin_type1_options, index=6)
        bsmtfin_type2 = st.selectbox("Basement Finish Type 2", options=bsmtfin_type2_options, index=6)

        st.markdown("#### **Utilities & Systems**")
        central_air = st.selectbox("Central Air", options=central_air_options, index=1)
        condition_1 = st.selectbox("Condition 1", options=condition1_options, index=2)
        condition_2 = st.selectbox("Condition 2", options=condition2_options, index=2)
        electrical = st.selectbox("Electrical", options=electrical_options, index=4)
        heating = st.selectbox("Heating", options=heating_options, index=1)
        heating_qc = st.selectbox("Heating Quality and Condition", options=heating_qc_options, index=2)

        st.markdown("#### **Additional Features**")
        fence = st.selectbox("Fence", options=fence_options, index=4)
        fireplace_qu = st.selectbox("Fireplace Quality", options=fireplace_qu_options, index=5)
        foundation = st.selectbox("Foundation", options=foundation_options, index=2)
        functional = st.selectbox("Functional", options=functional_options, index=6)
        garage_cond = st.selectbox("Garage Condition", options=garage_cond_options, index=5)
        garage_finish = st.selectbox("Garage Finish", options=garage_finish_options, index=3)
        garage_qual = st.selectbox("Garage Quality", options=garage_qual_options, index=5)
        garage_type = st.selectbox("Garage Type", options=garage_type_options, index=6)
        house_style = st.selectbox("House Style", options=house_style_options, index=4)
        kitchen_qual = st.selectbox("Kitchen Quality", options=kitchen_qual_options, index=2)
        land_contour = st.selectbox("Land Contour", options=land_contour_options, index=3)
        land_slope = st.selectbox("Land Slope", options=land_slope_options, index=0)
        lot_config = st.selectbox("Lot Configuration", options=lot_config_options, index=4)
        lot_shape = st.selectbox("Lot Shape", options=lot_shape_options, index=0)
        mas_vnr_type = st.selectbox("Masonry Veneer Type", options=mas_vnr_type_options, index=1)
        misc_feature = st.selectbox("Miscellaneous Feature", options=misc_feature_options, index=3)
        paved_drive = st.selectbox("Paved Drive", options=paved_drive_options, index=2)
        pool_qc = st.selectbox("Pool Quality", options=pool_qc_options, index=4)
        roof_matl = st.selectbox("Roof Material", options=roof_matl_options, index=0)
        roof_style = st.selectbox("Roof Style", options=roof_style_options, index=1)
        sale_condition = st.selectbox("Sale Condition", options=sale_condition_options, index=4)
        sale_type = st.selectbox("Sale Type", options=sale_type_options, index=9)
        utilities = st.selectbox("Utilities", options=utilities_options, index=0)

    submit_button = st.form_submit_button("🔮 Predict Price", use_container_width=True)

# Preprocessing function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Feature engineering
    df['TotalSF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF'] + df['Garage Area']
    df['Age'] = df['Yr Sold'] - df['Year Built']
    df['Qual_LivArea'] = df['Overall Qual'] * df['Gr Liv Area']
    df['Remodeled'] = (df['Year Remod/Add'] > df['Year Built']).astype(int)

    # Define columns
    numeric_cols = [
        'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
        'BsmtFin SF 1', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Gr Liv Area',
        'Garage Area', 'Yr Sold', 'TotalSF', 'Age', 'Qual_LivArea', 'Remodeled',
        '3Ssn Porch', 'Bedroom AbvGr',
        'Lot Frontage', 'Mas Vnr Area', 'Bsmt Unf SF', 'Bsmt Full Bath', 'Full Bath',
        'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Wood Deck SF', 'Open Porch SF',
        'Enclosed Porch', 'Mo Sold', 'Order', 'PID', 'MS SubClass', 'Half Bath',
        'Bsmt Half Bath', 'Screen Porch', 'Low Qual Fin SF', 'Garage Cars', 'BsmtFin SF 2',
        'Misc Val', 'Kitchen AbvGr', 'Pool Area'
    ]
    high_cardinality_cols = ['Neighborhood', 'Exterior 1st', 'Exterior 2nd']
    low_cardinality_cols = [
        'MS Zoning', 'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities',
        'Lot Config', 'Land Slope', 'Condition 1', 'Condition 2', 'Bldg Type',
        'House Style', 'Roof Style', 'Roof Matl', 'Mas Vnr Type', 'Exter Qual',
        'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
        'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating', 'Heating QC', 'Central Air',
        'Electrical', 'Kitchen Qual', 'Functional', 'Fireplace Qu', 'Garage Type',
        'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive', 'Pool QC',
        'Fence', 'Misc Feature', 'Sale Type', 'Sale Condition'
    ]

    # Map 'None' to 'NA' for relevant categorical columns
    na_columns = ['Alley', 'Bsmt Cond', 'Bsmt Exposure', 'Bsmt Qual', 'BsmtFin Type 1',
                  'BsmtFin Type 2', 'Fence', 'Fireplace Qu', 'Garage Cond', 'Garage Finish',
                  'Garage Qual', 'Garage Type', 'Pool QC', 'Misc Feature']
    for col in na_columns:
        if col in df.columns:
            df[col] = df[col].replace('None', 'NA')

    # Fill missing or NaN values in categorical columns
    df[low_cardinality_cols + high_cardinality_cols] = df[low_cardinality_cols + high_cardinality_cols].fillna('NA')

    # One-hot encoding for low-cardinality columns
    if low_cardinality_cols:
        try:
            encoded_cols = onehot_encoder.transform(df[low_cardinality_cols])
            encoded_df = pd.DataFrame(
                encoded_cols,
                columns=onehot_encoder.get_feature_names_out(low_cardinality_cols),
                index=df.index
            )
            df = df.drop(columns=low_cardinality_cols).join(encoded_df)
        except Exception as e:
            st.error(f"Error in one-hot encoding: {str(e)}")
            st.stop()

    # Add Alley for missing columns
    if 'Alley' in data and data['Alley'] == 'NA':
        df['Alley_Grvl'] = 0.0
        df['Alley_Pave'] = 0.0

    # Ensure all expected one-hot columns exist
    expected_onehot_features = onehot_encoder.get_feature_names_out(low_cardinality_cols)
    for col in expected_onehot_features:
        if col not in df.columns:
            df[col] = 0.0

    # Add missing numerical columns with default values
    default_values = {
        'Lot Frontage': 69.0, 'Mas Vnr Area': 0.0, 'Bsmt Unf SF': 0.0,
        'Bsmt Full Bath': 0.0, 'Full Bath': 2.0, 'TotRms AbvGrd': 6.0,
        'Fireplaces': 0.0, 'Garage Yr Blt': df['Year Built'].iloc[0] if 'Year Built' in df else 1973.0,
        'Wood Deck SF': 0.0, 'Open Porch SF': 0.0, 'Enclosed Porch': 0.0,
        'Mo Sold': 6.0, 'Order': 1.0, 'PID': 0.0, 'MS SubClass': 20.0, 'Half Bath': 0.0,
        'Bsmt Half Bath': 0.0, 'Screen Porch': 0.0, 'Low Qual Fin SF': 0.0, 'Garage Cars': 2.0,
        'BsmtFin SF 2': 0.0, 'Misc Val': 0.0, 'Kitchen AbvGr': 0.0, 'Pool Area': 100.0
    }
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = default_values.get(col, 0.0)

    # Imputation for numerical columns
    imputer = KNNImputer(n_neighbors=3)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Target encoding for high-cardinality columns
    try:
        df[high_cardinality_cols] = target_encoder.transform(df[high_cardinality_cols])
    except Exception as e:
        st.error(f"Error in target encoding: {str(e)}")
        st.stop()

    # Scaling
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

    # Ensure all one-hot encoded columns are present
    all_onehot_cols = onehot_encoder.get_feature_names_out(low_cardinality_cols)
    for col in all_onehot_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if not submit_button:
        st.markdown("### 👈 Enter your house details in the sidebar")
        st.info("Fill out the form in the sidebar and click 'Predict Price' to get your house price estimate!")

        # Show some sample statistics or features
        st.markdown("### 📊 Model Information")
        st.markdown(f"- **Features used**: {len(selected_features)} carefully selected features")
        st.markdown("- **Model type**: XGBoost Regressor")
        st.markdown("- **Accuracy**: Trained on comprehensive housing dataset")

with col2:
    if not submit_button:
        st.markdown("### 🎯 Quick Tips")
        st.markdown("""
        - **Overall Quality**: Rate 1-10 based on materials and finish
        - **Year Built**: Original construction year
        - **Living Area**: Above ground square footage
        - **Neighborhood**: Significantly impacts price
        - **Advanced Options**: Click to expand for detailed features
        """)

# Prediction logic
if submit_button:
    input_data = {
        'Lot Area': lot_area, 'Overall Qual': overall_qual, 'Overall Cond': overall_cond,
        'Year Built': year_built, 'Year Remod/Add': year_remod_add, 'BsmtFin SF 1': bsmtfin_sf_1,
        'Total Bsmt SF': total_bsmt_sf, '1st Flr SF': first_flr_sf, '2nd Flr SF': second_flr_sf,
        'Gr Liv Area': gr_liv_area, 'Garage Area': garage_area, 'Yr Sold': yr_sold,
        'Neighborhood': neighborhood, 'Exterior 1st': exterior_1st, 'Exterior 2nd': exterior_2nd,
        '3Ssn Porch': ssn_porch, 'Bedroom AbvGr': bedroom_abvgr, 'Alley': alley,
        'Bldg Type': bldg_type, 'MS Zoning': ms_zoning, 'Street': street,
        'Bsmt Cond': bsmt_cond, 'Bsmt Exposure': bsmt_exposure, 'Bsmt Qual': bsmt_qual,
        'BsmtFin Type 1': bsmtfin_type1, 'BsmtFin Type 2': bsmtfin_type2,
        'Central Air': central_air, 'Condition 1': condition_1, 'Condition 2': condition_2,
        'Electrical': electrical, 'Exter Cond': exter_cond, 'Exter Qual': exter_qual,
        'Fence': fence, 'Fireplace Qu': fireplace_qu, 'Foundation': foundation,
        'Functional': functional, 'Garage Cond': garage_cond, 'Garage Finish': garage_finish,
        'Garage Qual': garage_qual, 'Garage Type': garage_type, 'Heating': heating,
        'Heating QC': heating_qc, 'House Style': house_style, 'Kitchen Qual': kitchen_qual,
        'Land Contour': land_contour, 'Land Slope': land_slope, 'Lot Config': lot_config,
        'Lot Shape': lot_shape, 'Mas Vnr Type': mas_vnr_type, 'Misc Feature': misc_feature,
        'Paved Drive': paved_drive, 'Pool QC': pool_qc, 'Roof Matl': roof_matl,
        'Roof Style': roof_style, 'Sale Condition': sale_condition, 'Sale Type': sale_type,
        'Utilities': utilities
    }

    # Preprocess input and predict
    try:
        with st.spinner('🔄 Processing your input and generating prediction...'):
          processed_data = preprocess_input(input_data)

          log_pred = model.predict(processed_data[selected_features])[0]
          pred_price = np.expm1(log_pred)  # Convert log price to actual price

          # Display prediction with custom styling
          st.markdown(f"""
          <div class="prediction-box">
              <h2>🎉 Prediction Complete!</h2>
              <p class="price-text">${pred_price:,.0f}</p>
              <p>Estimated House Price</p>
          </div>
          """, unsafe_allow_html=True)
    except Exception as e:
      st.error(f"Error during prediction: {str(e)}")
      # Optional: Add more detailed error information for debugging
      st.write("Please check your input data and try again.")
