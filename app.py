import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(layout="centered")

# -----------------------------
# Load model (Handling Load Error)
# -----------------------------
try:
    # Attempt to load the model from your specific path
    model = joblib.load("/Users/klevizane/Documents/DS_Quant/03_Proyectos/DS/car_predict_price/Nuevo/Model/final_model.pkl")
    MODEL_LOADED = True
except Exception as e:
    # Use a mock model if loading fails
    st.warning(f"WARNING: Could not load the model. Using simulated prediction. Error: {e}")
    MODEL_LOADED = False
    class MockModel:
        def predict(self, df):
            # Simulates a logarithmic prediction (e.g., log1p(50000) ≈ 10.819)
            return np.array([10.819])
    model = MockModel()

# -----------------------------
# Mappings and Data
# -----------------------------
model_mapping = {
    'Aston Martin': ['DB11', 'DBX707', 'Vantage'], 
    'Audi': ['A3', 'A4', 'A4 allroad', 'A5', 'A6', 'A6 allroad', 'A7', 'A8', 'Q3', 'Q4 Sportback e-tron', 'Q4 e-tron', 'Q5', 'Q5 Sportback', 'Q7', 'Q8', 'Q8 Sportback e-tron', 'Q8 e-tron', 'R8', 'RS 3', 'RS 5', 'RS 6', 'RS 7', 'RS Q8', 'RS e-tron GT', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'SQ5', 'SQ5 Sportback', 'SQ7', 'SQ8', 'SQ8 Sportback e-tron', 'SQ8 e-tron', 'TT', 'TTS', 'e-tron', 'e-tron GT', 'e-tron S', 'e-tron S Sportback', 'e-tron Sportback'], 
    'BMW': ['2 Series', '2 Series Gran Coupe', '3 Series', '4 Series', '4 Series Gran Coupe', '5 Series', '7 Series', '8 Series', '8 Series Gran Coupe', 'ALPINA B8 Gran Coupe', 'ALPINA XB7', 'M2', 'M3', 'M3 CS', 'M4', 'M4 CSL', 'M5', 'M8', 'M8 Gran Coupe', 'X1', 'X2', 'X3', 'X3 M', 'X4', 'X4 M', 'X5', 'X5 M', 'X6', 'X6 M', 'X7', 'XM', 'Z4', 'i4', 'i5', 'i7', 'iX'], 
    'Bentley': ['Bentayga', 'Bentayga EWB', 'Continental', 'Flying Spur'], 
    'Ford': ['Bronco', 'Bronco Sport', 'E-Transit Cargo Van', 'Edge', 'Escape', 'Expedition', 'Explorer', 'F-150', 'F-150 Lightning', 'F-250 Super Duty', 'F-350 Super Duty', 'F-450 Super Duty', 'Maverick', 'Mustang', 'Mustang Mach-E', 'Ranger', 'Transit Cargo Van', 'Transit Connect Cargo Van', 'Transit Connect Passenger Wagon', 'Transit Crew Van', 'Transit Passenger Van'], 
    'Mercedes-Benz': ['AMG GT', 'C-Class', 'CLA-Class', 'CLE', 'CLS-Class', 'E-Class', 'EQB', 'EQE', 'EQE SUV', 'EQS', 'EQS SUV', 'G-Class', 'GLA-Class', 'GLB-Class', 'GLC-Class', 'GLC-Class Coupe', 'GLE-Class', 'GLE-Class Coupe', 'GLS-Class', 'Maybach', 'Maybach EQS SUV', 'Maybach GLS', 'Metris', 'S-Class', 'SL-Class', 'Sprinter', 'eSprinter'], 
    'Nissan': ['ARIYA', 'Altima', 'Armada', 'Frontier', 'GT-R', 'Kicks', 'LEAF', 'Maxima', 'Murano', 'Pathfinder', 'Rogue', 'Sentra', 'Titan', 'Titan XD', 'Versa', 'Z']
}
available_makes = sorted(model_mapping.keys())

model_to_cylinders = {
    # Full cylinder mapping as provided by the user
    'DB11': ['V8'], 'DBX707': ['V8'], 'Vantage': ['V12', 'V8'], 'A3': ['I4'], 'A4': ['I4'], 'A4 allroad': ['I4'], 'A5': ['I4'], 
    'A6': ['I4', 'V6'], 'A6 allroad': ['V6'], 'A7': ['V6'], 'A8': ['V6'], 'Q3': ['I4'], 'Q4 Sportback e-tron': ['V6'], 'Q4 e-tron': ['I4', 'V6'], 
    'Q5': ['I4'], 'Q5 Sportback': ['I4'], 'Q7': ['I4', 'V6'], 'Q8': ['V6'], 'Q8 Sportback e-tron': ['V8', 'V6'], 'Q8 e-tron': ['V8', 'V6'], 
    'R8': ['V10'], 'RS 3': ['I5'], 'RS 5': ['V6'], 'RS 6': ['V8'], 'RS 7': ['V8'], 'RS Q8': ['V8'], 'RS e-tron GT': ['V8'], 'S3': ['I4'], 
    'S4': ['V6'], 'S5': ['V6'], 'S6': ['V6'], 'S7': ['V6'], 'S8': ['V8'], 'SQ5': ['V6'], 'SQ5 Sportback': ['V6'], 'SQ7': ['V8'], 
    'SQ8': ['V8'], 'SQ8 Sportback e-tron': ['V8'], 'SQ8 e-tron': ['V8'], 'TT': ['I4'], 'TTS': ['I4'], 'e-tron': ['V6'], 'e-tron GT': ['V6'], 
    'e-tron S': ['V8'], 'e-tron S Sportback': ['V8'], 'e-tron Sportback': ['V6'], '2 Series': ['I4', 'I6'], '2 Series Gran Coupe': ['I4'], 
    '3 Series': ['I4', 'I6'], '4 Series': ['I4', 'I6'], '4 Series Gran Coupe': ['I4', 'I6'], '5 Series': ['I4', 'I6', 'V8'], 
    '7 Series': ['I6', 'V8'], '8 Series': ['I6', 'V8'], '8 Series Gran Coupe': ['I6', 'V8'], 'ALPINA B8 Gran Coupe': ['V8'], 
    'ALPINA XB7': ['V8'], 'M2': ['I6'], 'M3': ['I6'], 'M3 CS': ['I6'], 'M4': ['I6'], 'M4 CSL': ['I6'], 'M5': ['V8'], 'M8': ['V8'], 
    'M8 Gran Coupe': ['V8'], 'X1': ['I4'], 'X2': ['I4'], 'X3': ['I4', 'I6'], 'X3 M': ['I6'], 'X4': ['I4', 'I6'], 'X4 M': ['I6'], 
    'X5': ['I6', 'V8'], 'X5 M': ['V8'], 'X6': ['I6', 'V8'], 'X6 M': ['V8'], 'X7': ['I6', 'V8'], 'XM': ['V8'], 'Z4': ['I4', 'I6'], 
    'i4': ['I4', 'I6'], 'i5': ['I6', 'V8'], 'i7': ['I6', 'V8'], 'iX': ['V8'], 'Bentayga': ['V6', 'V8', 'W12'], 'Bentayga EWB': ['V8'], 
    'Continental': ['V8', 'W12'], 'Flying Spur': ['V6', 'V8', 'W12'], 'Bronco': ['I4', 'V6'], 'Bronco Sport': ['I3', 'I4'], 
    'E-Transit Cargo Van': ['V6'], 'Edge': ['I4', 'V6'], 'Escape': ['I3', 'I4'], 'Expedition': ['V6'], 'Explorer': ['I4', 'V6'], 
    'F-150': ['V6', 'V8'], 'F-150 Lightning': ['V6', 'V8'], 'F-250 Super Duty': ['V8'], 'F-350 Super Duty': ['V8'], 
    'F-450 Super Duty': ['V8'], 'Maverick': ['I4'], 'Mustang': ['I4', 'V8'], 'Mustang Mach-E': ['I4', 'V6', 'V8'], 'Ranger': ['I4', 'V6'], 
    'Transit Cargo Van': ['V6'], 'Transit Connect Cargo Van': ['I4'], 'Transit Connect Passenger Wagon': ['I4'], 
    'Transit Crew Van': ['V6'], 'Transit Passenger Van': ['V6'], 'AMG GT': ['I6', 'V8'], 'C-Class': ['I4', 'V6', 'V8'], 
    'CLA-Class': ['I4'], 'CLE': ['I4', 'I6'], 'CLS-Class': ['I6'], 'E-Class': ['I4', 'I6', 'V8'], 'EQB': ['I4', 'V6'], 
    'EQE': ['I6', 'V6', 'V8'], 'EQE SUV': ['V6', 'V8'], 'EQS': ['I6', 'V8'], 'EQS SUV': ['I6', 'V8'], 'G-Class': ['V8'], 
    'GLA-Class': ['I4'], 'GLB-Class': ['I4'], 'GLC-Class': ['I4'], 'GLC-Class Coupe': ['I4', 'V6'], 
    'GLE-Class': ['I4', 'I6', 'V8'], 'GLE-Class Coupe': ['I6', 'V8'], 'GLS-Class': ['I6', 'V8'], 'Maybach': ['V12', 'V8'], 
    'Maybach EQS SUV': ['V8'], 'Maybach GLS': ['V8'], 'Metris': ['I4'], 'S-Class': ['I6', 'V8'], 'SL-Class': ['I4', 'V8'], 
    'Sprinter': ['I4'], 'eSprinter': ['I4'], 'ARIYA': ['I4', 'V6'], 'Altima': ['I4'], 'Armada': ['V8'], 'Frontier': ['V6'], 
    'GT-R': ['V6'], 'Kicks': ['I4'], 'LEAF': ['I4'], 'Maxima': ['V6'], 'Murano': ['V6'], 'Pathfinder': ['V6'], 'Rogue': ['I3'], 
    'Sentra': ['I4'], 'Titan': ['V8'], 'Titan XD': ['V8'], 'Versa': ['I4'], 'Z': ['V6']
}

# Brand_Tier Mapping (THE NEW FEATURE)
brand_tier_mapping = {
    'Aston Martin': 'Extreme_Luxury',
    'Bentley': 'Extreme_Luxury',
    'Mercedes-Benz': 'Mid_High_Premium', 
    'BMW': 'Mid_High_Premium',
    'Audi': 'Mid_High_Premium',
    'Ford': 'Mass_Market',
    'Nissan': 'Mass_Market'
}


# -----------------------------
# Sidebar: User input
# -----------------------------
def get_user_input():
    st.sidebar.header("Vehicle Specifications")

    # 1. MAKE - This drives the Brand_Tier assignment
    make = st.sidebar.selectbox('Make', available_makes)
    
    # Automatic Brand_Tier Assignment (The requested functionality)
    brand_tier = brand_tier_mapping.get(make, 'Mass_Market') # Default to Mass_Market
    
    # 2. MODEL - Filtered based on Make
    available_models = model_mapping.get(make, [])
    model_name = st.sidebar.selectbox('Model', available_models)
    
    # 3. CYLINDERS - Filtered based on Model
    available_cylinders = model_to_cylinders.get(model_name, [])
    if not available_cylinders:
        # Fallback to a common list if the model is not mapped
        available_cylinders = ["V6", "I4", "V8", "I6", "I3", "W12", "V10", "V12", "I5"]

    cylinders = st.sidebar.selectbox('Cylinders', available_cylinders)

    # The rest of the inputs
    year = st.sidebar.selectbox('Year', [2024, 2023]) # Available Years
    body_size = st.sidebar.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style = st.sidebar.selectbox('Body Style', [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])
    horsepower = st.sidebar.number_input('Horsepower', min_value=120, max_value=835, step=1, value=300)
    torque = st.sidebar.number_input('Torque', min_value=100, max_value=815, step=1, value=400)
    
    
    # Collected data (NOTE: Only includes the 12 actual features used for training)
    user_data = {
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Body Size": body_size,
        "Body Style": body_style,
        "Cylinders": cylinders, 
        "Engine Aspiration": engine_aspiration,
        "Drivetrain": drivetrain,
        "Transmission": transmission,
        "Horsepower": horsepower,
        "Torque": torque,
        "Brand_Tier": brand_tier, # AUTOMATICALLY ASSIGNED FEATURE
    }
    return user_data

# -----------------------------
# Main Layout and Prediction Logic
# -----------------------------

st.markdown("<h1 style='text-align: center;'>Vehicle Price Prediction </h1>", unsafe_allow_html=True)
st.markdown("---")

# Get user input (placed in the sidebar by get_user_input)
user_data = get_user_input()

# Function to prepare the input into a DataFrame
def prepare_input(data):
    # Define the column order exactly as the model expects (12 columns)
    feature_list = [
        "Make", "Model", "Year", "Body Size", "Body Style", "Cylinders", 
        "Engine Aspiration", "Drivetrain", "Transmission", "Horsepower", 
        "Torque", "Brand_Tier"
    ]
    
    # Create the DataFrame
    input_df = pd.DataFrame([data], columns=feature_list)
    return input_df


# Prediction button centered in the main area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Predict Vehicle Price", use_container_width=True):
        
        if not MODEL_LOADED:
            st.error("Model not loaded correctly. The prediction shown is a simulated value.")

        # Prepare the input data
        input_df = prepare_input(user_data)
        
        try:
            # Make the prediction (assuming it's in the logarithmic scale)
            log_prediction = model.predict(input_df)
            
            # Revert the logarithmic transformation (e^x - 1)
            prediction = np.expm1(log_prediction)[0]
            
            st.success(f"**${prediction:,.2f}**")
            
        except Exception as e:
            st.error(f"Error during prediction. Ensure the model ('final_model.pkl') and preprocessing are aligned with the 12 input columns. Detail: {e}")

# How to run: streamlit run "/Users/klevizane/Documents/DS_Quant/03_Proyectos/DS/car_predict_price/Nuevo/app_final.py"