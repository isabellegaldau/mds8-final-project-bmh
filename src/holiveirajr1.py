import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle
from joblib import load
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib


model = joblib.load("xgb_model_default_42.sav")
model_columns = joblib.load("/workspaces/mds8-final-project-bmh/EDA/model_columns.pkl")
county_to_cities = joblib.load("county_to_cities.pkl")
city_encoder = joblib.load("city_encoder.pkl")
county_encoder = joblib.load("county_encoder.pkl")
property_encoder = joblib.load("propertyType_encoder.pkl")


# Title of the app
st.title("Your Florida Top 25 Homes' Price Prediction")


city_to_zipcodes = {
    "Fort Myers": [33916, 33905, 33907, 33901],
    "Jacksonville": [32277, 32208, 32256, 32210],
    "Homestead": [33032, 33033, 33034, 33035],
    "Miami": [33145, 33125, 33129, 33132],
    "Bonita Springs": [34135, 34134],
    "Merritt Island": [32952],
    "Palm Harbor": [34683],
    "Lake Worth": [33460, 33461],
    "Temple Terrace": [33617],
    "Royal Palm Beach": [33411],
    "Estero": [33967],
    "Edgewater": [32141],
    "Sunrise": [33323],
    "Lake Mary": [32746],
    "Tampa": [33602, 33603, 33604, 33605],
    "Port Charlotte": [33948],
    "Boynton Beach": [33435, 33436, 33437, 33472],
    "Ponte Vedra": [32081],
    "Oviedo": [32765],
    "Deltona": [32725],
    "Palm Bay": [32905, 32907],
    "West Palm Beach": [33401, 33405, 33407, 33409],
    "Boca Raton": [33427, 33428, 33431, 33432],
    "Hollywood": [33019, 33020],
    "Palm Coast": [32164],
    "Leesburg": [34748],
    "Davenport": [33896],
    "Naples": [34108, 34114],
    "Orange Park": [32073],
    "Spring Hill": [34609],
    "New Smyrna Beach": [32168],
    "Clermont": [34711],
    "Clearwater": [33755],
    "Lakeland": [33801, 33803],
    "Delray Beach": [33444, 33445, 33446, 33448],
    "Englewood": [34224],
    "Davie": [33314],
    "Hialeah": [33010, 33012],
    "Land O Lakes": [34638],
    "Wesley Chapel": [33543],
    "Saint Petersburg": [33701, 33711],
    "Kissimmee": [34741],
    "New Port Richey": [34652],
    "Sarasota": [34231],
    "Apollo Beach": [33572],
    "Pembroke Pines": [33028, 33024],
    "Saint Augustine": [32084],
    "Lake Wales": [33853],
    "Orlando": [32801, 32803, 32804, 32805],
    "Ocala": [34474],
    "Riverview": [33578],
    "Valrico": [33594],
    "Auburndale": [33823],
    "Winter Springs": [32708],
    "Palmetto": [34221],
    "Fleming Island": [32003],
    "Cutler Bay": [33157],
    "Port Richey": [34668],
    "Largo": [33770],
    "Saint Cloud": [34769],
    "Green Cove Springs": [32043],
    "Winter Haven": [33880],
    "Tarpon Springs": [34689],
    "Bradenton": [34205],
    "Melbourne": [32901, 32935],
    "Eustis": [32726],
    "Wimauma": [33598],
    "Seffner": [33584],
    "Cape Coral": [33904, 33909, 33914, 33990],
    "Fort Pierce": [34947],
    "Port Saint Lucie": [34952],
    "Venice": [34285],
    "Punta Gorda": [33950],
    "North Port": [34286],
    "Jupiter": [33458],
    "Deland": [32720],
    "Casselberry": [32707],
    "Ocoee": [34761],
    "Cocoa": [32922],
    "Lakewood Ranch": [34202],
    "Lady Lake": [32159],
    "Doral": [33172],
    "Tamarac": [33321],
    "Zephyrhills": [33542],
    "Ruskin": [33570],
    "Altamonte Springs": [32701],
    "Coconut Creek": [33063],
    "Rotonda West": [33947],
    "Fort Lauderdale": [33301, 33304, 33305, 33306],
    "Wellington": [33414],
    "Deerfield Beach": [33441],
    "Lutz": [33558],
    "Lehigh Acres": [33936, 33971],
    "Sanford": [32771],
    "Plantation": [33322],
    "Palm Beach Gardens": [33410],
    "Coral Springs": [33065, 33071],
    "Nokomis": [34275],
    "Hudson": [34667],
    "Pompano Beach": [33060],
    "Dunedin": [34698],
    "Wilton Manors": [33311],
    "Lauderdale Lakes": [33319],
    "Loxahatchee": [33470],
    "Seminole": [33772],
    "Winter Garden": [34787],
    "The Villages": [32162],
    "Titusville": [32780],
    "Ormond Beach": [32174],
    "Rockledge": [32955],
    "Plant City": [33563],
    "Summerfield": [34491],
    "Longwood": [32779],
    "Parrish": [34219],
    "Pinellas Park": [33781],
    "Riviera Beach": [33404],
    "Odessa": [33556],
    "Miami Beach": [33139, 33140],
    "Jacksonville Beach": [32250],
    "Windermere": [34786],
    "Greenacres": [33413],
    "Port Orange": [32129],
    "Lauderhill": [33313],
    "Miramar": [33023],
    "Middleburg": [32068],
    "Tavares": [32778],
    "Weston": [33326],
    "Sun City Center": [33573],
    "Mount Dora": [32757],
    "North Fort Myers": [33903],
    "Groveland": [34736],
    "Dunnellon": [34432],
    "Haines City": [33844],
    "North Lauderdale": [33068],
    "Brandon": [33510],
    "Apopka": [32703],
    "Daytona Beach": [32114],
    "South Daytona": [32119],
    "Margate": [33063],
    "Miami Gardens": [33056],
    "Debary": [32713],
    "Saint Johns": [32259],
    "Oakland Park": [33334],
    "Brooksville": [34601],
    "Hallandale Beach": [33009],
    "Orange City": [32763],
    "Ponte Vedra Beach": [32082],
    "Gulfport": [33707],
    "Oldsmar": [34677],
    "Satellite Beach": [32937],
    "Opa Locka": [33054],
    "Winter Park": [32789],
    "Holiday": [34690],
    "Lithia": [33547],
    "Sebastian": [32958],
    "Cocoa Beach": [32931],
    "Lake Worth Beach": [33460],
    "Maitland": [32751],
    "North Miami Beach": [33160],
    "Mulberry": [33860],
    "North Palm Beach": [33408],
    "Jensen Beach": [34957],
    "North Miami": [33161]
}


# Input fields for the user
county_label = st.selectbox("County", list(county_to_cities.keys()))
city_label = st.selectbox("City", county_to_cities[county_label])
zip_code = st.selectbox("Zip Code", city_to_zipcodes[city_label])
#latitude = st.number_input("Latitude", format="%.6f")
#longitude = st.number_input("Longitude", format="%.6f")
property_type_label = st.selectbox("Property Type", list(property_encoder.classes_))
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=19, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, step=1)
square_footage = st.number_input("Square Footage", min_value=200, max_value=11615, step=10)
lot_size = st.number_input("Lot Size (in sqft)", min_value=100, step=10)
year_built = st.number_input("Year Built", min_value=1980, max_value=2025, step=1)
cooling_present = st.selectbox("Cooling Present", options=[0, 1])
fireplace_present = st.selectbox("Fireplace Present", options=[0, 1])
garage_present = st.selectbox("Garage Present", options=[0, 1])
heating_present = st.selectbox("Heating Present", options=[0, 1])
pool_present = st.selectbox("Pool Present", options=[0, 1])
floor_count = st.number_input("Floor Count", min_value=1, step=1)
garage_spaces = st.number_input("Garage Spaces", min_value=0, step=1)
#city_label = st.selectbox("City", list(city_encoder.classes_))
#county_label = st.selectbox("County", list(county_encoder.classes_))
#property_type_label = st.selectbox("Property Type", list(property_encoder.classes_))
#zip_code = st.selectbox("Zip Code", city_to_zipcodes[city_label])

city_encoded = city_encoder.transform([city_label])[0]
county_encoded = county_encoder.transform([county_label])[0]
property_type_encoded = property_encoder.transform([property_type_label])[0]

# Create a DataFrame with the input values
# input_data = pd.DataFrame({
#     'zipCode': [zip_code],
#     'latitude': [latitude],
#     'longitude': [longitude],
#     'bedrooms': [bedrooms],
#     'bathrooms': [bathrooms],
#     'squareFootage': [square_footage],
#     'lotSize': [lot_size],
#     'yearBuilt': [year_built],
#     #'lastSalePrice': [last_sale_price],
#     #'m_rate': [m_rate],
#     'cooling_present': [cooling_present],
#     'fireplace_present': [fireplace_present],
#     'garage_present': [garage_present],
#     'heating_present': [heating_present],
#     'pool_present': [pool_present],
#     'floorCount': [floor_count],
#     'garageSpaces': [garage_spaces],
#     'city': [city],
#     'county': [county_encoded],
#     'propertyType': [property_type_encoded]

# })

input_data = pd.DataFrame([{
    'zipCode': zip_code,
    'latitude': 0,
    'longitude': 0,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'squareFootage': square_footage,
    'lotSize': lot_size,
    'yearBuilt': year_built,
    'lastSalePrice': 0,       # placeholder
    'm_rate': 0.0,
    'cooling_present': cooling_present,
    'fireplace_present': fireplace_present,
    'garage_present': garage_present,
    'heating_present': heating_present,
    'pool_present': pool_present,
    'floorCount': floor_count,
    'garageSpaces': garage_spaces,
    'city_encoded': city_encoded,
    'county_encoded': county_encoded,
    'propertyType_encoded': property_type_encoded
}])
# Match model columns (no get_dummies since this is all numeric)
input_data = input_data[model_columns]

#['zipCode', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'squareFootage', 'lotSize', 
#'yearBuilt', 'lastSalePrice', 'm_rate', 'cooling_present', 'fireplace_present', 'garage_present', 
#'heating_present', 'pool_present', 
#'floorCount', 'garageSpaces', 'city_encoded', 'county_encoded', 'propertyType_encoded']

user_encoded = pd.get_dummies(input_data)
user_encoded = user_encoded.reindex(columns=model_columns, fill_value=0)

st.write(user_encoded.dtypes)

# Predict the price using the model
prediction = model.predict(user_encoded)

# Display the prediction
st.subheader(f"Predicted Price: ${prediction[0]:,.2f}")