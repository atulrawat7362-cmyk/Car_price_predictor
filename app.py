import pickle
import pandas as pd

from flask import Flask, request, render_template


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD MODEL
# ==========================================

with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# LOAD PREPROCESSOR
# ==========================================

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("cardekho_dataset.csv")


# ==========================================
# GET DROPDOWN VALUES
# ==========================================

brands = sorted(
    df["brand"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

models = sorted(
    df["model"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

seller_types = sorted(
    df["seller_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

fuel_types = sorted(
    df["fuel_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

transmission_types = sorted(
    df["transmission_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ==========================================
# BRAND -> MODEL MAPPING
# ==========================================

brand_models = {}

for brand in brands:

    brand_models[brand] = sorted(
        df.loc[
            df["brand"].astype(str) == brand,
            "model"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# PREDICTION PAGE
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # ======================================
    # GET REQUEST
    # ======================================

    if request.method == "GET":

        return render_template(
            "predict.html",

            prediction=None,

            brands=brands,

            models=models,

            brand_models=brand_models,

            seller_types=seller_types,

            fuel_types=fuel_types,

            transmission_types=transmission_types
        )


    # ======================================
    # POST REQUEST
    # ======================================

    if request.method == "POST":

        # ----------------------------------
        # GET FORM VALUES
        # ----------------------------------

        brand = request.form["brand"]

        model_name = request.form["model"]

        vehicle_age = float(
            request.form["vehicle_age"]
        )

        km_driven = float(
            request.form["km_driven"]
        )

        seller_type = request.form["seller_type"]

        fuel_type = request.form["fuel_type"]

        transmission_type = request.form["transmission_type"]

        mileage = float(
            request.form["mileage"]
        )

        engine = float(
            request.form["engine"]
        )

        max_power = float(
            request.form["max_power"]
        )

        seats = float(
            request.form["seats"]
        )


        # ----------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------

        input_data = pd.DataFrame({

            "brand": [brand],

            "model": [model_name],

            "vehicle_age": [vehicle_age],

            "km_driven": [km_driven],

            "seller_type": [seller_type],

            "fuel_type": [fuel_type],

            "transmission_type": [transmission_type],

            "mileage": [mileage],

            "engine": [engine],

            "max_power": [max_power],

            "seats": [seats]

        })


        # ----------------------------------
        # PREPROCESS
        # ----------------------------------

        input_processed = preprocessor.transform(
            input_data
        )


        # ----------------------------------
        # PREDICTION
        # ----------------------------------

        prediction = model.predict(
            input_processed
        )


        predicted_price = round(
            float(prediction[0])
        )


        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return render_template(

            "predict.html",

            prediction=predicted_price,

            brands=brands,

            models=models,

            brand_models=brand_models,

            seller_types=seller_types,

            fuel_types=fuel_types,

            transmission_types=transmission_types

        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)