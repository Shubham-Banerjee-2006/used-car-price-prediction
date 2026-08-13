import pandas as pd
from flask import Flask, render_template, request
import joblib
app = Flask(__name__, static_folder="static")
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict_price():

    brand = request.form.get("brand")
    model_name = request.form.get("model")
    variant = request.form.get("variant")
    body_type = request.form.get("body_type")

    car_age = float(request.form.get("car_age"))
    kilometers = float(request.form.get("kilometers"))

    fuel_type = request.form.get("fuel_type")
    transmission = request.form.get("transmission")
    owner_type = request.form.get("owner_type")

    seats = float(request.form.get("seats"))
    engine_cc = float(request.form.get("engine_cc"))
    power_bhp = float(request.form.get("power_bhp"))
    torque_nm = float(request.form.get("torque_nm"))
    mileage_kmpl = float(request.form.get("mileage_kmpl"))

    color = request.form.get("color")
    airbags = float(request.form.get("airbags"))

    insurance_type = request.form.get("insurance_type")
    service_history = request.form.get("service_history")
    registration_type = request.form.get("registration_type")

    previous_accidents = float(request.form.get("previous_accidents"))

    state = request.form.get("state")
    city = request.form.get("city")

    # Convert Yes/No into 1/0
    metro_value = request.form.get("is_metro_city")

    if metro_value == "Yes":
        is_metro_city = 1
    else:
        is_metro_city = 0

    # Create input data in the same order as the model features
    input_data = pd.DataFrame([{
        "Brand": brand,
        "Model": model_name,
        "Variant": variant,
        "Body_Type": body_type,
        "Car_Age": car_age,
        "Kilometers_Driven": kilometers,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Seats": seats,
        "Engine_CC": engine_cc,
        "Power_BHP": power_bhp,
        "Torque_Nm": torque_nm,
        "Mileage_kmpl": mileage_kmpl,
        "Color": color,
        "Airbags": airbags,
        "Insurance_Type": insurance_type,
        "Service_History": service_history,
        "Registration_Type": registration_type,
        "Previous_Accidents": previous_accidents,
        "State": state,
        "City": city,
        "Is_Metro_City": is_metro_city
    }])

    # Preprocess the input
    processed_data = preprocessor.transform(input_data)

    # Predict price
    prediction = model.predict(processed_data)[0]

    # Show result
    return render_template(
        "index.html",
        predicted_price=prediction
    )
print("Model features:")
print(model.n_features_in_)

print("\nPreprocessor features:")
print(preprocessor.feature_names_in_)
if __name__ == "__main__":
    app.run()