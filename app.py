from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and preprocessing pipeline
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        brand = request.form["brand"]
        model_name = request.form["model"]
        variant = request.form["variant"]
        body_type = request.form["body_type"]

        car_age = int(request.form["car_age"])
        kilometers = int(request.form["kilometers"])

        fuel_type = request.form["fuel_type"]
        transmission = request.form["transmission"]

        owner_type = request.form["owner_type"]

        seats = int(request.form["seats"])
        engine_cc = int(request.form["engine_cc"])
        power_bhp = float(request.form["power_bhp"])
        torque_nm = float(request.form["torque_nm"])
        mileage_kmpl = float(request.form["mileage_kmpl"])

        color = request.form["color"]
        airbags = int(request.form["airbags"])

        insurance_type = request.form["insurance_type"]
        service_history = request.form["service_history"]
        registration_type = request.form["registration_type"]

        previous_accidents = int(request.form["previous_accidents"])

        state = request.form["state"]
        city = request.form["city"]

        # Convert Yes/No from HTML into 1/0
        is_metro_city = 1 if request.form["is_metro_city"] == "Yes" else 0

        # Convert HTML owner names to the values used during training
        owner_mapping = {
            "First Owner": "First",
            "Second Owner": "Second",
            "Third Owner": "Third",
            "Fourth & Above Owner": "Fourth & Above"
        }

        owner_type = owner_mapping.get(owner_type, owner_type)

        # Create DataFrame with exactly the same features
        # used while training the model
        car_data = pd.DataFrame({
            "Brand": [brand],
            "Model": [model_name],
            "Variant": [variant],
            "Body_Type": [body_type],
            "Car_Age": [car_age],
            "Kilometers_Driven": [kilometers],
            "Fuel_Type": [fuel_type],
            "Transmission": [transmission],
            "Owner_Type": [owner_type],
            "Seats": [seats],
            "Engine_CC": [engine_cc],
            "Power_BHP": [power_bhp],
            "Torque_Nm": [torque_nm],
            "Mileage_kmpl": [mileage_kmpl],
            "Color": [color],
            "Airbags": [airbags],
            "Insurance_Type": [insurance_type],
            "Service_History": [service_history],
            "Registration_Type": [registration_type],
            "Previous_Accidents": [previous_accidents],
            "State": [state],
            "City": [city],
            "Is_Metro_City": [is_metro_city]
        })

        # Apply the same preprocessing used during training
        processed_data = preprocessor.transform(car_data)

        # Predict price
        predicted_price = model.predict(processed_data)[0]

        # Send prediction to result page
        return render_template(
            "index.html",
            predicted_price=predicted_price
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
