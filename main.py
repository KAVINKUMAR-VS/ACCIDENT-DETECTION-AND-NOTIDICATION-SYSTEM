import streamlit as st
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense
from twilio.rest import Client
import geopy
import geopy.geocoders
import tempfile

# Add custom CSS styles
st.markdown("""
    <style>
        body {
            color: #333;
            background-color: #f4f4f4;
        }
        .st-bq {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .st-cb {
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

class AccidentDetectionModel:
    class_nums = ['Accident', 'No Accident']

    def __init__(self, model_json_file, model_weights_file):
        try:
            # Define the model architecture using the functional API
            input_layer = Input(shape=(250, 250, 3), name='input_layer')
            
            x = BatchNormalization(name='batch_normalization')(input_layer)
            
            x = Conv2D(32, (3, 3), activation='relu', name='conv2d')(x)
            x = MaxPooling2D((2, 2), name='max_pooling2d')(x)
            
            x = Conv2D(64, (3, 3), activation='relu', name='conv2d_1')(x)
            x = MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
            
            x = Conv2D(128, (3, 3), activation='relu', name='conv2d_2')(x)
            x = MaxPooling2D((2, 2), name='max_pooling2d_2')(x)
            
            x = Conv2D(256, (3, 3), activation='relu', name='conv2d_3')(x)
            x = MaxPooling2D((2, 2), name='max_pooling2d_3')(x)
            
            x = Flatten(name='flatten')(x)
            
            x = Dense(512, activation='relu', name='dense')(x)
            
            output_layer = Dense(2, activation='softmax', name='dense_1')(x)
            
            self.loaded_model = Model(inputs=input_layer, outputs=output_layer)
            
            # Load model weights
            self.loaded_model.load_weights(model_weights_file)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def predict_accident(self, img):
        try:
            self.preds = self.loaded_model.predict(img)
            return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds
        except Exception as e:
            raise RuntimeError(f"Error predicting accident: {e}")

def get_location():
    try:
        locator = geopy.geocoders.Nominatim(user_agent="AccidentDetectionApp")
        location = locator.geocode("Chennai")
        return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Error getting location: {e}")
        return None, None

def get_address(latitude, longitude):
    try:
        locator = geopy.geocoders.Nominatim(user_agent="AccidentDetectionApp")
        location = locator.reverse((latitude, longitude))
        return location.address
    except Exception as e:
        st.error(f"Error getting address: {e}")
        return None

def send_sms_twilio():
    try:
        account_sid = 'ACda3432c307b6f6e241eab51d4338ce0e'
        auth_token = '2862f061b43ae94f12fe27306b2d9c3a'
        from_number = '+12164467689'
        to_number = '+916382150416'

        client = Client(account_sid, auth_token)

        latitude, longitude = get_location()
        
        if latitude is None or longitude is None:
            return False

        address = get_address(latitude, longitude)

        message_body = f"ACCIDENT DETECTED at {address} (Latitude: {latitude}, Longitude: {longitude}) please hurry up to this place"

        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )

        st.success("SMS sent successfully!")
        return True
    except Exception as e:
        st.error(f"Error sending SMS: {e}")
        return False

def main():
    st.title("Accident Detection System")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the uploaded file as a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Read the temporary file using OpenCV
        video = cv2.VideoCapture(temp_file.name)

        model = AccidentDetectionModel("model.json", "model_weights.h5")

        # Initialize SMS sent flag
        sms_sent = False

        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(gray_frame, (250, 250))

            pred, prob = model.predict_accident(roi[np.newaxis, :, :, :])
            prob_percentage = round(prob[0][0] * 100, 2)

            if 93 <= prob_percentage <= 100:
                cv2.putText(frame, f"Prediction: {pred} - Probability: {prob_percentage}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"{pred} {prob_percentage}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                st.image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), channels="BGR")

                if not sms_sent:
                    send_sms_twilio()
                    sms_sent = True

        # Release video capture
        video.release()

        # Close the temporary file
        temp_file.close()

if __name__ == "__main__":
    main()
