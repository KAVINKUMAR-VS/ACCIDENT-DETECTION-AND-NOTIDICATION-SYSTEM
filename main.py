import streamlit as st
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from twilio.rest import Client
import geocoder
import requests

class AccidentDetectionModel:
    class_nums = ['Accident', 'No Accident']

    def __init__(self, model_weights_file):
        try:
            input_layer = Input(shape=(250, 250, 3))
            x = Conv2D(32, (3, 3), activation='relu')(input_layer)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Flatten()(x)
            x = Dense(64, activation='relu')(x)
            output_layer = Dense(2, activation='softmax')(x)

            self.loaded_model = Model(input_layer, output_layer)
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
        ip_address = requests.get('https://api.ipify.org').text
        location = geocoder.ip(ip_address)
        return location.latlng[0], location.latlng[1]
    except Exception as e:
        print(f"Error getting location: {e}")
        return None

def get_address(latitude, longitude):
    try:
        location = geocoder.osm([latitude, longitude], method='reverse')
        return location.address
    except Exception as e:
        print(f"Error getting address: {e}")
        return None

def send_sms_twilio():
    try:
        account_sid = 'AC73e32b2265b51c773bd5c1bc945f998b'
        auth_token = 'd44aabf880d474d2082b420dca519ae0'
        from_number = '+12513134656'
        to_number = '+916382150416'

        client = Client(account_sid, auth_token)
        latitude, longitude = get_location()
        address = get_address(latitude, longitude)

        message_body = f"ACCIDENT DETECTED at {address} (Latitude: {latitude}, Longitude: {longitude}) please hurry up to this place"

        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )

        st.write(f"SMS sent: {message.sid}")

    except Exception as e:
        st.write(f"Error sending SMS: {e}")

def main():
    st.title("Accident Detection and Alerting System")
    
    model = AccidentDetectionModel('model_weights.h5')
    font = cv2.FONT_HERSHEY_SIMPLEX

    video_path = st.text_input("Enter video path:", "head_on_collision_101.mp4")
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        st.write(f"Error: Couldn't open the video source. Check the file path: {video_path}")
        return

    while True:
        ret, frame = video.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :, :])
        prob_percentage = round(prob[0][0] * 100, 2)

        if 93 <= prob_percentage <= 100:
            cv2.putText(frame, f"Prediction: {pred} - Probability: {prob_percentage}%", (10, 30), font, 0.7, (255, 0, 0), 2)
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob_percentage}%", (20, 30), font, 1, (255, 0, 0), 2)  

            # Send SMS
            send_sms_twilio()  

            st.image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), channels="BGR")

        if st.button("Stop"):
            break

if __name__ == '__main__':
    main()
