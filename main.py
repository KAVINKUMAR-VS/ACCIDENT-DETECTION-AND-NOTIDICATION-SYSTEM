import streamlit as st
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense
from twilio.rest import Client
import geocoder
import os
import base64
import tempfile

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

def annotate_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    highest_probability = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        prob_percentage = round(prob[0][0] * 100, 2)

        if pred == "Accident" and prob_percentage > highest_probability:
            highest_probability = prob_percentage
            text = f"Accident: {highest_probability}%"
        else:
            text = f"No Accident: {prob_percentage}%"

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()

def get_location():
    try:
        location = geocoder.ip('me')
        return location.latlng[0], location.latlng[1]
    except Exception as e:
        st.error(f"Error getting location: {e}")
        return None

def get_address(latitude, longitude):
    try:
        location = geocoder.osm([latitude, longitude], method='reverse')
        return location.address
    except Exception as e:
        st.error(f"Error getting address: {e}")
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

        st.success("SMS sent successfully!")
    except Exception as e:
        st.error(f"Error sending SMS: {e}")

def main():
    st.title("Accident Detection System")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the uploaded file as a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Annotate the video with the highest probability
        output_path = os.path.join(tempfile.gettempdir(), "annotated_video.mp4")
        model = AccidentDetectionModel("model.json", "model_weights.h5")
        annotate_video(temp_file.name, output_path, model)

        # Display the annotated video in a new tab
        st.markdown(f'<a href="{output_path}" target="_blank">Download Annotated Video</a>', unsafe_allow_html=True)

        highest_probability = model.preds[:, 0].max() * 100
        if highest_probability > 50:  # Set your desired threshold
            send_sms_twilio()

        # Close the temporary file
        temp_file.close()

if __name__ == "__main__":
    main()
