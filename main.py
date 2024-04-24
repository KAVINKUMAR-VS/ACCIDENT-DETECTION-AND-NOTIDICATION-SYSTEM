import streamlit as st
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense
from twilio.rest import Client
import geocoder
import os
import base64

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

        # Read the uploaded file using OpenCV
        video_array = np.frombuffer(uploaded_file.read(), np.uint8)
        video = cv2.VideoCapture(video_array)

        model = AccidentDetectionModel("model.json", "model_weights.h5")

        # Initialize SMS sent flag
        sms_sent = False

        # Define the directory to save annotated images
        os.makedirs("annotated_frames", exist_ok=True)
        
        frame_count = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        annotated_video_path = "annotated_video.mp4"
        annotated_video = cv2.VideoWriter(annotated_video_path, fourcc, 25.0, (640, 480))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(gray_frame, (250, 250))

            pred, prob = model.predict_accident(roi[np.newaxis, :, :])
            prob_percentage = round(prob[0][0] * 100, 2)

            # Annotate probability on the video frame
            cv2.putText(frame, f"Prediction: {pred} - Probability: {prob_percentage}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Save annotated frame as an image
            annotated_frame_path = f"annotated_frames/frame_{frame_count}.jpg"
            cv2.imwrite(annotated_frame_path, frame)

            # Write frame to annotated video
            annotated_frame = cv2.imread(annotated_frame_path)
            annotated_video.write(annotated_frame)

            if pred == "Accident" and not sms_sent:
                send_sms_twilio()
                sms_sent = True
                st.warning(f"Accident detected with {prob_percentage}% probability!")

            frame_count += 1

        # Release everything if job is finished
        video.release()
        annotated_video.release()
        cv2.destroyAllWindows()

        # Generate download link for annotated video
        with open(annotated_video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.markdown(f"Download annotated video: [annotated_video.mp4](data:video/mp4;base64,{base64.b64encode(video_bytes).decode()})", unsafe_allow_html=True)

        # Remove the temporary directory
        for i in range(frame_count):
            os.remove(f"annotated_frames/frame_{i}.jpg")
        os.rmdir("annotated_frames")
        os.remove(annotated_video_path)

if __name__ == "__main__":
    main()
