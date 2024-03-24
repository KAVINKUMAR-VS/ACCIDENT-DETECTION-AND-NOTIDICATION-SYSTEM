import cv2
from keras.models import model_from_json
import numpy as np
from twilio.rest import Client
import geocoder
import requests

class AccidentDetectionModel:
    class_nums = ['Accident', 'No Accident']

    def __init__(self, model_json_file, model_weights_file):
        try:
            with open(model_json_file, 'r') as json_file:
                loaded_model_json = json_file.read()
                self.loaded_model = model_from_json(loaded_model_json)

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
        # Get your IP address
        ip_address = requests.get('https://api.ipify.org').text
        
        # Geocode the IP address
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
        # Twilio credentials
        account_sid = 'your_account_sid'
        auth_token = 'your_auth_token'
        from_number = 'your_twilio_number'
        to_number = 'your_number'

        client = Client(account_sid, auth_token)

        # Get location
        latitude, longitude = get_location()
        address = get_address(latitude, longitude)

        # Message body
        message_body = f"ACCIDENT DETECTED at {address} (Latitude: {latitude}, Longitude: {longitude}) please hurry up to this place"

        # Send SMS
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number
        )

        print(f"SMS sent: {message.sid}")

    except Exception as e:
        print(f"Error sending SMS: {e}")

def start_application():
    sms_sent = False  
    try:
        model = AccidentDetectionModel("model.json", 'model_weights.h5')
        font = cv2.FONT_HERSHEY_SIMPLEX

        video_path = "E:\\Downloads\\Accident-Detection-System-main\\Accident-Detection-System-main\\head_on_collision_101 (1).mp4"
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Couldn't open the video source. Check the file path: {video_path}")
            raise RuntimeError("Couldn't open the video source.")

        while True:
            ret, frame = video.read()

            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(gray_frame, (250, 250))

            pred, prob = model.predict_accident(roi[np.newaxis, :, :])
            prob_percentage = round(prob[0][0] * 100, 2)

            cv2.putText(frame, f"Prediction: {pred} - Probability: {prob_percentage}%", (10, 30), font, 0.7, (255, 0, 0), 2)

            if pred == "Accident" and not sms_sent:
                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"{pred} {prob_percentage}%", (20, 30), font, 1, (255, 0, 0), 2)  

                send_sms_twilio()  
                sms_sent = True  

            cv2.imshow('Video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_application()


