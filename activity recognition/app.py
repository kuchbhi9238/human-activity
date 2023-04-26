import streamlit as st
import vid_pred
import cv2
import shutil
import os
import helper
import beepy
import smtplib
from email.mime.text import MIMEText

frame_th = 50
helper.del_dir('temp')

def save_video(video_file):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())
    return "uploaded_video.mp4"

def read_video(path):
    video_file = open(path, 'rb')
    video_bytes = video_file.read()
    return video_bytes


def send_email(subject, body):
    sender_email = 'kuchbhi9238@gmail.com' ## enter your mail id
    sender_password = 'gvcsnjfppqcozcvg'
    receiver_email = 'lcarnage977@gmail.com'
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, sender_password)
    smtp_server.sendmail(sender_email, receiver_email, message.as_string())
    smtp_server.quit()

CLASSES_LIST = ["Punch", "YoYo", "PlayingGuitar", "HorseRace","Lunges"]

st.title('Activity detection app')
choice = st.sidebar.radio(label='Select',options=['Upload'])

if choice == 'Upload':
    video_file = st.file_uploader("Upload video", type=["mp4","avi"])
    if st.button('Predict'):
        if video_file is not None:
            vid_path = save_video(video_file)
            video_bytes = read_video(vid_path)
            st.video(video_bytes)
            pred,prob = vid_pred.vid_class_pred(vid_path,CLASSES_LIST)
            beepy.beep(sound='error')
            st.markdown(f'Prediction class : {pred}')
            st.markdown(f'Prediction prob : {prob}')
            send_email('Activity detected', f'Activity detected in the video. Class: {pred}, Probability: {prob}')
        else:
            st.error('Video file not uploaded')
else:
    if st.button('Start camera'):
        helper.del_dir('temp')
        if os.path.isfile('temp.mp4'):
            os.remove('temp.mp4')
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        cap.set(3, 800)  # width
        cap.set(4, 480)  # height
        frame_counter = 0
        numbers = st.empty()
        while True:
            ret, frame = cap.read()
            if ret:
                frame_counter += 1
                file_name = os.path.join('temp',f"frame_{frame_counter}.png")
                cv2.imwrite(file_name, frame)
                if frame_counter % frame_th == 0:
                    helper.frame_to_vid('temp')
                    pred,prob = vid_pred.vid_class_pred('temp.mp4',CLASSES_LIST)
                    with numbers.container():
                        st.markdown(f'Prediction class : {pred}')
                        beepy.beep(sound='error')
                        st.markdown(f'Prediction prob : {prob}')
                        send_email('Activity detected', f'Activity detected in the video. Class: {pred}, Probability: {prob}')
                    os.remove('temp.mp4')
                    helper.del_dir('temp')
                    frame_counter = 0
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),use_column_width=True)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()