import streamlit as st
from streamlit_webrtc import webrtc_streamer
from deepface import DeepFace
import cv2
import numpy as np
from openai import OpenAI
import av

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="StressEmoAge Lite", layout="wide")
st.title("🧠 Real-time Age, Gender & Emotion Detector (No TensorFlow)")

def contextual_explanation(attrs):
    prompt = f"Explain briefly what this suggests: {attrs}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return resp.choices[0].message.content

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        results = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        attrs = results[0]
        age = attrs['age']
        gender = attrs['dominant_gender']
        emotion = attrs['dominant_emotion']
        conf = attrs['emotion'][emotion]
        text = f"{gender}, Age: {age}, Emotion: {emotion} ({conf:.2f})"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        explanation = contextual_explanation({"age": age, "gender": gender, "emotion": emotion})
        cv2.putText(img, explanation, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    except Exception as e:
        cv2.putText(img, f"Error: {str(e)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="cam", video_frame_callback=video_frame_callback)
