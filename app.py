import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AI Perception Lite", layout="wide")

st.title("👁️ AI Perception Lite – Human Attribute Recognition")

# Load TFLite models
interpreter_face = tf.lite.Interpreter(model_path="blazeface.tflite")
interpreter_age_gender = tf.lite.Interpreter(model_path="age_gender.tflite")
interpreter_emotion = tf.lite.Interpreter(model_path="emotion.tflite")

def analyze_frame(frame):
    # Face detection, crop & inference (pseudo)
    # ...
    return {"age": 36, "gender": "male", "emotion": "neutral", "confidence": 0.88}

def contextual_explanation(attrs):
    prompt = f"Explain briefly what the visual context suggests: {attrs}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=60
    )
    return resp.choices[0].message.content

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    attrs = analyze_frame(img)
    explanation = contextual_explanation(attrs)
    # Draw overlay (age, gender, emotion)
    cv2.putText(img, f"{attrs['gender']} {attrs['age']} Emotion:{attrs['emotion']}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img, explanation, (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="realtime", video_frame_callback=video_frame_callback)
