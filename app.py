
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import streamlit as st
import cv2
#Design the streamlit App

st.title("Object detection and Image Segmentation using YOLOv8 in Python")
modeltype=st.sidebar.radio("Select the task:",['Object Detection','Segmentation'])
confidence=float(st.sidebar.slider("Select the confidence Score:",25,100,40))/100
#Select the Model
if modeltype=='Object Detection':
  modelpath=Path("yolov8n.pt")
#load pretrained model
model=YOLO(modelpath)
# Design the Image Display part
st.sidebar.header("Image to be Detected")
sourceimg=None
sourceimg=st.sidebar.file_uploader("Choose an image...",type=("jpg","jpeg","png","webp","bmp"))
col1,col2=st.columns(2)
with col1:
  if sourceimg is not None:
    uploadedimg=Image.open(sourceimg)
    st.image(sourceimg,caption="Uploaded image",use_column_width=True)
with col2:
  if st.sidebar.button("Detect Object"):
    res=model.predict(uploadedimg,conf=confidence)
    boxes=res[0].boxes
    res_plotted=res[0].plot()[:,:,::-1]
    st.image(res_plotted,caption="Detected Image",use_column_width=True)
