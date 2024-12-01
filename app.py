import glob  
import streamlit as st  
import wget  
from PIL import Image  
import torch  
import cv2  
import os  
import time  


st.set_page_config(layout="centered")  

cfg_model_path = 'models/yolov5s.pt'  
model = None  
confidence = 0.25  

# For Image
def image_input(data_src):  
    img_file = None  
    img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])  
    if img_bytes:  
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]  
        Image.open(img_bytes).save(img_file)  

    if img_file:  
        col1, col2 = st.columns(2)  
        with col1:  
            st.image(img_file, caption="Selected Image")  
        with col2:  
            img = infer_image(img_file)  
            st.image(img, caption="Model Prediction")  

# For video
def video_input(data_src):  
    vid_file = None  
    vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])  
    if vid_bytes:  
    # Ensure the directory exists  
        upload_dir = "data/uploaded_data" 
         # Create the directory if it doesn't exist 
        os.makedirs(upload_dir, exist_ok=True)   
        vid_file = os.path.join(upload_dir, "upload." + vid_bytes.name.split('.')[-1])  
        with open(vid_file, 'wb') as out:  
            out.write(vid_bytes.read())  

    if vid_file:  
        cap = cv2.VideoCapture(vid_file)  
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
      
        output = st.empty()  
        prev_time = 0  
        curr_time = 0  
        while True:  
            ret, frame = cap.read()  
            if not ret:  
                st.write("Can't read frame, stream ended? Exiting ....")  
                break  
            frame = cv2.resize(frame, (width, height))  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            output_img = infer_image(frame)  
            output.image(output_img)  
            curr_time = time.time()  
            fps = 1 / (curr_time - prev_time) if prev_time else 0  
            prev_time = curr_time  
            

        cap.release()


def infer_image(img, size=None):  
    model.conf = confidence  
    result = model(img, size=size) if size else model(img)  
    result.render()  
    image = Image.fromarray(result.ims[0])  
    return image  


@st.cache_resource  
def load_model(path, device):  
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)  
    model_.to(device)  
    print("Model loaded to ", device)  
    return model_  


@st.cache_resource  
def download_model(url):  
    model_file = wget.download(url, out="models")  
    return model_file  

# Backgroung Color 
def set_background_color():  
    st.markdown(  
        """  
        <style>  
        .stApp {  
            background-color: #c4c1e0;   
            color: black;               
            
        }  
        </style>  
        """,  
        unsafe_allow_html=True  
    )






def main():  
    global model, confidence, cfg_model_path  
    set_background_color()
    
    st.title("Object Detection Model") 
    st.write("This is a simple object detection model using YOLOv5")
   
    st.sidebar.title("Customise the Settings")  

    # Upload model  
    model_src = st.sidebar.header("Model YOLOv5s")  
    
    # Check if model file is available  
    if not os.path.isfile(cfg_model_path):  
        st.warning("Model file not available! Please add it to the model folder.", icon="⚠️")  
    else:  
        # Device options  
        device_option ='cpu'  
        st.sidebar.text(f"Using device: {device_option}")  

        # Load model  
        model = load_model(cfg_model_path, device_option)  

        # Confidence slider  
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)  

        # Custom classes  
        if st.sidebar.checkbox("Custom Classes"):  
            model_names = list(model.names.values())  
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])  
            classes = [model_names.index(name) for name in assigned_class]  
            model.classes = classes  
        else:  
            model.classes = list(model.names.keys())  

        # Input options  
        input_option = st.sidebar.radio("Select your input type: ", ['image', 'video'])  

        # Input source option  
        data_src = st.subheader("Upload Your file")  

        if input_option == 'image':  
            image_input(data_src)  
        else:  
            video_input(data_src)  




if __name__ == "__main__":  
    try:  
        main()  
    except SystemExit:  
        pass