import streamlit as st
import os

from src.full_model import FullModel

st.set_page_config(
    page_title="Streamlit App",
    page_icon="",
)

st.write("# Welcome to Streamlit! 👋")

export_path = "./dataset/output"
target_path = "./dataset/target"

fullModel = FullModel(model="src/detection/best.pt")

# upload image to predict
image = st.file_uploader("Upload Image", type=["jpg"])

if image:
    
    # Add image to target folder
    with open(target_path + "/image.jpg", "wb") as f:
        f.write(image.read())
        
    print("Image uploaded successfully!")
    
    st.image(image, caption="Uploaded Image", use_column_width=False)
    
    st.write("Running model...")
    
    fullModel.run(export_path, target_path, load_mode=False)
    
    bbox = fullModel.associate_bounding_boxes()
    proximity, texts = fullModel.get_proximity(export_path)

    for file_name in os.listdir(export_path):
        image_path = os.path.join(export_path, file_name)
        
        st.image(image_path, caption=f"Image: {file_name}", use_column_width=True)
        
        
        if file_name in bbox:
            for i in range(len(bbox[file_name])):
                st.write(f"Proximity {i}:", bbox[file_name][i]["distance"])
                st.write("Caption:\n", bbox[file_name][i]["text_caption"])
        else:
            st.write("No bounding boxes for this image.")
        
        
        if file_name in proximity:
            for i in range(len(proximity[file_name])):
                st.write(f"Proximity {i}:", proximity[file_name][i])
                st.write("Text:\n", texts[i])
        else:
            st.write("No proximity data for this image.")
    