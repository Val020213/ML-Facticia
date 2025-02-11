import streamlit as st
import os
import shutil

from src.full_model import FullModel

st.set_page_config(
    page_title="CF-ERL",
    page_icon="",
    # layout="wide",
)

st.write("# Welcome! 👋")
st.write("###### This is a Streamlit app from the CF-ERL ML Project.")
export_path = "./dataset/output"
target_path = "./dataset/target"
target_export_folder = export_path + "/image"
target_file = target_path + "/image.jpg"
fullModel = FullModel(model="src/detection/best.pt")

# upload image to predict
image = st.file_uploader(
    "Upload Image",
    type=["jpg"],
)

if image:
    # Add image to target folder
    if os.path.exists(target_file):
        os.remove(target_file)

    if os.path.exists(target_export_folder):
        shutil.rmtree(target_export_folder)

    with open(target_file, "wb") as f:
        f.write(image.read())

    print("Image uploaded successfully!")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running model, please wait..."):
        fullModel.run(export_path, target_path, load_mode=False)

    bbox = fullModel.associate_bounding_boxes()

    proximity, texts = fullModel.get_proximity(export_path)

    st.success("Model run successfully!")
    st.write("## Image Details")

    for file_name in os.listdir(target_export_folder):
        if not file_name.lower().endswith(".jpg"):
            continue
        image_path = os.path.join(target_export_folder, file_name)
        st.image(image_path, caption=f"Image: {file_name}", use_column_width=True)
        filename = file_name.split(".")[0]
        ftype = fullModel.get_type(filename)
        if filename in bbox:
            st.success("Caption Bounding Boxes found!")
            for i in range(len(bbox[filename])):
                with st.expander(f"Bounding Box {i} Details", expanded=False):
                    st.write("*Distance from Image:*", bbox[filename][i]["distance"])
                    st.write("*Text:*", bbox[filename][i]["text_caption"])
        elif ftype == 2:
            st.info("No bounding boxes for this image.")

        if filename in proximity:
            st.success("Proximity text found!")
            for i in range(len(proximity[filename])):
                with st.expander(f"Proximity {i} Details", expanded=False):
                    st.write(f"*Proximity Value {i}:*", proximity[filename][i])
                    st.write("*Associated Text:*", texts[i])

        elif ftype == 2:
            st.info("No proximity data for this image.")
