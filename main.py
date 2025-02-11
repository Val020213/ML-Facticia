from src.full_model import FullModel

export_path = "./dataset/output"

fullModel = FullModel(model="src/detection/best.pt")

fullModel.run(export_path, "./dataset/target", load_mode=False)

bbox = fullModel.associate_bounding_boxes()

for key, value in bbox.items():
    print("key: ", key, " value: ", value)
    
proximity, _ = fullModel.get_proximity(export_path)

for key, value in proximity.items():
    print("key: ", key, " value: ", value)

