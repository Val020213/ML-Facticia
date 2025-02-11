import json
import torch


class DataFormat:

    def __init__(self, filename, xyxyxyxy, type, text=""):

        self.filename = filename
        self.xyxyxyxy = xyxyxyxy
        self.type = type
        self.text = text

    def to_dict(self):

        return {
            "filename": self.filename,
            "xyxyxyxy": self.xyxyxyxy.tolist(),
            "type": int(self.type.item()),
            "text": self.text,
        }

    def __str__(self):
        return "Filename: {}, Type: {}, Text: {}".format(
            self.filename, self.type, self.text
        )


def from_json(json_file: str) -> list[DataFormat]:

    data = {}
    crops = []

    with open(json_file, "r") as f:
        data = json.load(f)

    for value in data.values():
        crop_name = value["filename"]
        crop_type = value["type"]
        crop_text = value["text"]
        crop_xyxyxyxy = torch.tensor(value["xyxyxyxy"])
        crops.append(DataFormat(crop_name, crop_xyxyxyxy, crop_type, crop_text))

    return crops
