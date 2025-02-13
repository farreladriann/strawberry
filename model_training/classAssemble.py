import os
from typing import Dict

def change_label_class(base_path: str, mappingNumber: Dict[int, int]) -> None:
    for folder in ["train", "valid", "test"]:
        folder_path = os.path.join(base_path, folder)
        labels_folder_path = os.path.join(folder_path, "labels")
        for file in os.listdir(labels_folder_path):
            file_path = os.path.join(labels_folder_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            with open(file_path, "w") as f:
                for line in lines:
                    class_number = int(line.split()[0])
                    if class_number in mappingNumber:
                        class_number = mappingNumber[class_number]
                    f.write(f"{class_number} {' '.join(line.split()[1:])}\n")


