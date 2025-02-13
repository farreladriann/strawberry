import os
import yaml
from typing import Dict

def normalize_name(name: str) -> str:
    return name.lower().replace('-', ' ').replace('_', ' ')

def combined_data_yaml(source_folder: str, target_folder: str) -> None:
    source_yaml = os.path.join(source_folder, "data.yaml")
    target_yaml = os.path.join(target_folder, "data.yaml")

    with open(source_yaml, "r") as file:
        source_data = yaml.load(file, Loader=yaml.FullLoader)

    with open(target_yaml, "r") as file:
        target_data = yaml.load(file, Loader=yaml.FullLoader)

    normalized_combined = [normalize_name(name) for name in target_data["names"]]
    
    for name in source_data["names"]:
        normalized_name = normalize_name(name)
        if normalized_name not in normalized_combined:
            normalized_combined.append(normalized_name)

    combined_data = {
        "train": target_data["train"],
        "val": target_data["val"],
        "test": target_data["test"],
        "names": normalized_combined,
        "nc": len(normalized_combined)
    }

    with open(target_yaml, "w") as file:
        yaml.dump(combined_data, file)

# return dictionary of names
def dictionary_yaml_old_to_new(source_folder: str, target_folder_with_new_data_yaml: str) -> Dict[int, int]:
    source_yaml = os.path.join(source_folder, "data.yaml")
    target_yaml = os.path.join(target_folder_with_new_data_yaml, "data.yaml")

    with open(source_yaml, "r") as file:
        source_data = yaml.load(file, Loader=yaml.FullLoader)

    with open(target_yaml, "r") as file:
        target_data = yaml.load(file, Loader=yaml.FullLoader)

    source_names = source_data["names"]
    target_names = target_data["names"]

    dictionary = {}
    for i, name in enumerate(source_names):
        normalized_name = normalize_name(name)
        if normalized_name in target_names:
            dictionary[i] = target_names.index(normalized_name)
        else:
            # ERROR
            print(f"{name} not found in target data.yaml")
            return {}

    return dictionary

def move_file(source_file: str, target_folder: str) -> None:
    source_file_name = os.path.basename(source_file)
    target_file = os.path.join(target_folder, source_file_name)
    if os.path.exists(target_file):
        # ERROR
        raise FileExistsError("File already exists in target folder")
    else:
        os.rename(source_file, target_file)

def move_label(source_label: str, target_folder: str, mapping: Dict[int, int]) -> None:
    source_label_name = os.path.basename(source_label)
    target_label = os.path.join(target_folder, source_label_name)
    if os.path.exists(target_label):
        # ERROR
        raise FileExistsError("File already exists in target folder")
    else:
        with open(source_label, "r") as file:
            lines = file.readlines()
        with open(target_label, "w") as file:
            for line in lines:
                line = line.split()
                line[0] = str(mapping[int(line[0])])
                line = " ".join(line)
                file.write(line + "\n")

def combine_dataset(source_folder: str, target_folder: str) -> None:
    main_folders = ["train", "valid", "test"]
    for main_folder in main_folders:
        # move images and labels
        source_images_folder = os.path.join(source_folder, main_folder, "images")
        target_images_folder = os.path.join(target_folder, main_folder, "images")

        source_labels_folder = os.path.join(source_folder, main_folder, "labels")
        target_labels_folder = os.path.join(target_folder, main_folder, "labels")

        combined_data_yaml(source_folder, target_folder)
        mapping = dictionary_yaml_old_to_new(source_folder, target_folder)
        if mapping == {}:  # ERROR
            return
        
        for source_image in os.listdir(source_images_folder):
            source_image_path = os.path.join(source_images_folder, source_image)
            move_file(source_image_path, target_images_folder)

            source_label = source_image.replace(".jpg", ".txt")
            source_label_path = os.path.join(source_labels_folder, source_label)
            move_label(source_label_path, target_labels_folder, mapping)