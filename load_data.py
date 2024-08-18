from datasets import load_dataset
import json


"""
d = load_dataset("keremberke/license-plate-object-detection", name="full")

car= '\car'

directory_for_labels = r"c:\Users\hp\Desktop\Image_segmentation_project\data\Labels"

directory_for_images = r'c:\Users\hp\Desktop\Image_segmentation_project\data\Images'


#Training data
for i in range(len(d)):
    
    #preparing label data 
    current_car = d['train']['objects'][i]
    with open(directory_for_labels + car + '_' +str(current_car['id'][0]) + '.txt', 'w') as file:
        box_cood = str(current_car['bbox'][0])
        box_cood = box_cood.replace('[','').replace(']','').replace(","," ")
        line = str(current_car['category'][0]) + ' ' + box_cood
        
        if len(current_car['bbox'])!=1:
            raise Exception('There is more than one box data in the image.')
        
        file.write(line)

    #preparing image data

    current_image = d['train']['image'][i]
    current_image_directory = directory_for_images + car + '_' +str(current_car['id'][0]) + '.jpeg'
    current_image.save(current_image_directory, 'JPEG')


"""

with open(r'c:\Users\hp\Desktop\Image_segmentation_project\data\Train\Data_config_json\_annotations.coco.json') as f:
    meta_data = json.load(f)
