import json
import os
import pycocotools.mask as maskUtils
from PIL import Image
import numpy as np
from collections import defaultdict

''' json file structured as follows
  {
      "info": {
          "year": 2024,
          "version": "1.0",
          "description": "Manga109 Segmentation",
          "url": "https://huggingface.co/datasets/MS92/MangaSegmentation",
      },
      "licenses": [
          {
              "id": 1,
              "name": "Attribution-NonCommercial-ShareAlike License",
              "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/"
          }
      ],
      "images": [
          {
            "license": 0,
            "id": 0,
            "width": 1654,
            "height": 1170,
            "file_name": "ARMS/000.jpg"
        },
          // More image entries here...
      ],
         
      "categories": [
        {"id": 1, "name": "frame", "supercategory": "frame"},
        {"id": 2, "name": "text", "supercategory": "text"},
        {"id": 3, "name": "face", "supercategory": "character"},
        {"id": 4, "name": "body", "supercategory": "character"},
        {"id": 5, "name": "balloon", "supercategory": "balloon"},
        {"id": 6, "name": "onomatopoeia", "supercategory": "text"}
      ],
      "annotations": [
          {
              "id": 0,
              "image_id": 2,
              "category_id": 1,
              "segmentation": [RLE format],
              "area": 670660,
              "bbox": [x, y, width, height],
              "iscrowd": 0
          },
          // More annotation entries here...
      ]
  }
'''

rle2mask = maskUtils.decode

def json2dict(json_path: str):
    with open(json_path, 'r', encoding='utf8') as f:
        metadata = json.loads(f.read())
    return metadata

json_dict = json2dict(r'D:\MangaSegmentation\jsons\ARMS.json')
annotations_point = json_dict['annotations']

img_annotation_dict = defaultdict(list)
for ann in annotations_point:
    img_annotation_dict[ann['image_id']].append(ann)

image_file = json_dict['images'] # [{img_name: ... , width, id:... } , {}, ... ]
img_dict = {img_file['id']: img_file for img_file in image_file}


img_id = 4
cat_id = 1


img_info = img_dict[img_id] #Thoong tin anh mang image = 4
print(img_info)
visualize = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)
annotations = img_annotation_dict[img_id] #Cac anno cua image = 4
print(annotations)

for ann in annotations:
    if ann['category_id'] == cat_id:
        mask = rle2mask(ann['segmentation'])
        visualize[mask > 0] = np.random.randint(0, 255, (1, 3))

if visualize is not None:
    visualize = Image.fromarray(visualize)
# visualize
os.makedirs(os.path.dirname(img_info['file_name']),exist_ok=True)
visualize.save(img_info['file_name'][:-4]+'.png')
