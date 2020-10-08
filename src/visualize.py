
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import os
# added - draw bounding box and label

CATEGORIES = [
    {
        'id': 1,
        'name': 'cat',
        'supercategory': 'pet',
    },

]

maindir = os.path.abspath("../")
setname = "pet"
undername = "cat"

ROOT_DIR = os.path.join(maindir, 'examples', setname, 'train')
RENAME_IMAGE_DIR = os.path.join(ROOT_DIR, "rename_" + undername)

annotation_file = ROOT_DIR + '/train2020_cat.json'
example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['patternless'])
image_ids = example_coco.getImgIds(catIds=category_ids)

print(image_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]


# added - draw bounding box and label
def showAnnsBBox(image, annos):
    color = (0, 255, 0)

    for ann in annotations:
        box = ann['bbox']
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), color,
                      3)
        class_name = [x['name'] for x in CATEGORIES if x['id'] == ann['category_id']][0]
        cv2.putText(image, class_name, (int(box[0]), int(box[1] - 20)), 0, 1, color, 2)
        plt.imshow(image);


# load and display instance annotations

image = io.imread(os.path.join(RENAME_IMAGE_DIR , image_data['file_name']))

plt.figure(figsize=(7, 7))

plt.imshow(image);
plt.axis('off')
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# draw segmentation
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)

example_coco.showAnns(annotations)

# added - draw bounding box and label
showAnnsBBox(image, annotations)

plt.show()