
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import pycococreatortools
import pathlib
from matplotlib import pylab as pl
import cv2


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def filter_for_images(root, files):

    file_types = ['*.jpeg', '*.jpg','*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.jpeg', '*.jpg','*.png']
    #file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():

    total_imagenum_count = 1

    for coco_type in total_coco:

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        ROOT_DIR = os.path.join(maindir, 'data', mother_dataset_name )
        IMAGE_DIR = os.path.join(ROOT_DIR, coco_type)
        ANNOTATION_DIR = os.path.join(ROOT_DIR, coco_type+'_annotations')

        RENAME_IMAGE_DIR = os.path.join(ROOT_DIR, "rename_" + coco_type)
        RENAME_ANNOTATION_DIR = os.path.join(ROOT_DIR,"rename_" + coco_type+"_annotations")

        os.makedirs(RENAME_IMAGE_DIR,exist_ok=True)
        os.makedirs(RENAME_ANNOTATION_DIR,exist_ok=True)

        img_files = sorted(os.listdir(IMAGE_DIR))
        anno_files = sorted(os.listdir(ANNOTATION_DIR))

        for i in range(len(anno_types)):
            blank = [name for name in anno_files if anno_types[i] in name]
            if annotation_dict[i]['annotation_types'] == anno_types[i] :
                annotation_dict[i]['filenames'] = blank

        count = total_imagenum_count
        for num_ in range(len(img_files)):

            img_name = img_files[num_]
            img_ext = pathlib.PurePosixPath(img_name).suffix
            img_new_name = img_name.replace(img_name, DATASET_NAME_PREFIX + "_" + TRAIN_DATASET + "_{0:012d}".format(count)) + img_ext
            origin_img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
            resize_img = resize_image(origin_img, width=780)
            cv2.imwrite(os.path.join(RENAME_IMAGE_DIR, img_new_name), resize_img)

            for anno_num in range(len(anno_types)) :

                anno_type = anno_types[anno_num] #dog, cat
                anno_name = annotation_dict[anno_num]['filenames'][num_]

                anno_ext = pathlib.PurePosixPath(anno_name).suffix
                anno_new_name = anno_name.replace(anno_name, DATASET_NAME_PREFIX + "_" + TRAIN_DATASET + "_{0:012d}".format(count))+ "_"+anno_types[anno_num]+anno_ext # 분류될 이름을 반드시 넣을 것!!! cat 이부분에
                origin_anno = cv2.imread(os.path.join(ANNOTATION_DIR, anno_name))

                resize_anno = resize_image(origin_anno, width=780)

                cv2.imwrite(os.path.join(RENAME_ANNOTATION_DIR, anno_new_name), resize_anno)

            count += 1

        print("Renaming image file name is done!")
        print("# of Dataset : {}".format(count - 1))
        total_imagenum_count += count

        image_id = 1
        segmentation_id = 1

        view_count = 1
        for root, _, files in os.walk(RENAME_IMAGE_DIR):

            image_files = filter_for_images(root, files) # image파일 골라내기

            for image_filename in image_files: # rename 이미지 파일들 하나씩 불러오기

                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
                # coco데이터 셋에 정보를 넣기위해 이미지 번호,이름,이미지 크기등을 입력하여

                coco_output["images"].append(image_info)
                # 위 coco_output의 images 칸에 image_info를 넣기

                for root, _, files in os.walk(RENAME_ANNOTATION_DIR):
                    annotation_files = filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                        # CATEGORIES에서 목록을 뽑아내고 annotation_filename에
                        # x의 'name'부분과 일치하는 부분이 있는지 확인 후, 일치하는 id number를 가져오는 부분.

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}  # is_crowd가 뭔지 아직 모름

                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8) #0과 1로 이루어진 마스크를 생성.

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)
                        # annotation의 정보를 만들어주는 듯. segmentation number를 저장하고 (위 image_id와 같은 세트이므로 같아야됨)
                        # 그리고 해당 annotation의 category 정보를 넣어주고, binary 마스크를 입력, 이미지 크기 또한 입력한다.
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1
                print("--------------------------")
                print("{}/{} is completed ! ".format(view_count,len(image_files)))
                view_count +=1
                image_id = image_id + 1

        with open(os.path.join(RENAME_ANNOTATION_DIR,coco_type+'2020.json').format(ROOT_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

if __name__ == "__main__":

    DATASET_NAME_PREFIX = 'pet'
    TRAIN_DATASET = 'train2020'
    VAL_DATASET = 'val2020'

    maindir = os.path.abspath("../")

    mother_dataset_name = "pet"

    coco_train = "coco_train"
    coco_valid = "coco_valid"
    coco_test = "coco_test"
    anno_types = ['dog', 'cat']

    total_coco = [coco_train,coco_valid,coco_test]

    INFO = {
        "description": "tak's dog&cat Dataset",
        "url": "https://github.com/asyncbridge/pycococreator",
        "version": "0.0.1",
        "year": 2020,
        "contributor": "asyncbridge",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    annotation_dict = [
        {
            "annotation_types":"dog",
            "filenames":[]
        },

        {
            "annotation_types": "cat",
            "filenames": []
        }
    ]
    CATEGORIES = [
        {
            'id': 1,
            'name': 'dog',
            'supercategory': 'pet',
        },
        {
            'id': 2,
            'name': 'cat',
            'supercategory': 'pet',
        },
    ]

    main()
