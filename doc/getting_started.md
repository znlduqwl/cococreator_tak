# Getting Started
cocodataset을 만드는 방법을 알려드립니다.

## Prepare datasets

아래와 같은 폴더 배치를 해줘야합니다.

```
pycococreator_tak
├── doc
├── src
├── data
│   ├── {나의 dataset명}
│   │   ├── coco_train
│   │   ├── coco_train_annotations
│   │   ├── coco_valid
│   │   ├── coco_valid_annotations
│   │   ├── coco_test
│   │   ├── coco_test_annotations

```

위와 같이 데이터를 준비한 후, 

src 폴더에 들어가 deploy.py를 실행합니다.

1. cd /{본 폴더가 있는 위치}/pycococreator_tak/src
2. python deploy.py

파일 실행이 끝난 후,

rename_으로 시작하는 폴더 6개가 만들어지게 됩니다.
이때, annotations안에 coco_dataset의 양식에 맞는 json 파일이 생성됩니다.

이 폴더,파일들을 이용하여 coco dataset을 활용하시면 됩니다. 