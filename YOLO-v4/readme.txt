https://drive.google.com/file/d/1PEMV7wxIcqM3uGw3wbX8FUOQLrIOluxQ/view?usp=sharing

1.위의 링크로 들어가 용량문제로 첨부되지 못한 데이터, pre-train weight, 모델을 다운받아주세요.

2.압축을 풀고 YOLO-v4 폴더에 넣어주세요
  아래와 같이 구성되어야합니다.
YOLO-v4
 └checkpoints
  core
  data
  yolov4-416
  Yolo_v4.ipynb
  
    
[issue]
1. 주피터 노트북으로 main문(object detection부분)을 실행후 또 다시 main문을 호출하게 되면 objectdetection이 실행되지 않습니다.
2. 이때 load model 블럭을 호출하고 다시 main문을 실행하게 되면 정상적으로 작동합니다.
