# MURF

## **Task #1: Shared information extraction**
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/MURF_task1_show.png" width="950" height="350"/></div>

### To train:
* Download the training data [RGB-IR](https://pan.baidu.com/s/1MPSmWuOhKr2KQxD8aj5gHA?pwd=e9gf) (created with [*RoadScene* dataset](https://github.com/hanna-xu/RoadScene)), [RGB-NIR](https://drive.google.com/drive/folders/1Zp5IsyrexcnJQA_ogjEyFIQl2kVEHRHd?usp=sharing) (created with [*VIS-NIR Scene* dataset](http://matthewalunbrown.com/nirscene/nirscene.html)), [PET-MRI](), [CT-MRI]() (created with [*Harvard* dataset](http://www.med.harvard.edu/AANLIB/home.html)) or create your training dataset according to [it](https://github.com/hanna-xu/utils):<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python main.py```
### To test:
* Put the test data in **./test_imgs/**<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python test.py```<br>

## Task #2: Multi-scale coarse registration
### To train:
### To test:

## Task #3: Fine registration and fusion
### To train:
### To test:


## Recommended Environment
:white_square_button: tensorflow 1.14.0<br>
:white_square_button: numpy 1.19.2
