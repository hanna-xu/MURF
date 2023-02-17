# MURF

## **Task #1: Shared information extraction**
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/MURF_task1_show.png" width="950" height="350"/></div>

### To train:
* Download the training data [RGB-IR](https://pan.baidu.com/s/1MPSmWuOhKr2KQxD8aj5gHA?pwd=e9gf) (created with [*RoadScene*](https://github.com/hanna-xu/RoadScene) dataset), [RGB-NIR](https://pan.baidu.com/s/1oakDnUKCtT0MaxjP-6Q0jA?pwd=epov) (created with [*VIS-NIR Scene*](http://matthewalunbrown.com/nirscene/nirscene.html) dataset), [PET-MRI](), [CT-MRI]() (created with [*Harvard*](http://www.med.harvard.edu/AANLIB/home.html) dataset) or create your training dataset according to [it](https://github.com/hanna-xu/utils):<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python main.py```
### To test:
* Put the test data in **./test_imgs/**<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python test.py```<br>

## Task #2: Multi-scale coarse registration
### To train:
* Download the training data [RGB-IR], [RGB-NIR], [PET-MRI], [CT-MRI] or create your training dataset.
* Adjust ```task1_model_path``` in ```main.py``` to the path where you store the model in task #1.
* Run ```CUDA_VISIBLE_DEVICES=0,1 python main.py```
### To test:

## Task #3: Fine registration and fusion
### To train:
### To test:


## Recommended Environment
:white_square_button: tensorflow 1.14.0<br>
:white_square_button: numpy 1.19.2
