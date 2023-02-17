# MURF

## **Task #1: Shared information extraction**
<div align=center><img src="https://github.com/hanna-xu/others/blob/master/images/MURF_task1_show.png" width="950" height="330"/></div>

### To train:
* Download the training data [RGB-IR](https://pan.baidu.com/s/1MPSmWuOhKr2KQxD8aj5gHA?pwd=e9gf) (created with [*RoadScene*](https://github.com/hanna-xu/RoadScene) dataset), [RGB-NIR](https://pan.baidu.com/s/1oakDnUKCtT0MaxjP-6Q0jA?pwd=epov) (created with [*VIS-NIR Scene*](http://matthewalunbrown.com/nirscene/nirscene.html) dataset), [PET-MRI](), [CT-MRI]() (created with [*Harvard*](http://www.med.harvard.edu/AANLIB/home.html) dataset) or create your training dataset according to [it](https://github.com/hanna-xu/utils):<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python main.py```
### To test:
* Put the test data in `./test_imgs/`<br>
* Run ```CUDA_VISIBLE_DEVICES=0 python test.py```<br>

## Task #2: Multi-scale coarse registration
### To train:
* Download the training data [RGB-IR], [RGB-NIR], [PET-MRI], [CT-MRI] or create your training dataset.
* Adjust `task1_model_path` in `main.py` to the path where you store the model in task #1.
* Run ```CUDA_VISIBLE_DEVICES=0,1 python main.py```
* Put some training images of large spatial resolution in `./large_images_for_training/`
* Finetune the trained model with images of original large spatial resolution by running ```CUDA_VISIBLE_DEVICES=0,1 python finetuning.py```
### To test:
* Prepare test data (one of the two ways):
    * Put the test images in `./test_data/images/` ***or*** 
    * Put the test data (including images and **landmark**) in `./test_data/LM/` in `.mat` format <br> 
* Run ```CUDA_VISIBLE_DEVICES=0 python test.py``` ***or*** run ```CUDA_VISIBLE_DEVICES=0,1 python test_w_finetuning.py``` 

## Task #3: Fine registration and fusion
### To train:
### To test:


## Recommended Environment
* tensorflow 1.14.0<br>
* numpy 1.19.2
