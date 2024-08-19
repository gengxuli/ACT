# Official implementation for Asymmetric Co-Training for Source-Free Few-Shot Domain Adaptation
The official GitHub page for paper "Asymmetric Co-Training for Source-Free Few-Shot Domain Adaptation". This paper explores the utilization of **Asymmetric Co-Training** in source free domain adaptation(SFDA), and designs special loss functions. The result demonstrate the effectiveness of these methods compared to tranditional source-free unsupervised domain adaptation(SFUDA) approaches.

## Ⅰ.Prerequisites:
- PyTorch >= 1.13.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.14.0
- python3
- Numpy, argparse, scipy, PIL
- sklearn, opencv-python, tqdm

## Ⅱ.Dataset:

We use four datasets:[Office], [Office-Home], [VisDA-C], [terra_incognita]; they can be downloaded from the official websites, and modify the path of images in each '.txt' under the folder './data/'. And how to generate such txt files could be found in https://github.com/tim-learn/Generate_list

### 1.Office-31

Original Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/), and it also can be downloaded [here](https://github.com/jindongwang/transferlearning/tree/master/data#office-31).

### 2.Office-Home

Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### 3.VisDA-C

Office-Home dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification).

### 4.terra_incognita

terra_incognita dataset can be found [here](https://beerys.github.io/CaltechCameraTraps/). 

The dataset we used can be downloaded by running download.py.

```python
python download.py --data_dir=./
```
If some URLs do not work due to various factors, you can copy the URLs and download them manually.

## Ⅲ.Training:

1. #### few-shot source-free Domain Adaptation (few-shot SFDA) on the Office dataset
- Train model on the source domain **A** (**s = 0**)
```python
python image_source.py --trte val --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --seed 2019
```

- Adaptation to other target domains **D and W**, respectively
```python
python image_target_three.py --da uda --gpu_id 0 --dset office --s 0 --few_shot 3 --seed 0 --SAM --lr 0.00003 --src_seed 2019
```

2. #### few-shot source-free Domain Adaptation (few-shot SFDA) on the Office-Home dataset
- Train model on the source domain **A** (**s = 0**)
```python
python image_source.py --trte val --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --seed 2021
```

- Adaptation to other target domains **C and P and R**, respectively
```python
python image_target_three.py --da uda --gpu_id 0 --dset office-home --s 0 --few_shot 1 --seed 0 --SAM --lr 0.00001 --src_seed 2019
```

3. #### few-shot source-free Domain Adaptation (few-shot SFDA) on the VisDA-C dataset
-  Train model on the source domain **Synthetic** (**s = 0**) 
```python
python image_source.py --trte val --da uda --gpu_id 0,1,2,3 --dset office-home --max_epoch 50 --s 0 --seed 2019
```

- Adaptation to other target domains **real**
```python
python image_target_three.py --da uda --gpu_id 0,1,2,3 --dset VISDA-C --s 0 --few_shot 10 --seed 0 --SAM --lr 0.00003 --src_seed 2019 --net resnet101
```

4. #### few-shot source-free Domain Adaptation (few-shot SFDA) on the terra_incognita dataset
- Train model on the source domain **L38** (**s = 0**)
```python
python image_source.py --trte val --da uda --gpu_id 0,1,2,3 --dset terra_incognita --max_epoch 50 --s 0 --seed 2019
```

- Adaptation to other target domains **L43 and L46 and L100**, respectively
```python
python image_target_three.py --da uda --gpu_id 0,1,2,3 --dset terra_incognita --s 0 --few_shot 1 --seed 0 --SAM --lr 0.00003 --src_seed 2019
```

5. #### few-shot source-free Partial-set Domain Adaptation (few-shot SFPDA) on the Office-Home dataset
- Train model on the source domain **A** (**s = 0**)
```python
python image_source.py --trte val --da pda --gpu_id 0,1,2,3 --dset office-home --max_epoch 50 --s 0 --seed 2019
```

- Adaptation to other target domains **C and P and R**, respectively
```python
python image_target_pda.py --da pda --gpu_id 0,1,2,3 --dset office-home --s 0 --few_shot 3 --seed 0 --SAM --lr 0.00003 --src_seed 2019
```

6. #### few-shot source-free Open-set Domain Adaptation (few-shot SFODA) on the Office-Home dataset
- Train model on the source domain **A** (**s = 0**)
```python
python image_source.py --trte val --da oda --gpu_id 0,1,2,3 --dset office-home --max_epoch 50 --s 0 --seed 2019
```
	
- Adaptation to other target domains **C and P and R**, respectively
```python
python image_target_oda.py --da oda --gpu_id 0,1,2,3 --dset office-home --s 0 --few_shot 3 --seed 0 --SAM --lr 0.00003 --src_seed 2019
```

## Ⅳ.Citation
If you find this code useful for your research, please cite our papers
```

```

## Ⅴ.Contact:

- [gengxuli123@gmail.com](mailto:gengxuli123@gmail.com)
