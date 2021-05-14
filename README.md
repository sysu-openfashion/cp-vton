# Toward Characteristic-Preserving Image-based Virtual Try-On Network

Reimplemented code for eccv2018 paper 'Toward Characteristic-Preserving Image-based Virtual Try-On Network'. 

The results may have some differences with those of the original code.

This code is tested with pytorch>=1.4.0.

## Dataset
You can get the processed data at [GoogleDrive](https://drive.google.com/open?id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo) or by running:

```
python data/data_download.py
```
## Pretrained Models
We (SYSU-OpenFashion) provide pretrained models for CP-VTON. You can download them [here](https://drive.google.com/file/d/1EhF8SXNkX0jk34WVPvvxWiEfnNTVMrZI/view?usp=sharing) and upzip to the root directory of this repo, and then you should be able to directly run the test process.
## Geometric Matching Module

### training
We just use L1 loss for criterion in this code. 

TV norm constraints for the offsets will make GMM more robust.

An example training command is
```
python train.py --name GMM --stage GMM --data_list train_pairs.txt --shuffle
```
You can see the results in tensorboard, as show below.
<div align="center">
  <img src="result/gmm_train_example.png" width="576px" />
    <p>Example of GMM train. The center image is the warped cloth.</p>
</div>

### testing

An example test command is
```
python test.py --name GMM --stage GMM --data_list test_pairs.txt --checkpoint checkpoints/GMM/gmm_final.pth
```

You can see the results in tensorboard, as show below.

<div align="center">
  <img src="result/gmm_test_example.png" width="576px" />
    <p>Example of GMM test. The center image is the warped cloth.</p>
</div>

## Try-On Module
### training
Before the trainning, you should generate warp-mask & warp-cloth, using the test process of GMM. Then specify `--warproot` with the directory containing the warping result of GMM. An example training command is

```
python train.py --name TOM --stage TOM --warproot result/GMM/gmm_final.pth --data_list train_pairs.txt --shuffle 
```
You can see the results in tensorboard, as show below.

<div align="center">
  <img src="result/tom_train_example.png" width="576px" />
    <p>Example of TOM train. The center image in the last row is the synthesized image.</p>
</div>


### testing
An example test command is

```
python test.py --name TOM --stage TOM --warproot result/GMM/gmm_final.pth --data_list test_pairs.txt --checkpoint checkpoints/TOM/tom_final.pth
```

You can see the results in tensorboard, as show below.

<div align="center">
  <img src="result/tom_test_example.png" width="576px" />
    <p>Example of TOM test. The center image in the last row is the synthesized image.</p>
</div>


## Citation
If this code helps your research, please cite our paper:

	@inproceedings{wang2018toward,
		title={Toward Characteristic-Preserving Image-based Virtual Try-On Network},
		author={Wang, Bochao and Zheng, Huabin and Liang, Xiaodan and Chen, Yimin and Lin, Liang},
		booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
		pages={589--604},
		year={2018}
	}


