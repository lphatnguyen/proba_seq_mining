# Efficient Dynamic Texture Classification with Probabilistic Motifs

## Introduction
This repository is for our research contribution dedicated to video classification by means of dynamic texture modeling. In this program, we mainly focused on how to use the *p-sequences* to mine probabilistically frequent motifs and create a new representation for video data of dynamic tetures. In this code, we demonstrates two ways of mining key motifs using both Numpy (for simple CPU computation) and PyTorch (for using GPU computation). 

To make our code work, we test it with a dataset called UCLA Dynamic texture. There are 3 schemes(challenges) in this datasets. In this repository, we use 2 schemes UCLA 9-class and UCLA 8-class. In our research work, we use a 50/50 validiation protocol where 50% of the samples in the scheme is used for training while the rest is for testing. The process is repeat 20 times. Furthermore, since we have a stochastic process in our pipeline, we repeat the testing for 8 times. 

The UCLA dataset is downloaded from [Bernard Ghanem's Homepage](http://www.bernardghanem.com/datasets). The split files can be found in the [split files](#split_files) folder. These files need to be copied to the dataset folder.

## Requirements
- Python >=3.6
- PyTorch >=1.3.0
- Numpy >= 1.20.3
- joblib
- tqdm
- scikit-learn

## Usage
For example, in order to run the whole training and testing process for a set of parameters, the following command are typed in the terminal:
```bash
python main.py --patch_size=12 --num_clusters=20 --coef=1 --gamma=1 --set_idx=1 --gap=5 --loop=1
```

In order to test our method with different set of parameters and subsets, please run the following command in terminal:
```bash
chmod +x main.sh
./main.sh
```

The results will be stored in multiple 'results' folders where each .pt will store a confusion matrix. To view the overall results with each set of parameters, run the following command in terminal:
```bash
python vis_res.py
```

## Citation
If you use this code for publication, please cite:

```bibtex
@misc{lphatnguyen2021,
    title   = {Efficient Dynamic Texture Classification with Probabilistic Motifs},
    author  = {Luong Phat Nguyen and Julien Mille and Dominique Li and Donatello Conte and Nicolas Ragot},
    year    = {2021},
    url     = {https://github.com/lphatnguyen/proba_seq_mining}
}
```