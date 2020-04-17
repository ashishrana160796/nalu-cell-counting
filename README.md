## Exploring Cell counting with Neural Arithmetic Logic Units

The big problem for neural network models which are trained to count instances is that whenever test range goes higher than training range generalization error increases i.e. they are not good generalizers outside training range. So, we trained regular CNN architectures with small addition of numerically biased layers which in turn resulted in increased accuracy while predicting the cell counts of data. For validation we used a custom dataset with higher counts of test images even then our model's performance sustained. Here, in this repository we'll provide the implementation of the models stated in the research paper.

__Note:__ Within 24 hours the final commit containing all the necessary codes will be added to this repository.

## Python module versions used for implementation: pip list

* jupyter 1.0.0
* Keras 2.3.1
* numpy 1.18.2
* imageio 2.4.1
* opencv-python 4.1.2.30
* Pillow 7.0.0
* scikit-image 0.16.2         
* scikit-learn 0.22.2.post1   
* scipy 1.4.1
* tensorflow 2.2.0rc2
* torch 1.4.0

April, 2020 environment of Google Colab with GPU training is used for implementation of this code. And the important libraries are
stated above for error-free replication and avoidance of any dependency errors.

## Directory structure and instructions

* `exploring-cell-counting`: Contains jupyter notebooks having implementation of the model used for carrying out experiments in the
paper. Also, corresponding python scripts are prepared that can be used directly as module.
  * Extract the `cell.zip` file and run the `jupyter-notebook` command on your machine. Or you can import this file to your google-colab repository and directly run the code cells in that environment.
  * Also, models standalone script `models.py` is made available for directly importing in your project.

* `dataset-preparation`: Data preparation scripts and custom created compressed data with random rotation replication of sub-images is present this directory. Also, python scripts containing these manipulation functions are provided that can be directly used as sub-module in your program.

* `nalu-experiments`: NALU/NAC based experiments that are specified in the research paper is presented in this section.

* `research-paper-tex`: Research paper's `.tex` file along with assets for reutilization is provided in this directory.

## Original design and code contributions

* New architectures of CNN models containing NALU/NAC based concatenated layers proposed, experimented and implemented to achieve
improved results.
  
* Python scripts used for extraction and creating the custom validation dataset. Also, an extra model trained specifically on this dataset is also provided and the compressed form of that dataset.

## Citing the research paper and experiments

If layer concatenation methodology demonstrated in this paper did helped you to improve your results. Please, cite: 

```
bibtex
@article{rana2020exploring,
Author = {Ashish Rana and Taranveer Singh and Harpreet Singh and Neeraj Kumar and Prashant Singh Rana},
Title = {Exploring Cell counting with Neural Arithmetic Logic Units},
    year={2020},
    eprint={2004.06674},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Repositories referred

Code from below mentioned repositories is utilized while designing experiments for this paper.

* FCRN and U-net cell counting base implementation: [cell_counting_v2](https://github.com/WeidiXie/cell_counting_v2) by  WeidiXie
* NALU and NAC base implementation: [NALU](https://github.com/kgrm/NALU) by kgrm
