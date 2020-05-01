## Custom BBBC005 high count dataset.

The dataset is present in [tar.xz](https://drive.google.com/file/d/1C9HKTmrubgCIISb6v5xJfmwhe_p7I7HA/view?usp=sharing) and [.zip](https://drive.google.com/open?id=1rr6h6ucbbH8yOoac9SnxCjTWNmHC84Wy) format for replication of experiments conducted in the stated research paper.
A sample view of the dataset is present in the research paper if you are interested in looking for preview of this dataset.
Also, this dataset is designed from [BBBC005 synthetic dataset](https://data.broadinstitute.org/bbbc/BBBC005/) with having naming convention portability of [VGG synthetic cell dataset](http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html) for making the code migration process easier for this dataset.
 

## Python module versions used for implementation: pip list

* jupyter 1.0.0
* numpy 1.18.2
* imageio 2.4.1
* opencv-python 4.1.2.30
* Pillow 7.0.0

## Directory structure and instructions

* `dataset-preparation`: Data preparation jupyter notebook and [link to](https://drive.google.com/file/d/1C9HKTmrubgCIISb6v5xJfmwhe_p7I7HA/view?usp=sharing) custom created compressed data with random rotation replication of sub-images is present this directory. For [.zip](https://drive.google.com/open?id=1rr6h6ucbbH8yOoac9SnxCjTWNmHC84Wy) you can click on this [link](https://drive.google.com/open?id=1rr6h6ucbbH8yOoac9SnxCjTWNmHC84Wy).
  * You can execute `jupyter-notebook` command and create your own version of dataset with the provided script if required with your customization as per your probelm use-case. Also, the script assumes the BBBC005 dataset to extracted in the same directory of the script. Please, change the `PATH` as per your requirement.

## Original design and code contributions

* A highcount dataset created from the already existing BBBC005 synthetic dataset. Use 70/30 split with test images being present with w2 width only for creating more robust image segementation and object counting models which can differentiate with overlapping cell images.


## Citing the research paper and experiments

If this custom high count dataset helped you in your research. Please, cite: 

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
