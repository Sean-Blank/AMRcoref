

# End-to-End AMR coreference resolution 

Code for the AMRcoref model
in our ACL 2021 paper "[End-to-End AMR coreference resolution](https://www.aclweb.org/anthology/2020.acl-main.609.pdf)".   


## 1. Environment Setup


The code has been tested on **Python 3.6** and **PyTorch 1.6.0**. 
All other dependencies are listed in [requirements.txt](requirements.txt).

Via conda:
```bash
conda create -n amrcoref python=3.6
source activate amrcoref
pip install -r requirements.txt
```

## 2. Data Preparation

Assuming that you're working on AMR 3.0 ([LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02)),
unzip the corpus to `data/AMR/LDC2020T02`, and make sure it has the following structure:
```bash
(stog)$ tree data/AMR/LDC2020T02 -L 2
data/AMR/LDC2017T10
├── data
│   ├── alignments
│   ├── amrs
│   └── frames
│   └── multisentence
├── docs
│   ├── AMR-alignment-format.txt
│   ├── amr-guidelines-v1.3.pdf
│   ├── file.tbl
│   ├── frameset.dtd
│   ├── PropBank-unification-notes.txt
│   └── README.txt
└── index.html
```

Prepare corpus:
```bash
awk FNR!=1 data/amrs/unsplit/* > amr3.txt
awk FNR!=1 data/alignments/unsplit/* > amr-alignments.txt
python prepare_data.py
```
You can get the data like:
```
├── data
│   ├── align_unsplit
│   ├── amr_unsplit
│   └── xml_unsplit
├── amr3.txt
├── amr3_filtered
└── amr-alignments.txt
```
run the prepocessing codes for train/dev/test data:

```bash
python preprecess.py
```

If you want the preprocessed data directly, please send to [us](fqiankun@gmail.com) and let us know that you are authorized to use LDC2020T02 data.

If there are some bugs, please contact [us](fqiankun@gmail.com). 

## 3. Training


```bash
python train.py
```

## 4. Citation


If you find our code is useful, please cite:
```
@inproceedings{fu-etal-2021-end,
    title = "End-to-End {AMR} Corefencence Resolution",
    author = "Fu, Qiankun  and
      Song, Linfeng  and
      Du, Wenyu  and
      Zhang, Yue",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.324",
    doi = "10.18653/v1/2021.acl-long.324",
    pages = "4204--4214"
}
```
