

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
@inproceedings{fu-etal-2020-drts,
    title = "{DRTS} Parsing with Structure-Aware Encoding and Decoding",
    author = "Fu, Qiankun  and
      Zhang, Yue  and
      Liu, Jiangming  and
      Zhang, Meishan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.609",
    doi = "10.18653/v1/2020.acl-main.609",
    pages = "6818--6828",}
```
