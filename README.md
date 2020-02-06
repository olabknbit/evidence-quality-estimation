
# Evidence quality estimation using selected machine learning techniques
Project created as part of research for Master's thesis at Warsaw University of Technology, graduate degree.

## Abstract
Evidence Based Medicine is a practice in which a medical action is required to be made using the best available evidence-based recommendations. A doctor has to gather the evidence and assess it before making a decision. Due to the lack of time, combined with an ever-growing selection of medical publications, medical practitioners often fail to follow the best recommendations of EBM. This thesis proposes a system for automatic grading of evidence, so that practitioners can focus their limited attention on the most promising publications, assessed as containing strong evidence and omit the low-quality ones.

Relevant features are extracted from the publication's abstract and metadata and experiments with different preprocessing steps and classifying techniques are performed. 
Evidence grading is approached as a multi-label classification task. The classes represent grades in a widely used Strength of Recommendation Taxonomy (SORT). A dataset, designed specifically for evidence quality estimation, is used to test the accuracy of the predictions. Classifiers are trained with combinations of high-level features, and their predictions are used for ensemble techniques. Numerous methods are experimented with, and the most successful ones are stacking classifiers and consensus method.

The ensemble technique which achieves the best results, uses a Support Vector Classifiers trained on multiple high level features, which predictions are then used to train a Random Forest Classifier. The accuracy score yielded by this pipeline is 75.41%, which is a significant improvement over the baseline - 48% achieved by classifying all instances as the majority class. 

Results reported by this work are very promising. The most important predictor is definitely the publication type of articles comprising the body of evidence. Designed system is tuned for SORT, however, due to it's generality, it can easily be used with other evidence grading systems.


## Getting the dataset

These code is written with an ALTA 2011 Shared Task dataset in mind. It can be obtained by contacting one of the authors of

Diego Molla, Abeed Sarker. _"Automatic Grading of Evidence: the 2011 ALTA Shared Task (2011)"_. 
Proceedings of the 2011 Australasian Language Technology Workshop, Canberra.

Or alternatively, by converting and using `ebmsumcorpus` dataset, 
which is available online at:
https://sourceforge.net/projects/ebmsumcorpus/ (last checked 4 Feb 2020).

The files should be placed in `data/alta_2011_shared_task` dir. This dir should have two subdirs: `dataset`, `testset` and one file `testset.txt`.

`dataset` dir should contain:  `devtestset` and `trainset` dirs and `trainset.txt` and `devtestset.txt` files.

`testset`, `devtestset` and `trainset` dirs should contain XML files that are PubMed downloaded abstracts.

`testset.txt`, `trainset.txt` and `devtestset.txt` files should be in a following format,
```
82691 A 17237298 16080084 12514443 15716561 16531939 
82692 B 15716561 
20952 C 11417373 9099150 
```
where first column is evidence id, second column is SORT grade, and remaining columns are PubMed references. 
For each reference, there should be a respective `txt` file in a respective dir.

## Requirements

2. Download MetaMap https://metamap.nlm.nih.gov/MetaMap.shtml and put it in `metamap` dir, so that `public_mm` is in `metamap` dir.
3. Download `python3.6` or higher
4. Install requirements with `pip install -r requirements.txt`
5. Run `installer.py` to download `NLTK`'s dependencies
1. Create file `filename.py` and put there variable `path` with absolute path to the Project. 
E.g. contents of this file could look like: 
```
path = "/Users/aleksandra/PycharmProjects/mgr/"
```

## Generate Meta Mapped content

Meta Mapped content needed to run existing scripts have been saved in the `data\metamapped` dir for user's convenience. 
Below are instructions for downloading and running MetaMap, if user prefers to generate their own Meta Mappings.

### How to use MetaMap 

```bash
cd metamap/public_mm
```
Start the server
```bash
./bin/skrmedpostctl start
./bin/wsdserverctl start
```
In another session run `generate_metamapped_content.py` with
```bash
python generate_metamapped_content.py
```

#### Trouble shooting
if you see an exeption
```
common::run_command() : [ERROR]: output = b'/Users/aleksandra/PycharmProjects/mgr/metamap/public_mm/bin/SKRrun.18 
/Users/aleksandra/PycharmProjects/mgr/metamap/public_mm/bin/metamap18.BINARY.Darwin --lexicon db -Z 2018AB -I
Failed to set locale to "default". Exiting',
error code = 1
```
run from command line (and not e.g. Python console)

## Running scripts

Please note that you cannot run the scripts before securing a dataset.
All of the below mentioned scripts are in the `examples` dir, and they represent example experiment setups.

`run_example.ipynb` provides the most straigh-forward examples of running various scripts.

In the sections described below, there are more sophisticated experimental setups - running for all features and all classifiers, 
experimenting with various parameters in an automated way. These require understanding basic python.

### Train base classifiers (First-level classifiers)

In order to train first-level classifiers, use one of the scripts:

`calclulate_confidence_intervals.py` - to calculate confidence intervals.

`run_grid_search.py` - to run grid search on self specified parameters, classifiers and features.

`generate_intermediate_features.py` - to generate intermediate features that will later be used as input in ensemble methods.
Please specify the dirname you want your intermediate features to be stored in.

These scripts were used for the experiments in first part of the thesis.
Examples of running are specified in these files.
### Train ensemble methods

Use scripts in 
`run_ensemble_algos.py` for consensus, auctioning and game_theory techniques

`run_ensemble_classifiers.py` for ensemble classifiers

and `run_ensemble_neural_networks.py` for neural networks.

These scripts are highly configurable. The user can specify which feature 
sets they want to use, what parameters etc., 
This may require basic understanding existing of the code.
However, simply running these scripts should be straight forward.

Examples of usage are in the mentioned files. (Please read the functions in the files and try running them / changing them.)