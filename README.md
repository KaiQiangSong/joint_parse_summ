# Joint Parsing and Generation for Abstractive Summarization

We provide the source code for the paper **"[Joint Parsing and Generation for Abstractive Summarization](https://arxiv.org/pdf/1911.10389.pdf)"**, accepted at AAAI'20. If you find the code useful, please cite the following paper. 

    @inproceedings{joint-parsing-summarization:2020,
     Author = {Kaiqiang Song and Logan Lebanoff and Qipeng Guo and Xipeng Qiu and Xiangyang Xue and Chen Li and Dong Yu and Fei Liu},
     Title = {Joint Parsing and Generation for Abstractive Summarization},
     Booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
     Year = {2020}}

## Goal

* Our system seeks to re-write a lengthy sentence, often the 1st sentence of a news article, to a concise, title-like summary. The average input and output lengths are 31 words and 8 words, respectively. 

* The code takes as input a text file with one sentence per line. It generates 2 text files ("summary.txt" and "parse.txt") in the working folder as the outputs, where each source sentence is replaced by a title-like summary and a corresponding dependency parsing tree.

* Example input and output are shown below. 
  > Belgian authorities are investigating the killing of two policewomen and a passerby in the eastern city of Liege on Tuesday as a terror attack, the country's prosecutor said.

  > belgian prosecutor confirms killing of two policewomen and passerby .

  > belgian prosecutor <-- confirms killing of two policewomen <-- <-- and --> passerby --> --> --> . --> <-- 


## Dependencies

The code is written in Python (v3.7) and Pytorch (v1.3). We suggest the following environment:

* A Linux machine (Ubuntu) with GPU
* [Python (v3.7)](https://www.anaconda.com/download/)
* [Pytorch (v1.3)](https://pytorch.org/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP)
* [Pyrouge](https://pypi.org/project/pyrouge/)

To install [Python (v3.7)](https://www.anaconda.com/download/), run the command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
$ bash Anaconda3-2019.10-Linux-x86_64.sh
$ source ~/.bashrc
```

To install [PyTorch (v1.3)](https://pytorch.org/) and its dependencies, run the below command.
```
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

To download the [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP) and use it as a server, run the command below. The CoreNLP toolkit helps tokenization (for both train and test) and collect dependency parse trees from target sentences (for train only).
```
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
$ unzip stanford-corenlp-full-2018-10-05.zip
$ cd stanford-corenlp-full-2018-10-05
$ nohup java -mx16g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 60000 &
$ cd -
```
To install [Pyrouge](https://pypi.org/project/pyrouge/), run the command below. Pyrouge is a Python wrapper for the ROUGE toolkit, an automatic metric used for summary evaluation.  
```
$ pip install pyrouge
```

## I Want to Generate Summaries..

1. Clone this repo. Download this ZIP  file ([`others.zip`](http://i2u.world/kqsong/model/aaai2020_kaiqiang_1/others.zip)) containing vocabulary files and trained models. Move the ZIP file to the working folder and uncompress.
    ```
    $ git clone git@github.com:KaiQiangSong/joint_parse_summ.git
    $ mv others.zip joint_parse_summ
    $ cd joint_parse_summ
    $ unzip others.zip
    $ rm others.zip
    $ mkdir log
    ```

2. Generating Summaries with our joint parsing and generating summarization model trained on selected dataset including: gigaword (default), newsroom, cnndm, websplit.
    ```
    $ python run.py --do_test --inputFile data/test.txt
    ```
    Or if you want runing models other than that trained on gigaword:
    ```
    $ python run.py --do_test --data newsroom --inputFile data/test.txt
    ```
   
## I Want to Train the Model..
1. Training the Model with train files and validation files.
    ```
    $ python run.py --do_train --train_prefix data/train --valid_prefix data/valid
    ```
    Or if you want to train other models (flatParse, flatToken)
    ```
    $ python run.py --do_train --model flatParse --train_prefix data/train --valid_prefix data/valid
    ```

2. (Optional) Modify the training options.
    
    You might want to change the parameters used for training. These are specified in `./setttings/training/gigaword.json` and explained blow.
    
```
{
	"reload":false, # If you want to reload from previous training model, in case of Issues like Power Off
	"reload_path":"./model/checkpoint_Epoch8.pth.tar", # Which file you want to reload
	"optimizer": # Using Adam in our optimizer
	{
		"type":"Adam",
		"params":
		{
			"lr":0.001,
			"betas":[0.9, 0.999],
			"eps":1e-08,
			"weight_decay":1e-06
		}
	},
	"grad_clip": # Gradient Clipping
	{
		"min":-5,
		"max":5
	},
	"stopConditions":
	{
		"max_epoch":30, # Maximum Running Epochs
		"earlyStopping":true, # Using Early Stopping
		"earlyStopping_metric":"valid_err", # Using Validation Loss as metric 
		"earlyStopping_bound":60000, #Stop the training when the validation loss didn't update for 60k batches
		"rateReduce_bound":24000 # Reduce the Learning Rate by half if the validation loss didn't update for 24k batches 
	},
	"checkingPoints":
	{
		"checkMin":10000, # First Checking Point after 10k batches
		"checkFreq":2000, # Check points after each 2k batches
		"everyEpoch":true # Save a checkpoint after each epoch
	}
}
```

HINT*: 60K batches (used for `earlyStopping_bound`) correspond to about 1 epoch for our dataset. 24K batches (used for `rateReduce_bound`) is slightly less than half of an epoch.
