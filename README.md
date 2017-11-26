# Name Disambiguation using Network Embedding
This repository provides a reference implementation of name disambiguation using network embedding as described in the paper:<br>
> Name Disambiguation in Anonymized Graphs using Network Embedding.<br>
> Baichuan Zhang and Mohammad Al Hasan.<br>
> Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM 2017)<br>
> <https://arxiv.org/pdf/1702.02287.pdf>
## Pre-Requisite

* [Python 2.7](https://www.python.org/) 
* [Numpy](http://www.numpy.org/)
* [Networkx](https://networkx.github.io/)

### Basic Usage

#### Example
To run disambiguation embedding code on name reference "Lei Wang", execute the following command from the project home directory:<br/>
	``python embedding_model/main.py sampled_data/lei_wang.xml 20 0.02 0.005 100 'uniform'``
  
#### Options
You can check out the hyper-parameter options using:<br/>
	``python embedding_model/main.py --help``

#### Output
The output is Macro-F1 result and ranking loss value in each epoch

### Citing
If you find this work useful for your research, please consider citing the following paper:

	@inproceedings{zhang-cikm2017,
	author = {Zhang, Baichuan and Al Hasan, Mohammad},
	 title = {Name Disambiguation in Anonymized Graphs using Network Embedding},
	 booktitle = {Proceedings of the ACM on Conference on Information and Knowledge Management (CIKM)},
	 year = {2017}
	}
