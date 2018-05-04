Self-regulation: Employing a Generative Adversarial Network to Improve Event Detection
=====

It is slightly simplified implementation of our Self-regulation: Employing a Generative Adversarial Network to Improve Event Detection paper in Tensorflow.

Requirements
-----
	Python 3.6
	Tensorflow 1.2
	Numpy
	sklearn



Corpus:
----
	The corpus of ACE 2005 has been used in our experiment. Though we have no right to share with anyone this corpus.
	One may submit a request to the Linguistic Data Consortium (LDC) for approval to download and use the corpus, 
	or find a way to obtain it in the paper "The Automatic Content Extraction (ACE) Program, Task, Data, and Evaluation".
	
	If one may carry out the adaptation experiments using the TAC-KBP event nugget datation corpus,
	we suggest to access to LDC or the homepage of TAC program.
	

Data sets:
----
	There are 5 data sets need to be used in the experiments for event detection, including lexicon, training and 
	development datasets, as well as the ones respectively contain event types and pretrained word embeddings.
	Listed below is the filenames of the datasets which are necessarily followed without any change.
	One may find the files in the directory of data:
	 
	data/train.txt: training dataset
	data/dev.txt: development dataset
	data/wordlist: lexicon
	data/labellist: event types
	data/wordvec: pretrained word embeddings
	
	In this package, we have provided the files named in that way. There is an example reserved in each of the files data/train, data/dev and data/wordvec.
	By contronst, data/wordlist contains all the tokens occurred in the ACE corpus and data/labellist lists all the concerned event types.
	

Note (about pretrained word and sentence embeddings):
----
	The length of an input sentence is limited to be longtr than 8 but shorter than 80.
	If the real length of a sentence is out of the range, padding or pruning needs to be used for the generation of sentence embedding.
	
	One may provide a file which contains the word embeddings pretrained by her/himself. In such a case, the file name should be the same with that we mentioned above.
	If not, you'd better comment out the 57th line in train.py，and meanwhile modify the 58th line as config['feature_init_weight'] = None.
	We recommend to use the word embeddings which have been previously trained well in Feng et al's work (Feng et al. 2016. A language-independent neural network for event detection. ACL'16).


Preprocess:
----
	python preporcess.py
	Run by excuting the command of preprocess.py, so as to obtain the files of train.txt and dev.txt
	

Train:
----
	**python train.py**
	Run by excuting the command of train.py. The default parameters are used.
	
	python train.py --help
	
	usage: train.py [-h] [--classes CLASSES] [--sequence_length SEQUENCE_LENGTH]
                [--num_layers NUM_LAYERS] [--hidden_size HIDDEN_SIZE]
                [--feature_weight_dropout FEATURE_WEIGHT_DROPOUT]
                [--dropout_rate DROPOUT_RATE] [--rnn_unit RNN_UNIT]
                [--lr_decay LR_DECAY] [--learning_rate LEARNING_RATE]
                [--l2_rate L2_RATE] [--num_epochs NUM_EPOCHS]
                [--train_max_patience TRAIN_MAX_PATIENCE]
                [--batch_size BATCH_SIZE] [--model_path MODEL_PATH]
                [--feature_weight_shape FEATURE_WEIGHT_SHAPE]

	optional arguments:
	  	-h, --help            show this help message and exit
	  	--classes CLASSES     Number of classification (default: 34)
	  	--sequence_length SEQUENCE_LENGTH
	                          Sentence length (default: 80)
	  	--num_layers NUM_LAYERS
	                          Number of hidden layers (default: 1)
	  	--hidden_size HIDDEN_SIZE
	                          Hidden size (default: 150)
	  	--feature_weight_dropout FEATURE_WEIGHT_DROPOUT
	                          Feature weight dropout rate (default: 0.2)
	  	--dropout_rate DROPOUT_RATE
	                          Dropout rate (default: 0)
	  	--rnn_unit RNN_UNIT   RNN unit type (default: lstm)
	  	--lr_decay LR_DECAY   LR decay rate (default: 0.95)
	  	--learning_rate LEARNING_RATE
	                          Learning rate (default: 0.3)
	  	--l2_rate L2_RATE     L2 rate (default: 0)
	  	--num_epochs NUM_EPOCHS
	                          Number of training epochs (default: 200)
	  	--train_max_patience TRAIN_MAX_PATIENCE
	                          default: 100
	  	--batch_size BATCH_SIZE
	                          Batch Size (default: 64)
	  	--model_path MODEL_PATH
	                          Path model to be saved (default: model/best.pkl)
	  	--feature_weight_shape FEATURE_WEIGHT_SHAPE
	                          Path model to be saved (default: [vocab_size, 300])

Eval:
----
	`python eval.py test.txt`

	The model which has been trained well will be preserved under the directory of model/.
	From here on, one may let the model perform on the test data set. Run by excuting the command of eval.py. The output results will be written into the file of test.txt.
	One would like to take into consideration the entity embeddings. If so, the source code needs to be modified a little bit (it is your turn now).



