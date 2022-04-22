# Vec2Seq



This repo contains codes for the paper [Vector-to-Sequence Models for Sentence Analogies](https://ieeexplore.ieee.org/document/9263191).

Vector-to-sequence model is the decoder trained for reconstructing sentences encoded by vectors in a specific sentence representation space. 
To generate the vector representations of answer sentences, we build a linear regression network which learns the mapping between the distribution of known and expected vectors. 
We subsequently leverage this pre-trained decoder to decode sentences from predicted vectors.

For more, please see the paper.

## Installation
        pip install -r requirements.txt

## Getting started

### Decoder

Please check out the notebook `vector_decoding.ipynb`.



The notebook is a copy of the script `run_decoder.py`.
You can either train a decoder model using:
        
        python3 run_decoder.py
        



<!-- To run the analogy solver, start by training a vector decoder on the text data. 
To customize your decoder, you need to modify the settings in `decoder_config.py`
* To train or test a decoder model, you need to specify `run` (e.g., run = "train")
* Place data files in a specific directory, and pass the path to `data_dir`. Note that one sentence per line in each file. Be sure to specify the `max_original_sent_len` (the maximum length of sentences).
* Customize the vector decoder model. For example, to train a basic RNN with single loss function : `decoder_net = "basicrnn"`, `decoder_loss = "sl"`.
* Set the sentence embedding model
* Create a directory to save output files, set `result_dir` to point to the dir


 -->
<!-- ### Sentence analogy solver

Preprocessing: encode sentences in analogies as vectors, and save vectors and sequences in a `.npz` file:

        python3 preprocess_data.py --text_data_path [PATH_TO_TRAIN/VALIDATE/TEST_SET] --embed_model_name_or_path [ENCODER_NAME_OR_PATH]
        
Note: The text data file contains sentence analogies.
One analogy per line. The format is `'A \t B \t C \t D \n'`.

Edit the setting file `solver_config` to configure the analogy solver.

Train a network with the capability of generating the vector of answer sentence D given three known vectors:

        python3 train_AnalogySolverinSpace.py
           
After you trained a decoder and an analogy solver, you can use them to solve sentence analogy using the script `test_AnalogySolver.py`
        
        python3 test_AnalogySolver.py
        
The results will be saved in the format:
`'A \t B \t C \t D (reference) \t D (prediction) \n'`.
 -->
