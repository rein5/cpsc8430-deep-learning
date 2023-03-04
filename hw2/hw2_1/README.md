# 8430 - Deep Learning #
## Homework 2 ##

This folder contains a python script to run the seq2seq model on a given test directory. 

Usage: 

	python model_script.py data_folder/ [output_file.txt]

The output file can then be evaluated using the bleu_eval.py script provided for the assignment. 


Note: it's assumed that the specified data_folder/ contains the file: 

	testing_data/id.txt

and the subfolder: 

	testing_data/feat/

containing the numpy files for the input feature vectors. 

- Giovanni Martino