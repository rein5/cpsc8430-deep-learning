# 8430 - Deep Learning #
## Homework 2 ##

This folder contains a python script to run the seq2seq model on a given test directory. 

Usage: 

	python model_script.py data_folder/ [output_file.txt]

Alternatively, the script seq2seq.sh can be used with identical arguments. 


The output file can be evaluated using the bleu_eval.py script provided for the assignment. 


Note: it's assumed that the specified data_folder/ contains the file: 

	data_folder/testing_data/id.txt

and the folder: 

	data_folder/testing_data/feat/

containing the numpy files for the input feature vectors. 


Giovanni Martino
