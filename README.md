# MusicRNN

Note: The final project document (Pred_Chord.pdf) contains an overview of the findings.

## Instructions to Run the Programs ON TEST DATA 

Note: Install requirements.txt before proceeding with the following. We require python version 3.9.18, so specify that when creating the python environment. 

First cd into the final_proj/ folder. This contains all the code that you require for the execution for the various models.

### To Run Naive Bayes
Run the following command:
```
python3 test.py
```
This should load the JSON file containing already parsed data for "training" Naive Bayes and then also load the pickle file containing the Music21 parsed version of the test data. It should then run Naive Bayes on each measure of the test data. THIS WILL TAKE A WHILE TO CREATE PICKLE FILES AND THEN LOAD ALL THE JSON DATA.


### To Run RNN
Run the following command:
```
python test_rnn -m [mode= 'pre', 'create', 'train', 'eval'] -E [epochs] -B [batch size] -L [l.r.]
```

Note: If you are working with the most recent version of the code and want to train and test the model, you only need to run train and eval. The preprocessing and tensor creation is already done.

### To Run LSTM
Run the following command:
```
python test_rnn -m [mode= 'pre', 'create', 'train', 'eval'] -E [epochs] -B [batch size] -L [l.r.]
```
Only include the flags if you want to use the modified datasets for the following:
```
--clean : [for the dataset with the simplified notes]
--downbeat : [for the dataset wth only downbeats]
```
Note: If you are working with the most recent version of the code and want to train and test the model, you only need to run train and eval. The preprocessing and tensor creation is already done.

### To Run Transformer
Run the following command:
```
python3 test_transformer.py
```

### General Advice
The RNN, LSTM, and Transformer models are set up to use CUDA and benefit greatly from using it.
