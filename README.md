# MusicRNN

Note: Install requirements.txt before proceeding with the following.

## Instructions to Run the Programs ON TEST DATA 
First cd into the final_proj/ folder

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

### To Run Transformer
Run the following command:
```
python3 test_transformer.py
```
