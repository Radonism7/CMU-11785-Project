bert_baseline.py is the codes for midterm report. 

It contains models, tokenizer, train and evaluate function and all other trivial functions needed to complete the basic sentiment analysis task based on BERT model. The 'bert-base-uncased' is a pretrained model that can be downloaded through torchtext.legacy. 

The dataset we experimented is IMDB movie reviews dataset. It can also be directly called under torchtext, which is convenient to be loaded and splited into train set, validation set, and test set. Also, an data iterator is created to serve as the dataloader that is similar to homework. 
