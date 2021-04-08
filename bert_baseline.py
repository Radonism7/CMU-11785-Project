import torch
import random
import numpy as np
import time
from transformers import BertTokenizer
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

### Global Random Seed ###
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

BATCH_SIZE = 64


################# Helper Function #################

### Tokenize sentences into words collection ###
def tokenize_into_word(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

### Data Loader ###
def load_data_iterator(train_data, valid_data, test_data, batch_size, device):
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size, 
        device = device)
    return train_iterator, valid_iterator, test_iterator

### Binary output accuracy calculation ### 
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

### Calculate time to complete an epoch ###
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################# Train Eval Function #####################
def train(model, train_iterator, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for batch in train_iterator:
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += acc.item()
        
    return running_loss / len(train_iterator), running_acc / len(train_iterator)

def evaluate(model, valid_iterator, criterion):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for batch in valid_iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            running_loss += loss.item()
            running_acc += acc.item()
        
    return running_loss / len(valid_iterator), running_acc / len(valid_iterator) 

######################## Model ############################## 
class BERT_GRU(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # Dont update parameters in pretrained bert model
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        _, hidden = self.rnn(embedded) 
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        output = self.linear(hidden)
        
        return output


######################### Load IMDB data ###################### Approximately 5 mins
TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_into_word,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = tokenizer.cls_token_id,
                  eos_token = tokenizer.sep_token_id,
                  pad_token = tokenizer.pad_token_id,
                  unk_token = tokenizer.unk_token_id)

LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print("Number of training examples: ", len(train_data))
print("Number of validation examples: ", len(valid_data))
print("Number of testing examples: ", len(test_data))
print("*** Complete Splitting Data ***")

LABEL.build_vocab(train_data)
# print(LABEL.vocab.stoi) # print expected output label

##################### Load data into data iterator ##########################
train_iterator, valid_iterator, test_iterator = load_data_iterator(train_data,
                                                                   valid_data,
                                                                   test_data,
                                                                   BATCH_SIZE,
                                                                   device)

print("*** Complete Loading Data ***")


#################### Define Model ####################
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
pretrained_bert = BertModel.from_pretrained('bert-base-uncased')

model = BERT_GRU(pretrained_bert,
                 HIDDEN_DIM,
                 OUTPUT_DIM,
                 N_LAYERS,
                 BIDIRECTIONAL,
                 DROPOUT)
### Dont update pretrained bert model ###
for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

print(model)


############################# Train epochs ##########################
N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')