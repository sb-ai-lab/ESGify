##ESGify
---
## Main information
We introduce the model for multilabel ESG risks classification. There is 47 classes methodology with granularial risk definition.   

## Usage 
```python
from collections import OrderedDict
from transformers import MPNetPreTrainedModel, MPNetModel, AutoTokenizer
import torch
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Definition of ESGify class because of custom,sentence-transformers like, mean pooling function and classifier head
class ESGify(MPNetPreTrainedModel):
    """Model for Classification ESG risks from text."""

    def __init__(self,config): #tuning only the head
        """
        """
        super().__init__(config)
        # Instantiate Parts of model
        self.mpnet = MPNetModel(config,add_pooling_layer=False)
        self.id2label =  config.id2label
        self.label2id =  config.label2id
        self.classifier = torch.nn.Sequential(OrderedDict([('norm',torch.nn.BatchNorm1d(768)),
                                                ('linear',torch.nn.Linear(768,512)),
                                                ('act',torch.nn.ReLU()),
                                                ('batch_n',torch.nn.BatchNorm1d(512)),
                                                ('drop_class', torch.nn.Dropout(0.2)),
                                                ('class_l',torch.nn.Linear(512 ,47))]))


    def forward(self, input_ids, attention_mask):


         # Feed input to mpnet model
        outputs = self.mpnet(input_ids=input_ids,
                             attention_mask=attention_mask)
         
        # mean pooling dataset and eed input to classifier to compute logits
        logits = self.classifier( mean_pooling(outputs['last_hidden_state'],attention_mask))
         
        # apply sigmoid
        logits  = 1.0 / (1.0 + torch.exp(-logits))
        return logits

model = ESGify.from_pretrained('ai-lab/ESGify')
tokenizer = AutoTokenizer.from_pretrained('ai-lab/ESGify')
texts = ['text1','text2']
to_model = tokenizer.batch_encode_plus(
                  texts,
                  add_special_tokens=True,
                  max_length=512,
                  return_token_type_ids=False,
                  padding="max_length",
                  truncation=True,
                  return_attention_mask=True,
                  return_tensors='pt',
                )
results = model(**to_model)


# We also recommend preprocess texts with using FLAIR model

from flair.data import Sentence
from flair.nn import Classifier
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
tagger = Classifier.load('ner-ontonotes-large')
tag_list = ['FAC','LOC','ORG','PERSON']
texts_with_masks = []
for example_sent in texts:
    filtered_sentence = []
    word_tokens = word_tokenize(example_sent)
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    for w in word_tokens:
        if w.lower() not in stop_words:
            filtered_sentence.append(w)
    # make a sentence
    sentence = Sentence(' '.join(filtered_sentence))
    # run NER over sentence
    tagger.predict(sentence)
    sent = ' '.join(filtered_sentence)
    k = 0
    new_string = ''
    start_t = 0 
    for i in sentence.get_labels():
        info = i.to_dict()
        val = info['value']
        if info['confidence']>0.8 and val in tag_list : 

            if i.data_point.start_position>start_t :
                new_string+=sent[start_t:i.data_point.start_position]
            start_t = i.data_point.end_position
            new_string+= f'<{val}>'
    new_string+=sent[start_t:-1]
    texts_with_masks.append(new_string)

to_model = tokenizer.batch_encode_plus(
                  texts_with_masks,
                  add_special_tokens=True,
                  max_length=512,
                  return_token_type_ids=False,
                  padding="max_length",
                  truncation=True,
                  return_attention_mask=True,
                  return_tensors='pt',
                )
results = model(**to_model)
```

------

## Background

The project aims to develop the ESG Risks classification model with a custom ESG risks definition methodology. 


## Training procedure

### Pre-training 

We use the pretrained [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) model. 
Next, we do the domain-adaptation procedure by Mask Language Modeling pertaining with using texts of ESG reports. 


#### Training data

We use the ESG news dataset of 2000 texts with manually annotation of ESG specialists.
