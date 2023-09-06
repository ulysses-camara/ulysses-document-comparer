from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class BERT():
    def __init__(self)->None:
        pass
    def set_model_file(self, model_file:str)->None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
    def return_entities(self, tokens:list)->list:
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            add_special_tokens = False,
            is_split_into_words = True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

        '''Loss used as some of score, as a placeholder'''
        labels = predicted_token_class_ids
        loss = self.model(**inputs, labels=labels).loss
        loss = round(loss.item(), 5)

        #remove punctuation at the end since it will never truly be part of the entity, in order to simplify
        #the code to extract the entities

        current_entity = []
        entities = []
        word_ids = inputs.word_ids()
        whole_tokens = tokens

        i = 0
        while(i < len(predicted_tokens_classes)):
            label = predicted_tokens_classes[i]
            if(len(current_entity) > 0 and (label == "O" or label[:2] == "B-")):
                entities.append([" ".join(current_entity),current_label_class[2:],loss])
                current_entity = []
            if(label != "O"):
                current_label_class = label
                id = word_ids[i]
                sub_words_ids = [j for j in range(i,len(word_ids)) if word_ids[j] == id]
                i += (len(sub_words_ids) - 1)
                current_entity.append(self.truncate_punctuation(whole_tokens[id]))
            i += 1
        if(len(current_entity) > 0): #Accounting for the case when sentence ends with an entity (no O or B- after)
            entities.append([" ".join(current_entity),current_label_class[2:],loss])

        return entities
    def truncate_punctuation(entity: str):
            if(entity[-1] in [',','.']):
                return entity[:-1]
            else:
                return entity
