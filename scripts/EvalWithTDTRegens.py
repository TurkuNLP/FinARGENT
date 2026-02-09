#Imports
from datasets import Dataset, DatasetDict
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn import metrics
import numpy as np
import os
from pprint import pprint

def separateDatasets(regen_ds_name):
    ds_items = []
    with open("data/EvaluationDatasets/"+regen_ds_name+".jsonl", 'r', encoding='utf-8') as reader:
        for l in reader:
            if len(l)>0:
                ds_items.append(json.loads(l.strip()))

    uniq_keys = set([x['model']+"_"+x['effort']+"_"+x['generation_round'] for x in ds_items])
    dss = {x:[] for x in uniq_keys}
    for x in ds_items:
        key = x['model']+"_"+x['effort']+"_"+x['generation_round']
        temp = dss[key]
        temp.append(x)
        dss[key] = temp

    ds_list = {x:Dataset.from_list(dss[x]) for x in dss}

    return ds_list


def main():

    ds_dict = separateDatasets("TDT_regens_eval_set")

    os.environ['WANDB_MODE'] = 'disabled'
    #Initializing model (regression BERT)
    MODEL_NAME = "FinARGENT_v-0-1/TurkuNLP_bert-base-finnish-cased-v1"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #Data collator so that everything works as intended :)
    data_collator = DataCollatorWithPadding(tokenizer)

    def tokenize(ex):
        return tokenizer(
            ex['text'],
            max_length=512,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
    
    tok_ds_dict = {x:ds_dict[x].map(tokenize, num_proc=len(os.sched_getaffinity(0)), batched=True) for x in ds_dict}

    trainer = Trainer(
        model=model,
        processing_class = tokenizer,
        data_collator = data_collator,
    )


    preds = {x:trainer.predict(tok_ds_dict[x]) for x in tok_ds_dict}
    with open("Eval_results.txt") as writer:
        for x in preds:
            writer.write(f'Predictions for {x} are on average: {np.mean(preds[x].predictions)}\n')
            writer.write(f'Prediction std for {x} is: {np.std(preds[x].predictions)}\n')
            writer.write(f'Prediction min for {x} is: {np.min(preds[x].predictions)}\n')
            writer.write(f'Prediction max for {x} is: {np.max(preds[x].predictions)}\n')
            writer.write('\n\n')


if __name__ == "__main__":
    main()