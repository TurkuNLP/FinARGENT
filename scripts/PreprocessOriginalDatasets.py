import json
import os
import re
from tqdm import tqdm

def main():
    #Load the original datasets
    for filename in os.listdir("data/OriginalDatasets"):
        ds_name = filename[:-6]
        ds_items = []
        with open("data/OriginalDatasets/"+filename, 'r', encoding='utf-8') as reader:
            for l in reader:
                ds_items.append(json.loads(l.strip())['text'])

        #Removing unnecessary bloat (mainly tables and html/css information of scientific documents)
        # Remove text between [TABLE] and [\TABLE] (including the delimiters)
        for i,text in enumerate(ds_items):
            text = re.sub(r'\[TABLE\].*?\[/TABLE\]', '', text, flags=re.DOTALL)
            
            # Remove text between < and > (including the delimiters)
            text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)

            #Remove \n's
            text = re.sub(r'\n+', ' ', text, flags=re.DOTALL)
            
            ds_items[i] = text
        
        # Split the texts into smaller chunks (just rule based and very rudementary method for now...)
        ds_items_snipped = []
        with tqdm(range(len(ds_items)), desc="Processing: "+ds_name) as pbar:
            while len(ds_items) > 0:
                text = ds_items.pop()
                approx_sentences = text.split('. ')
                if len(approx_sentences) > 14:
                    for i in range(0, len(approx_sentences), 15):
                        chunk = approx_sentences[i:i+15]
                        new_text = '.'.join(chunk)
                        if len(new_text) > 100:
                            ds_items_snipped.append(new_text.strip())
                else:
                    chunk = approx_sentences
                    new_text = '.'.join(chunk)
                    if len(new_text) > 100:
                        ds_items_snipped.append(new_text.strip())
                pbar.update()

        to_write=[]
        for i,d in enumerate(ds_items_snipped):
            to_write.append({'id':ds_name+"_"+str(i), 'text':d})

        with open("data/OriginalDatasets_ready/"+filename, 'w', encoding='utf-8') as writer:
            for x in to_write:
                    writer.write(json.dumps(x)+'\n')


    pass



if __name__ == "__main__":
    main()