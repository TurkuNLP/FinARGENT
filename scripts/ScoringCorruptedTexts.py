from bleurt import score
from tqdm import tqdm
import sys
import json

def main(cmd_args):
    ds_name = cmd_args[0]
    corr_ds = []
    with open("data/CorruptedDatasets_cache/"+ds_name+".jsonl", 'r', encoding='utf-8') as reader:
        for l in reader:
            if len(l)>0:
                corr_ds.append(json.loads(l.strip()))
    #BLEURT-score calculations
    checkpoint = "BLEURT-20"

    scorer = score.BleurtScorer(checkpoint)
    with tqdm(range(len(corr_ds))) as pbar:
        for i, x in enumerate(corr_ds):
            s = scorer.score(references=[x['ref_text']], candidates=[x['text']])[0]
            x['bleurt_score'] = min(1,s)
            corr_ds[i] = x
            pbar.update()
    
    with open("data/CorruptedDatasets_ready/"+ds_name+".jsonl", 'w', encoding='utf-8') as writer:
        for l in corr_ds:
            writer.write(json.dumps(l)+'\n')

if __name__ == "__main__":
    main(sys.argv[1:])