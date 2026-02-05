import os
import random as rand
import re
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import sys

#Helper methods

def getWordFormDicts():
    lex_data_df = pd.read_csv("data/TCBLex/whole_with_features_csv/Whole.csv", sep=';', keep_default_na=False)
    lex_data_df = lex_data_df[['text', 'lemma', 'upos+features']]

    lex_data_df_words = lex_data_df[['text', 'lemma']].groupby(['text','lemma'], as_index=False).count()

    lex_data_words2lemmas = dict(zip(lex_data_df_words['text'].to_numpy().tolist(), lex_data_df_words['lemma'].to_numpy().tolist()))
    lex_data_lemmas2words = {x:[] for x in lex_data_words2lemmas.values()}
    for x in lex_data_words2lemmas:
        lemma = lex_data_words2lemmas[x]
        l = lex_data_lemmas2words[lemma]
        if x.lower() not in l:
            l.append(x.lower())
        lex_data_lemmas2words[lemma] = l
    return lex_data_words2lemmas, lex_data_lemmas2words

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def generateProbabilityDistribution(num_items, min_prob, max_prob):
    # Function that "guarantees" varying levels of corruption ends up in the training dataset
    # This is done, because we only get one corruption per sample, so it's likely we would only get quite corrupted versions...
    return list(np.linspace(min_prob, max_prob, num_items))

def capitalize(text):
    # A very simple function that fixes some outrageous capitalization mistakes that can result from swapping places of words
    # DOES NOT FIX, JUST MAKES IT SLIGHTLY BETTER
    punc_filter = re.compile('([.!?;]\\s*)')
    split_with_punctuation = punc_filter.split(text)
    for i,j in enumerate(split_with_punctuation):
        if len(j) > 1:
            split_with_punctuation[i] = j[0].upper() + j[1:]
    text = ''.join(split_with_punctuation)
    return text

#Corruption methods
def corruptWordOrder(sample_dict, probability, get_all_corrupted_forms=False):
    text = sample_dict['text']
    corrupted_text = text
    #Since we just want a lot of corrupted samples, keep track of every corrupted version we ever see and return all of them
    if get_all_corrupted_forms:
        returnable = []
    corruptions = []
    #First pick up only clean words
    words = text.split(" ")
    for i, w in enumerate(words):
        if not w.isalpha() and len(w)>0:
            if w[-1] in ['!', '?', ',', '.']:
                words[i] = w[:-1]
            else:
                words[i] = ""
    for w in words:
        if len(w) < 2:
            words.remove(w)
    # Iterate through words to find candidates for swapping
    for i in range(len(words) - 1):
        if rand.random() <= probability:
            # Get the current word and next word with their original formatting
            word1 = words[i]
            word2 = words[i+1]
            
            # Find the position of this word pair in the text
            word_pair = word1 + " " + word2
            swapped_pair = word2 + " " + word1
            
            # Replace only the first occurrence starting from where we expect the pair to be
            # This helps avoid replacing the wrong instances if the same word appears multiple times
            position = corrupted_text.find(word_pair)
            if position != -1:
                # Create a new string with the swapped pair
                corrupted_text = corrupted_text[:position] + swapped_pair + corrupted_text[position + len(word_pair):]

                _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
                corrupted_text = _RE_COMBINE_WHITESPACE.sub(" ", corrupted_text).strip()
                corrupted_text = capitalize(corrupted_text)
                corruptions.append(word1+"<->"+word2)
                if get_all_corrupted_forms:
                    returnable.append({'id':sample_dict['id'], 'text':corrupted_text, 'corruptions':corruptions.copy()})
    if get_all_corrupted_forms:
        return returnable
    else:
        return {'id':sample_dict['id'], 'text':corrupted_text, 'corruptions':corruptions.copy()}

def corruptWordForm(sample_dict, probability, lex_data_words2lemmas, lex_data_lemmas2words, max_words_to_try=None, get_all_corrupted_forms=False):
    text = sample_dict['text']
    corrupted_text = text
    #IF we want all corrupted forms:
    if get_all_corrupted_forms:
        #Since we just want a lot of corrupted samples, keep track of every corrupted version we ever see and return all of them
        returnable = []
    #First pick up only clean words
    words = text.split(" ")
    for i, w in enumerate(words):
        if not w.isalpha() and len(w)>0:
            if w[-1] in ['!', '?', ',', '.']:
                words[i] = w[:-1]
            else:
                words[i] = ""
    for w in words:
        if len(w) < 2:
            words.remove(w)
    #Additional metadata
    corruptions = sample_dict.get('corruptions', []).copy()
    #print('Splitting words time:', round(t2-t1, 2), 's')
    if max_words_to_try:
        words = list(rand.sample(words, min(max_words_to_try, len(words))))
    #Then start iterating over words and making transformations
    for w in words:
        #According to prob
        if rand.random()<=probability:
            lemma = lex_data_words2lemmas.get(w, '')
            if len(lemma) > 0:
                forms = lex_data_lemmas2words[lemma]+[lemma]
                if len(forms) > 1:
                    while w.lower() in forms:
                        forms.remove(w.lower())
                    if len(forms) == 0:
                        break
                    transformation = rand.sample(forms, 1)[0]
                    transformation = transformation.replace('#', '')
                    if w[0].isupper():
                        transformation = transformation.capitalize()
                    corrupted_text = corrupted_text.replace(" "+w, " "+transformation, 1)
                    corruptions.append(w+" --> "+transformation)
                    if get_all_corrupted_forms:
                        returnable.append({'id':sample_dict['id'], 'text':corrupted_text, 'corruptions':corruptions.copy()})
    if get_all_corrupted_forms:
        return returnable
    #IF we only want the one corrupted sample (where every word has a chance of corruption:)
    else:
        return {'id':sample_dict['id'], 'text':corrupted_text, 'corruptions':corruptions.copy()}
    

def applyCorruptions(gold_texts, min_probability, max_probability):
    corrupted_ds_items = []
    # 
    corruption_level_distr = generateProbabilityDistribution(len(gold_texts), min_probability, max_probability)
    w2l_dict, l2w_dict = getWordFormDicts()
    #Shuffle to introduce some randomness
    rand.shuffle(corruption_level_distr)
    with tqdm(range(len(gold_texts)), desc="Applying corruption methods...") as pbar:
        for d in gold_texts:
            prob = corruption_level_distr.pop()
            corr_d = corruptWordOrder(d, prob)
            corr_d = corruptWordForm(corr_d, prob, w2l_dict, l2w_dict)
            corrupted_ds_items.append(corr_d)
            pbar.update()
    return corrupted_ds_items

def main(cmd_args):

    gold_ds_folder = cmd_args[0]
    min_probability = float(cmd_args[1])
    max_probability = float(cmd_args[2])
    ds_name = cmd_args[3]
    ds_items = []
    filenames = []
    # Loading all the gold standard datasets (intended to be "very good quality" / human written texts)
    for ds_file in absoluteFilePaths(gold_ds_folder):
        filename = Path(ds_file).stem
        filenames.append(filename)
        with open(ds_file, 'r', encoding='utf-8') as reader:
            for i, line in enumerate(reader):
                item_id = filename+"_"+str(i)
                # We expect the data to be either jsonl OR raw text per line
                try:
                    file_contents = json.loads(line.strip())['text']
                except:
                    file_contents = line.strip()
                ds_items.append({'id':item_id, 'text':file_contents})
    corr_ds_items = applyCorruptions(ds_items, min_probability, max_probability)

    with open("data/CorruptedDatasets_cache/"+ds_name+".jsonl", 'w', encoding="utf-8") as writer:
        for i in range(len(corr_ds_items)):
            gold = ds_items[i]
            corr = corr_ds_items[i]
            assert gold['id'] == corr['id']
            to_write = {
                'id':gold['id'],
                'ref_text':gold['text'],
                'text':corr['text']
            }
            writer.write(json.dumps(to_write)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])