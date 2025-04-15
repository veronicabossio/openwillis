"""
This script processes a speech readability dataset (CLEAR) to prepare augmented excerpts for downstream analysis. 
It performs the following steps:
1. Loads and bins grade levels into educational groups (Kindergarten through Adult).
2. Samples 100 excerpts per group to ensure a balanced dataset.
3. Converts each excerpt to a token-level pseudo-JSON format including POS tags and timing estimates (openwillis-compatible JSON format).
4. Subsamples truncated versions of the excerpts at random lengths to simulate varying speech lengths.
5. Outputs the final augmented dataset to aCSV.

Dependencies: nltk, numpy, pandas
"""

import random

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

PATH_TO_CLEAR = '/Users/veronicabossio/Library/Mobile Documents/com~apple~CloudDocs/brooklyn_health/data/CLEAR/CLEAR.csv'
PATH_TO_OUTPUT = '/Users/veronicabossio/Library/Mobile Documents/com~apple~CloudDocs/brooklyn_health/data/CLEAR/augmented_CLEAR.csv'

df = pd.read_csv(PATH_TO_CLEAR)

# bin grade level into 5 groups and get a balanced sample of 100 excerpts per grade level
df = df[['ID','Flesch-Kincaid-Grade-Level','Flesch-Reading-Ease','Excerpt']]
df['Grade Level'] = pd.cut(df['Flesch-Kincaid-Grade-Level'], [0,4,8,12,16,50],labels = ['Kindergarten','Elementary','Teen','College','Adult'])

# subset a balanced sample of 100 excerpts from each grade level
random.seed(10)
grades = ['Kindergarten','Elementary','Teen','College','Adult']
grades_ls = np.array([random.sample(df[df['Grade Level'] == grade]['ID'].values.tolist(), 100)
                      for grade in grades]).flatten()
df = df[df['ID'].isin(grades_ls)]

# some functions
def get_word_type(tag):
    if tag.startswith('J'):
        return 'Adjective'
    elif tag.startswith('V'):
        return 'Verb'
    elif tag.startswith('N'):
        return 'Noun'
    elif tag.startswith('R'):
        return 'Adverb'
    else:
        return 'Other'

def transcript_to_json(transcript):
    """
    convert a transcript string to openwillis-compatible JSON-like format
    by simulating the timing of each word
    """
    word_list = word_tokenize(transcript)

    pos_tagged = nltk.pos_tag(word_list)

    json_data = []
    start_time = 0.0
    for i, (word, tag) in enumerate(pos_tagged):
        end_time = round(start_time + random.uniform(0.2, 0.4), 2)
        json_data.append({
            'conf': 1.0,
            'end': end_time,
            'start': start_time,
            'word': word,
            'old_idx': i,
            'tag': get_word_type(tag)
        })
        start_time = end_time

    return json_data

def subsample_excerpt_by_length(df, min_len, max_len, num_samples=4):
    """
    Subsample excerpts by length. Each excerpt is a truncated substring 
     of the original CLEAR excerpt, starting at the beginning and ending
        at a random length between min_len and max_len.   
    """    
    
    truncated_rows_uniform = []
    for _, row in df.iterrows():
        text = row['Excerpt']
        if not isinstance(text, str):
            continue
        words = text.split()
        max_len_ex = len(words)
        for _ in range(num_samples):
            length = random.randint(min_len, min(max_len, max_len_ex))
            truncated_text = " ".join(words[:length])
            new_row = row.copy()
            new_row['Excerpt'] = truncated_text
            new_row['Excerpt_Length'] = length
            new_row['Truncated'] = True
            truncated_rows_uniform.append(new_row)

    df_truncated_uniform = pd.DataFrame(truncated_rows_uniform)

    # Update original full-length rows
    df['Excerpt_Length'] = df['Excerpt'].apply(lambda x: len(x.split()) if isinstance(x, str) else None)
    df['Truncated'] = False

    # Combine both
    subsampled_df = pd.concat([df, df_truncated_uniform], ignore_index=True)
    return subsampled_df


# create json
df['json_output'] = df['Excerpt'].apply(lambda x: transcript_to_json(x))
df = df.reset_index(drop=True)

diff_length_df = subsample_excerpt_by_length(df, 5, 200, num_samples=5)
#diff_length_df.to_csv(PATH_TO_OUTPUT, index=False)
diff_length_df.to_pickle(PATH_TO_OUTPUT.replace('.csv', '.pkl'))