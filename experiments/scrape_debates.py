from utils.scraping import get_multilingual_transcript
import pandas as pd 
import re 
import json
import argparse
import os
from tqdm import tqdm

def get_debate_urls(term, meta_data_file):

    # load meta data
    with open(meta_data_file, 'r') as infile:
        data = json.load(infile)

    # store activities in dataframe 
    cres = []
    for d in data:
        if 'CRE' in d:
            cres.extend(d['CRE'])
    cre_df = pd.DataFrame(cres)
    term = cre_df[cre_df['term'] == term]

    debate_urls = term[term['title'].str.contains('debate')]['url'].map(lambda x: x.split('&')[0])
    return sorted(set(debate_urls))

def scrape_debate(url, overwrite = False):
    """
    scrape a debate and store each interventio
    """
    
    debate_id = "-".join(url.split('+')[1:-3])

    if not os.path.exists(f'../data/debates/{debate_id}/interventions'):
        os.makedirs(f'../data/debates/{debate_id}/interventions')
    elif not overwrite: 
        return None

    try: 
        data = get_multilingual_transcript(url)
    except Exception as e:
        print(f"Error scraping debate {url}: {e}")
        return None
    
    if data is None:
        return None
    

    for intervention_id, meta in data.items():
        meta['debate_id'] = debate_id
        meta['intervention_id'] = intervention_id
        with open(f'../data/debates/{debate_id}/interventions/{intervention_id}.json', 'w') as outfile:
            json.dump(meta, outfile)
    return data
    
def main(term, meta_data_file, num_debates = 100, overwrite = False):
    debate_urls = get_debate_urls(term, meta_data_file)
    for url in tqdm(debate_urls[:num_debates]):
        scrape_debate(url, overwrite)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--term', type=int, default=6)
    args.add_argument('--meta_data_file', type=str, default='../data/ep_mep_activities.json')
    args.add_argument('--num_debates', type=int, default=100)
    args.add_argument('--overwrite', action='store_true')
    args = args.parse_args()
    main(args.term, args.meta_data_file, args.num_debates, args.overwrite)