from openai import OpenAI
import pandas as pd
from tqdm import tqdm

import os
os.chdir('..')
from secret_retriever import retrieve_secret
os.chdir('benchmarks')

# initializing openai configuration
BASE_URL = retrieve_secret('cs4973_base_url')
API_KEY=api_key = retrieve_secret('cs4973_api_key')
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# initializing system prompt for question generation
SYSTEM_PROMPT = '''
Come up with a hypothetical question that would be relevant to the university given. 
Return only the question. Be sure to include the name of the university in the question.

Examples:
Question: Boston Univeristy
Answer: How do people feel about the dorms at Boston University?
Question: University of Washington
Answer: At the udub, do people like the quality of the food?
Question: Algonquin College
Answer: Do students at Algonquin like the academic curriculum?
'''

# convert college subreddits csv to dataframe
df = pd.read_csv('../data/college_subreddits.csv')
df.drop(columns=['location'], inplace=True)


def generate_for_benchmark():
    # generating 1 question per college subreddit
    benchmark_df = pd.DataFrame(columns=['subreddit', 'question'])
    for index, row in tqdm(df.iterrows()):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages.append({'role': 'user', 'content': row['name']})

        resp = client.chat.completions.create(
            messages = messages,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.9,
            max_tokens=500,
        )
        resp = resp.choices[0].message.content
        benchmark_df.loc[len(benchmark_df)] = {'subreddit': row['name'], 'question': resp}
        
    # post-processing of questions and saving to file
    benchmark_df.drop_duplicates(subset=['question'], inplace=True)
    benchmark_df.to_csv('subreddit_benchmark.csv', index=False)
    
    return benchmark_df