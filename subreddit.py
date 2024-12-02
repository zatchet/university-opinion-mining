from openai import OpenAI
import pandas as pd
from secret_retriever import retrieve_secret

# initializing openai configuration
BASE_URL = retrieve_secret('cs4973_base_url')
API_KEY=api_key = retrieve_secret('cs4973_api_key')
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# loading college subreddit dataset
college_subreddits = pd.read_csv('csv_files/college_subreddits.csv')
college_subreddits["name"] = college_subreddits["name"].str.lower()

# initializing system prompt
SYSTEM_PROMPT = '''
Given a question, return the full name of the university the question is asking about.
Return nothing else.

Examples:
Question: How do people feel about the dorms at Harvard?
Answer: Harvard Univeristy
Question: Do students at Algonquin College like the academic curriculum?
Answer: Algonquin College
'''


def get_subreddit_name(question):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.append({'role': 'user', 'content': f'Question: {question}'})

    for _ in range(5):
        resp = client.chat.completions.create(
            messages = messages,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.9,
            max_tokens=500,
            stop=[".", "\n"]
        )
        resp = resp.choices[0].message.content.lower()
        if resp in set(college_subreddits["name"]):
            return college_subreddits[college_subreddits["name"] == resp]["subreddit"].iloc[0]
    return None
