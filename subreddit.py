from openai import OpenAI
import pandas as pd
from secret_retriever import retrieve_secret
import praw
import prawcore

# initializing openai configuration
BASE_URL = retrieve_secret('cs4973_base_url')
API_KEY=api_key = retrieve_secret('cs4973_api_key')
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

api_client = praw.Reddit(
    client_id=retrieve_secret('praw_client_id'),
    client_secret=retrieve_secret('praw_client_secret'),
    user_agent=retrieve_secret('praw_user_agent')
)

# loading college subreddit dataset
college_subreddits = pd.read_csv('data/college_subreddits.csv')
college_subreddits["name"] = college_subreddits["name"].str.lower()

# get subreddit given a question
def get_subreddit(question):
    subreddit_name = get_subreddit_name(question)
    if subreddit_name is None:
        return None
    subreddit = api_client.subreddit(subreddit_name)
    try:
        if subreddit.description is None:
            # sub does not exist
            return None
    except prawcore.exceptions.Redirect as e:
        # sub does not exist
        return None
    return subreddit

# get subreddit name given a question by getting college name and then mapping to subreddit
def get_subreddit_name(question):
    # initializing system prompt
    college_retrieval_system_prompt = '''
    Given a question, return the full name of the university the question is asking about.
    Return nothing else.

    Examples:
    Question: How do people feel about the dorms at Harvard?
    Answer: Harvard Univeristy
    Question: Do students at Algonquin College like the academic curriculum?
    Answer: Algonquin College
    '''
    messages = [{'role': 'system', 'content': college_retrieval_system_prompt}]
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