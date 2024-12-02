from secret_retriever import retrieve_secret
import praw
import prawcore
from openai import OpenAI

# initializing openai configuration
BASE_URL = retrieve_secret('cs4973_base_url')
API_KEY=api_key = retrieve_secret('cs4973_api_key')
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

from subreddit import get_subreddit_name
from fact_opinion_classification.classifier import predict

api_client = praw.Reddit(
    client_id=retrieve_secret('praw_client_id'),
    client_secret=retrieve_secret('praw_client_secret'),
    user_agent=retrieve_secret('praw_user_agent')
)

def remove_uni_name(question):
    SYSTEM_PROMPT = """
    Given the question, return the name of the university or college that is being asked about.
    Return exactly as it appears in the question.

    Question: How do people feel about the hatred against Palestine at BU?
    Answer: BU
    Question: At Harvard University, what is the overall opinion of the party scene?
    Answer: Harvard University
    """

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend([{'role': 'user', 'content': question}])
    resp = client.chat.completions.create(
        messages = messages,
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.7)

    model_response = resp.choices[0].message.content
    return question.replace(model_response, '').strip()

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

def process_query(query: str):
    # remove stop words and other words that are not useful
    with open('stopwords.txt', 'r') as f:
        stopwords = f.read().split('\n')
    # print(query.split())
    return ' '.join([word for word in remove_uni_name(query).split() if word.lower() not in stopwords])

def get_top_post(query: str):
    subreddit = get_subreddit(query)
    if subreddit is None:
        return None
    print("Searching", process_query(query))
    posts = subreddit.search(query=process_query(query), sort='relevance', limit=1000)
    # posts = subreddit.top(limit=1000)
    # filter out posts with no selftext
    posts = [post for post in posts if post.selftext is not None and len(post.selftext) > 20]
    return posts[0]

def get_comments_of_post(post):
    return post.comments

def get_only_opinionated_comments(comments):
    return [comment for comment in comments if predict(comment) == 'opinion']