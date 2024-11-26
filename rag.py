from secret_retriever import retrieve_secret
import praw
import prawcore

api_client = praw.Reddit(
    client_id=retrieve_secret('reddit_client_id'),
    client_secret=retrieve_secret('reddit_client_secret'),
    user_agent=retrieve_secret('reddit_user_agent')
)

def get_subreddit(subreddit_name):
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
    words_to_remove = stopwords + ['university', 'college']
    return ' '.join([word for word in query.split() if word not in words_to_remove])

def get_top_post(processed_query: str, subreddit_name: str):
    subreddit = get_subreddit(subreddit_name)
    if subreddit is None:
        return None
    posts = subreddit.search(query=processed_query, sort='top', limit=1000)
    # posts = subreddit.top(limit=1000)
    # filter out posts with no selftext
    posts = [post for post in posts if post.selftext is not None and len(post.selftext) > 20]
    return posts[0]

def get_comments_of_post(post):
    return post.comments

def get_only_opinionated_comments(comments):
    # need BERT fine tune for this
    pass