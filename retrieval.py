from secret_retriever import retrieve_secret
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from fact_opinion_classification.classifier import predict
from subreddit import get_subreddit

# initializing openai configuration
BASE_URL = retrieve_secret('cs4973_base_url')
API_KEY=api_key = retrieve_secret('cs4973_api_key')
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# get the top post for a given query
def get_top_post(query: str):
    subreddit = get_subreddit(query)
    if subreddit is None:
        return None
    processed_query = process_query(query)
    posts = subreddit.search(query=processed_query, time_filter='all', sort='relevance', limit=1000)

    # filter out posts with no text or very short text
    posts = [post for post in posts if post.selftext is not None and len(post.selftext) > 20]
    if posts == []:
        return None
    return posts[0]

# process the query to remove stop words and other words that are not useful
def process_query(query: str):
    # remove stop words and other words that are not useful
    with open('data/stopwords.txt', 'r') as f:
        stopwords = f.read().split('\n')
    return ' '.join([word for word in remove_uni_name(query).split() if word.lower() not in stopwords])

# remove the name of the university from the question
def remove_uni_name(question):
    remove_uni_system_prompt = """
    Given the question, return the name of the university or college that is being asked about.
    Return exactly as it appears in the question.

    Question: How do people feel about the hatred against Palestine at BU?
    Answer: BU
    Question: At Harvard University, what is the overall opinion of the party scene?
    Answer: Harvard University
    """
    question = question.lower()

    messages = [{'role': 'system', 'content': remove_uni_system_prompt}]
    messages.extend([{'role': 'user', 'content': question}])

    for _ in range(5):
        resp = client.chat.completions.create(
            messages = messages,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.7)
        potential_uni_name = resp.choices[0].message.content
        question = question.replace(potential_uni_name.lower(), '')
    return question.strip()

# get the comments of a post
def get_comments_of_post(post):
    return [comment.body for comment in post.comments if comment.body is not None and '[deleted]' not in comment.body]

# filter to only keep the opinionated comments
def get_only_opinionated_comments(comments: list):
    return [comment for comment in comments if predict(comment) == 'opinion']

# generate chunks of text given a list of documents
def generate_chunks(context_documents: list):
    context = '....'.join(context_documents)
    text_splitter = CharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 0,
        length_function=len,
        separator = '\n\n'
    )
    return text_splitter.split_text(context)