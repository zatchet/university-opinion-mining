from openai import OpenAI

import retrieval
from secret_retriever import retrieve_secret

# initializing openai configuration
BASE_URL = retrieve_secret('cs4973_base_url')
API_KEY=api_key = retrieve_secret('cs4973_api_key')
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

class Agent:
    def __init__(self):
        self.conversation = []

    def get_sentiment_and_post(self, query: str):
        post = retrieval.get_top_post(query)
        if post is None:
            return (None, 'Could not find any content')
        comments = retrieval.get_comments_of_post(post)
        opinionated_comments = retrieval.get_only_opinionated_comments(comments)
        return (post.shortlink, self.get_sentiment(opinionated_comments, post, query))

    def get_sentiment(self, opinionated_comments, post, query):
        sentiment_system_prompt = """
        Given the post and its comments as context, determine the overall opinion on the given question.

        Return ONLY "positive", "negative", or "neutral" as the answer. Nothing else.
        """

        chunked_comments = retrieval.generate_chunks(opinionated_comments[::-1])
        chunked_post = retrieval.generate_chunks([post.title, post.selftext])

        messages = [{'role': 'system', 'content': sentiment_system_prompt}]
        messages.extend([{'role': 'user', 'content': 'COMMENTS:'}])
        messages.extend([{'role': 'user', 'content': chunk} for chunk in chunked_comments])
        messages.extend([{'role': 'user', 'content': 'POST:'}])
        messages.extend([{'role': 'user', 'content': chunk} for chunk in chunked_post])
        messages.extend([{'role': 'user', 'content': f'QUESTION: {query}'}])

        # print("HERE")
        resp = client.chat.completions.create(
            messages = messages,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.7)
        
        response = resp.choices[0].message.content
        self.conversation = messages[1:]
        self.conversation.extend([{'role': 'assistant', 'content': response}])
        return response

    def followup_question(self, question):
        followup_system_prompt = """
        Using only the post and comments that have been provided, answer the user's follow-up question.
        """
        self.conversation.extend([{'role': 'user', 'content': question}])
        resp = client.chat.completions.create(
            messages = [{'role': 'system', 'content': followup_system_prompt}] + self.conversation,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.7)
        
        response = resp.choices[0].message.content
        self.conversation.append({'role': 'assistant', 'content': response})
        return response