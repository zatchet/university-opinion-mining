import gradio as gr
from agent import Agent
import os

FOLLOWUP_PROMPT = "Do you have any follow-up questions? If so, ask them here. Otherwise, type 'no'."
START_OVER_PROMPT = "Sure! Let's start over. Ask me about public opinion related to a topic in a university."
INITIAL_PROMPT = "Ask me about public opinion related to a topic in a university."
NOT_FOUND_RESPONSE = "Could not find any content related to that topic. Please try again."

conversation = []
agent = Agent()

def get_sentiment_and_post(query):
    url, sentiment = agent.get_sentiment_and_post(query)
    print(query, sentiment, url)
    if url is None:
        agent.conversation = []
        return NOT_FOUND_RESPONSE
    response = f'Sentiment: {sentiment}\n\n[Link to the post]({url})\n\n{FOLLOWUP_PROMPT}'
    # conversation.append((query, response))
    return response

def followup_question(question):
    response = f'{agent.followup_question(question)}\n\n{FOLLOWUP_PROMPT}'
    # conversation.append((question, response))
    return response

# Define the function that interacts with the agent
def interact_with_agent(user_input):
    global conversation
    if user_input.lower() == "no":
        response = START_OVER_PROMPT
        agent.conversation = []
    elif agent.conversation == []:
        response = get_sentiment_and_post(user_input)
    else:
        response = followup_question(user_input)

    conversation.append((user_input, response))
    # Format the conversation for the Chatbot component
    formatted_conversation = [(msg[0], msg[1]) for msg in conversation]
    return formatted_conversation, ""

# Define the function to clear the conversation
def clear_conversation():
    global conversation
    conversation = []
    agent.conversation = []
    return gr.update(value=[])

with gr.Blocks(title="Reddit Sentiment Analysis") as interface:
    gr.Markdown("# Reddit Sentiment Analysis")
    gr.Markdown(f'## {INITIAL_PROMPT}')
    
    chatbot = gr.Chatbot()
    user_input = gr.Textbox()
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear")

    user_input.submit(interact_with_agent, inputs=user_input, outputs=[chatbot, user_input])
    submit_button.click(interact_with_agent, inputs=user_input, outputs=[chatbot, user_input])
    clear_button.click(clear_conversation, outputs=chatbot)

if __name__ == "__main__":
    interface.launch()