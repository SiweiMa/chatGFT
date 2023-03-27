import os
import numpy as np
import pandas as pd
import openai
from openai.embeddings_utils import distances_from_embeddings
import tiktoken
import gradio as gr

openai.api_key = os.environ.get('OPENAI_KEY')


df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

class chatGFT:
    def __init__(self, df, max_context_len=3000):
        self.chat_history = [{"role": "system", "content": "We have provided context information below. if the question can't be answered based on the context, say \"I don't know\". If the question can be answered based on the context, please be accurate and concise."},
                             {"role": "system", "content": ""}]
        self.df = df
        self.max_len = 4000
        self.max_context_len = max_context_len
        self.cur_len = 0
        self.context = []
        self.link = []

    def create_context(
        self, input, size="ada"
    ):
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(input=input, engine='text-embedding-ada-002')['data'][0]['embedding']

        # Get the distances from the embeddings
        self.df['distances'] = distances_from_embeddings(q_embeddings, self.df['embeddings'].values, distance_metric='cosine')

        self.context = []
        self.link = []
        context_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for _, row in self.df.sort_values('distances', ascending=True).iterrows():
            # Add the length of the text to the current length
            # If the context is too long, break
            if context_len + row['n_tokens'] + 4 > self.max_context_len:
                break
            else:
                context_len += row['n_tokens'] + 4  
                # Else add it to the text that is being returned
                self.context.append(row["text"])
                self.link.append(row['links'])


        # Return the context
        return self.get_context()#"\n\n###\n\n".join(self.context)


    def get_output(self, input):
        """
        Get the AI model's response to the user's input

        input: str
            the user's input
        
        Returns:
        None
        """
        # append user's input to the chat history
        self.chat_history.append({"role": "user", "content": f"{input}"})

        # check if the maximum token has been reached, if yes exit the function
        if self.check_max_len(' '.join([i['content'] for i in self.chat_history if i['role'] != 'system'])):
            return self.chat_history.append({"role": "assistant", "content": "The maximum token was reached. Please restart the conversation."}) 
 
        # update the AI model's context with the most similar text from the data
        self.chat_history[1] = {"role": "system", "content": f"{self.create_context(input)}"}
        # self.chat_history[1] = {"role": "system", "content": f"{self.get_context()}"}

        # generate the model's response to the user's input
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=self.chat_history,
        temperature=0
        )        
        reply_content = completion.choices[0].message.content#.replace('```python', '<pre>').replace('```', '</pre>')

        self.chat_history.append({"role": "assistant", "content": f"{reply_content}"}) 


    def get_chatbot_output(self, input):
        """
        Get the chatbot's output

        input: str
            the user's input

        Returns:
        tuple: a tuple the chat history and generated context
        """
        self.get_output(input)
        print(self.chat_history)
        chat_history_content = [i['content'] for i in self.chat_history if i['role'] != 'system']
        chat_history_content_output = [(chat_history_content[i], chat_history_content[i+1]) for i in range(0, len(chat_history_content)-1, 2)]
        return chat_history_content_output, self.get_souce() # convert to tuples of list

    def get_souce(self):
        source = []
        result = list(zip(self.link, self.context))
        for index, item in enumerate(result):
            source.append(f'[{index+1}] {item[0]}: {item[1][:200]}....')
        return "\n".join(source)

    def get_context(self):
        source = []
        result = list(zip(self.link, self.context))
        for index, item in enumerate(result):
            source.append(f'[{index+1}] {item[0]}: {item[1]}')
        return "\n".join(source)

    def count_token(self, text):
        """
        Get the number of tokens in the input text

        text: str
            the input text

        Returns:
        int: the number of tokens
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
    
    def check_max_len(self, text):
        """
        Check if the maximum token has been reached

        text: str
            the text to check the token count

        Returns:
        bool: True if the maximum token has been reached, False otherwise
        """
        self.cur_len += self.count_token(text)
        if self.cur_len > self.max_len - self.max_context_len:
            self.chat_history = self.chat_history[:2] + [self.chat_history[-1]]
            self.cur_len = 0
            return True
        
    def reboot_bot(self):
        self.cur_len = 0
        self.context = []
        self.link = []
        self.chat_history = [{"role": "system", "content": "We have provided context information below. if the question can't be answered based on the context, say \"I don't know\". If the question can be answered based on the context, please be accurate and concise."},
                             {"role": "system", "content": ""}]

with gr.Blocks(title="chatGFT") as demo: 
    
    chatGFTbot = chatGFT(df, 1000)

    with gr.Column(): 
        chatbot = gr.Chatbot() 
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        clear = gr.Button("Clear")
        source = gr.Textbox(label='Sources', placeholder="Sources")


    txt.submit(chatGFTbot.get_chatbot_output, txt, [chatbot, source]) # submit(function, input, output)
    txt.submit(None, None, txt, _js="() => {''}") # No function, no input to that function, submit action to textbox is a js function that returns empty string, so it clears immediately.
    clear.click(chatGFTbot.reboot_bot, None, None)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    demo.launch()