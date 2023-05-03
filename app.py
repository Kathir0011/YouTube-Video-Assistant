import gradio as gr

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def create_db_from_video_url(video_url, api_key):
    """
    Creates an Embedding of the Video and makes it suitable for similarity searching.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    loader = YoutubeLoader.from_youtube_url(video_url)
    transcripts = loader.load()
    # cannot provide this directly to the model so we are splitting the transcripts into small chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcripts)

    db = FAISS.from_documents(docs, embedding=embeddings)

    return db

def get_response(api_key, video_url, request):
    """
    Usind gpt-3.5-turbo to obtain the response. It can handle upto 4096 tokens.
    """

    db = create_db_from_video_url(video_url, api_key)

    docs = db.similarity_search(query=request, k=4)
    docs_content = " ".join([doc.page_content for doc in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=api_key)

    # creating a template for request
    template = """
    You are an assistant that can answer questions about youtube videos based on
    video transcripts: {docs}

    Only use factual information from the transcript to answer the question.

    If you don't have enough information to answer the question, say "I don't know".

    Your Answers should be detailed.
    """

    system_msg_prompt = SystemMessagePromptTemplate.from_template(template)

    # human prompt
    human_template = "Answer the following questions: {question}"
    human_msg_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_msg_prompt, human_msg_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=request, docs=docs_content)

    return response

# creating title, description for the web app
title = "YouTube Video Assistant 🧑‍💻"
description = "Answers to the Questions asked by the user on the specified YouTube video. (English Only)"
article = "Other Projects:\n"\
          "💰 [Health Insurance Predictor](http://health-insurance-cost-predictor-k19.streamlit.app/)\n"\
          "📰 [Fake News Detector](https://fake-news-detector-k19.streamlit.app/)\n"\
          "🪶 [Birds Classifier](https://huggingface.co/spaces/Kathir0011/Birds_Classification)"
# building the app
youtube_video_assistant = gr.Interface(
    fn=get_response,
    inputs=[gr.Text(label="Enter the OpenAI API Key:", placeholder=f"Example: sk-{'*' * 45}AgM"), 
            gr.Text(label="Enter the Youtube Video URL:", placeholder="Example: https://www.youtube.com/watch?v=MnDudvCyWpc"),
            gr.Text(label="Enter your Question", placeholder="Example: What's the video is about?")],
    outputs=gr.TextArea(label="Answers using gpt-3.5-turbo:"),
    title=title,
    description=description,
    article=article
)

# launching the web app
youtube_video_assistant.launch()
