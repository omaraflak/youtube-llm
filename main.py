import fire
from typing import Optional
from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


@dataclass
class ChatResponse:
    text: str
    links: list[str]


def load_vector_db() -> Chroma:
    return Chroma(
        persist_directory="chroma",
        embedding_function=HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            query_instruction="Represent the query for retrieval: ",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
    )


def insert_in_vector_db(documents: list[Document], batch_size: int = 150):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    page_splits = text_splitter.split_documents(documents)
    database = load_vector_db()
    for i in range(0, len(page_splits), batch_size):
        database.add_documents(documents=page_splits[i : i + batch_size])


def load_chain() -> ConversationalRetrievalChain:
    template = """
    You are a helpful AI assistant that provides answers based on the following context:

    {context}

    Question: {question}
    Answer:
    """.strip()

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    return ConversationalRetrievalChain.from_llm(
        llm=HuggingFacePipeline.from_model_id(
            model_id="tiiuae/falcon-7b-instruct",
            task="text-generation",
            model_kwargs={"max_length": 300, "do_sample": False, "trust_remote_code": True}
        ),
        retriever=load_vector_db().as_retriever(),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def run_chain(chain: ConversationalRetrievalChain, query: str) -> ChatResponse:
    response = chain({"question": query})
    return ChatResponse(
        response["answer"],
        [source.metadata["link"] for source in response["source_documents"]]
    )


def documents_from_youtube(youtube_uid: str) -> list[Document]:
    link = lambda t: f"https://www.youtube.com/watch?v={youtube_uid}&t={t}s"
    return [
        Document(
            page_content=json.get("text"),
            metadata={"link": link(int(json.get("start")))}
        )
        for json in YouTubeTranscriptApi.get_transcript(youtube_uid)
    ]


def chat_loop():
    chain = load_chain()
    while True:
        query = input("query: ")
        if query == "exit":
            return
        print(run_chain(chain, query), end="\n\n")


def main(youtube_uid: Optional[str] = None):
    if youtube_uid:
        insert_in_vector_db(documents_from_youtube(youtube_uid))
    chat_loop()


if __name__ == "__main__":
    fire.Fire(main)