from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


class Question(BaseModel):
    text: str
    max_length: int = 100

class AskResponse(BaseModel):
    answer: str
    question_length: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting... building retriever now")
    app.state.retriever = build_retriever()
    app.state.chain = build_chain()
    yield
    print("Server shutting down")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hybrid-rag-with-fastapi-endpoint.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/check")
def check():
    return {"retriever_type": str(type(app.state.retriever))}


@app.post("/ask", response_model=AskResponse)
def ask(question: Question, request: Request):
    if len(question.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    retriever = request.app.state.retriever
    chain = request.app.state.chain

    docs = retriever.invoke(question.text)
    context = "\n\n".join(d.page_content for d in docs)
    answer = chain.invoke({"context": context, "question": question.text})

    return AskResponse(
        answer=answer,
        question_length=len(question.text)
    )


RESUME_PATH = "Bharath_P_Resume.pdf"

def build_retriever():
    doc = fitz.open(RESUME_PATH)
    text = "".join(page.get_text() for page in doc)

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L12-v2",
        huggingfacehub_api_token=os.environ["HF_TOKEN"]
    )
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)

    bm25 = BM25Retriever.from_texts(chunks)
    bm25.k = 5
    chroma = vectorstore.as_retriever(search_kwargs={"k": 5})

    return EnsembleRetriever(retrievers=[bm25, chroma], weights=[0.5, 0.5])


def build_chain():
    llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="openai/gpt-oss-120b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context. Use tables, bullet points to beautify the answer."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    return prompt | llm | StrOutputParser()
