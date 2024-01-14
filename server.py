import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from annotation_storage import AnnotationStorage
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pydantic import BaseModel
import math
import random
from dataclasses import dataclass, field
from typing import Any, List
import numpy as np
import json

client = OpenAI(
    organization='org-iDnTJYhL78qGMpGFWY2J5tox',
    api_key='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'
)



app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding(text: str):
    return client.embeddings.create(input = [text.replace('\n', ' ')], model="text-embedding-ada-002").data[0].embedding


def get_keyword(text: str):
    query = f"""Task:
Write a short keyword with a maximum of three words that catches the sentiment of the following review.

Review:
{text}"""
    keyword = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": query}]
        ).choices[0].message.content.split(' ')[0].replace('\"', '').replace(',', '').replace(' ', '')
    
    return keyword

def get_sentiment(text: str):
    query = f"""Task:
Answer in one word if the following review has a positive, neutral, or negative sentiment.

Review:
{text}"""
    
    conversion = {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    }
    sentiment = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": query}]
        ).choices[0].message.content.replace('\n', ' ').split(' ')[0].replace('\"', '').replace(',', '').replace(' ', '').lower()
    
    print(sentiment)
    
    if sentiment in conversion.keys():
        return conversion[sentiment]
    else: 
        return 0


storage = AnnotationStorage(get_embedding, get_sentiment, '.')

class Annotation(BaseModel):
    query_id: int
    annotation: str


class Query(BaseModel):
    query: str

@dataclass
class QueryStorageElement:
    query: str
    model: str
    response: str = ""
    confidence: List[float] = field(default_factory=list)
    done: bool = False

class QueryStorage:
    def __init__(self, path='query_storage.pickle'):
        self.storage = {}
        self.path = path
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.storage = pickle.load(f)

    def get(self, name):
        return self.storage[name]

    def set(self, name, value):
        self.storage[name] = value
        self.save()
    
    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.storage, f)

    def __len__(self):
        return len(self.storage)

query_storage = QueryStorage()


@app.post("/query/{model}")
def query_model(query: Query, model: str = 'gpt-4'):
    new_id = len(query_storage)
    query_storage.set(new_id, QueryStorageElement(query=query.query, model=model))
    return { 'queryId': new_id }

@app.get("/query/{queryId}/response")
def stream_query_model(queryId: int):
    async def stream_response():
        queryStorageElement = query_storage.get(queryId)
        stream = client.chat.completions.create(
            model=queryStorageElement.model,
            messages=[{"role": "user", "content": queryStorageElement.query}],
            stream=True,
            logprobs=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                queryStorageElement.response += chunk.choices[0].delta.content
                queryStorageElement.confidence.extend([el.logprob for el in chunk.choices[0].logprobs.content])
                yield chunk.choices[0].delta.content
        queryStorageElement.done = True
        query_storage.save()
    return StreamingResponse(stream_response())

@app.get("/query/{queryId}/confidence")
def confidence(queryId: int):
    queryStorageElement = query_storage.get(queryId)
    if not queryStorageElement.done:
        raise HTTPException(status_code=404, detail="Query not done yet")
    confidence = 0
    for logprob in queryStorageElement.confidence:
        confidence += math.exp(logprob)
    confidence /= len(queryStorageElement.confidence)
    return { 'confidence': confidence }

@app.post("/upvote/{index}")
def upvote(index: int):
    storage.rate_annotation(index, 1)

@app.post("/downvote/{index}")
def downvote(index: int):
    storage.rate_annotation(index, -1)

@app.get("/query_annotation/{index}")
def get_annotation(index: int):
    response = query_storage.get(index).response
    return storage.query_annotation(response)

@app.get("/get_expertise_models/{query_id}")
def get_expertise_models(query_id: int):
    models = ['Chat GPT 4', 'Chat GPT 3.5', 'Chat GPT 3.5 (latest)']
    expertise_levels = [round(random.uniform(0.5, 0.96), 2) for _ in range(len(models))]
    return [{'model': model, 'expertise': expertise} for model, expertise in sorted(zip(models, expertise_levels), key=lambda x: x[1], reverse=True)]


@app.post("/add_annotation")
def add_annotation(annotation: Annotation):
    query_storage_element = query_storage.get(annotation.query_id)
    response = query_storage_element.response
    return storage.add_annotation(response, annotation.annotation, get_keyword(annotation.annotation))
