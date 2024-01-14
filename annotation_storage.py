import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np


class AnnotationStorage:
    def __init__(self, embedding_callback, sentiment_callbac, save_path: Path):
        save_path = Path(save_path)
        self.model = embedding_callback
        self.sentiment_model = sentiment_callbac
        self.annotations = []
        vector_dimension = 1536
        self.index = faiss.IndexFlatIP(vector_dimension)
        self.save_path = save_path
        if os.path.exists(save_path / 'annotations.json'):
            with open(save_path / 'annotations.json', 'r') as f:
                self.annotations = json.load(f)
            with open(save_path / 'index.pickle', 'rb') as f:
                self.index = pickle.load(f)

    def save_state(self):
        with open(self.save_path / 'annotations.json', 'w') as f:
            json.dump(self.annotations, f)
        with open(self.save_path / 'index.pickle', 'wb') as f:
            pickle.dump(self.index, f)

    def add_annotation(self, response, annotation, keyword, model):
        sentiment = self.sentiment_model(annotation)
        response_embedding = np.array([self.model(response)], dtype=np.float32)
        faiss.normalize_L2(response_embedding)
        self.index.add(response_embedding)
        annotation = {
            'annotation': annotation,
            'keyword': keyword,
            'num_votes': 0,
            'index': len(self.annotations),
            'sentiment': sentiment,
            'model': model
            }
        self.annotations.append(annotation)
        self.save_state()
        return annotation
    
    def query_models(self, response):
        response_embedding = np.array([self.model(response)], dtype=np.float32)

        faiss.normalize_L2(response_embedding)
        sims, ann = self.index.search(response_embedding, k=100)
        similarity_indices = ann[0]
        sims = sims[0]
        sims_mask = sims > 0.95
        similarity_indices = similarity_indices[(similarity_indices != -1) & sims_mask]
        similar = [
            {
                'index': i.item(),
                'annotation': self.annotations[i]['annotation'],
                'sentiment': self.annotations[i]['sentiment'],
                'num_votes': self.annotations[i]['num_votes'],
                'model': self.annotations[i]['model']
            } for i in similarity_indices
        ]
        models = {
            'gpt-3.5-turbo-1106': 'Chat GPT 3.5 (latest)',
            'gpt-3.5-turbo': 'Chat GPT 3.5',
            'gpt-4': 'Chat GPT 4'
        }
        counts = { k: 0 for k in models.keys() }
        counts_count = { k: 0 for k in models.keys() }
        for el in similar:
            if el['model'] not in counts:
                counts[el['model']] = 0
                counts_count[el['model']] = 0
            counts[el['model']] += max(0, el['sentiment'])
            counts_count[el['model']] += 1
        result = [{ 'model': k, 'expertise': (counts[k] / counts_count[k]) if counts_count[k] > 0 else -1.0 } for k in counts.keys()]
        print(result)
        return result

    def query_annotation(self, response):
        response_embedding = np.array([self.model(response)], dtype=np.float32)

        faiss.normalize_L2(response_embedding)
        sims, ann = self.index.search(response_embedding, k=4)
        similarity_indices = ann[0]
        sims = sims[0]
        sims_mask = sims > 0.95
        similarity_indices = similarity_indices[(similarity_indices != -1) & sims_mask]

        if len(similarity_indices) > 3:
            votes = np.array([self.annotations[i]['num_votes'] for i in similarity_indices]) 
            votes = votes + abs(min(np.min(votes), 0)) + 1
            votes = votes / np.sum(votes)
            similarity_indices = np.random.choice(similarity_indices, p=votes, size=3, replace=False)

        return [
            {
                'annotation': self.annotations[i]['annotation'],
                'index': i.item(),
                'keyword': self.annotations[i]['keyword'],
                'sentiment': self.annotations[i]['sentiment'],
                'num_votes': self.annotations[i]['num_votes'],
                'model': self.annotations[i]['model']
            } for i in similarity_indices
        ]

    def rate_annotation(self, index, rating):
        self.annotations[index]['num_votes'] += rating
        self.save_state()
