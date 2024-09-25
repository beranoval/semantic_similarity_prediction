import pandas as pd
import ast
import matplotlib.pyplot as plt 
import numpy as np
import itertools
import random
import seaborn as sns
from scipy.spatial.distance import cdist
import gensim
from gensim.models import Word2Vec,FastText
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from gensim.models import KeyedVectors
import multiprocessing
import tqdm
from joblib import Parallel, delayed


class NormOfVector():
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def get_norm_of_embeddings(self):
        embedding_array = self.embeddings.values
        norms = np.linalg.norm(embedding_array, axis=1)
        return norms
    
    def check_if_embeddings_are_normalized(self):
        norms = self.get_norm_of_embeddings()
        is_normalized = np.allclose(norms, 1.0)
        if is_normalized:
            return 
        else:
            return 
    
   


class FastTextEmbeddings:
    def __init__(self, tokenized_corpus, model_name):
        self.tokenized_corpus = tokenized_corpus
        self.model_name = model_name
        
    def train_model(self, window, min_count, seed, sg, vector_size, norm,epochs):
        model = FastText(sentences=self.tokenized_corpus,
                             window=window, 
                             min_count=min_count, 
                             workers=multiprocessing.cpu_count(), 
                             seed=seed, 
                             sg=sg, 
                             vector_size=vector_size,
                             epochs=epochs,
                             word_ngrams=1)
        if norm:
            model.wv.init_sims(replace=True)  # Normalize vectors
        model.save(self.model_name + ".bin")
    
    def get_embeddings(self, words):
        model = FastText.load(self.model_name + ".bin")
        vectors = []
        word_names = []
        for word in words:
            if word in model.wv.key_to_index:
                vectors.append(model.wv[word])
                word_names.append(word)
        vectors_pd = pd.DataFrame(vectors)
        vectors_pd.index = word_names
        return vectors_pd.drop_duplicates()
    
    
    def train_and_get_emb(self, words, window, min_count, seed, sg, vector_size, norm,epochs):
        self.train_model(window, min_count, seed, sg, vector_size, norm, epochs)
        df = self.get_embeddings(words)
        nov = NormOfVector(embeddings=df)
        nov.check_if_embeddings_are_normalized()
        return df

    
class FastTextEmbeddings_1:
    def __init__(self, tokenized_corpus):
        self.tokenized_corpus = tokenized_corpus
        self.model = None
        
    def train_model(self, window, min_count, workers, seed, sg, vector_size, norm):
        self.model = FastText(sentences=self.tokenized_corpus,
                              window=window, 
                              min_count=min_count, 
                              workers=workers, 
                              seed=seed, 
                              sg=sg, 
                              vector_size=vector_size,
                              word_ngrams=1)
        
        if norm:
            self.model.init_sims(replace=True)  # Normalization of vectors

    def get_embeddings(self, words):
        vectors = []
        word_names = []
        for doc in self.tokenized_corpus:
            for word in doc:
                if word in words:
                    if word in self.model.wv.key_to_index:
                        vectors.append(self.model.wv[word])
                        word_names.append(word)

        vectors_pd = pd.DataFrame(vectors)
        vectors_pd.index = word_names
        return vectors_pd.drop_duplicates()
    
    def train_and_get_emb(self, words, window, min_count, workers, seed, sg, vector_size, norm=True):
        self.train_model(window, min_count, workers, seed, sg, vector_size, norm=norm)
        df = self.get_embeddings(words)
        nov = NormOfVector(embeddings=df)
        nov.check_if_embeddings_are_normalized()
        return df

    
class Word2VecEmbeddings:
    def __init__(self, tokenized_corpus, model_name):
        self.tokenized_corpus = tokenized_corpus
        self.model_name = model_name
        
    def train_model(self, window, min_count, seed, sg, vector_size, norm, epochs):
        model = Word2Vec(sentences=self.tokenized_corpus,
                              window=window, 
                              min_count=min_count, 
                              workers=multiprocessing.cpu_count(), 
                              seed=seed, 
                              sg=sg, 
                              vector_size=vector_size,
                              epochs = epochs)
        if norm == True:
            model.init_sims(replace=True) # Normování vektorů
        model.save(self.model_name +".bin")
    
    def get_embeddings(self, words):
        model = Word2Vec.load(self.model_name +".bin")
        vectors = []
        word_names = []
        for word in words:
            if word in model.wv.key_to_index:
                vectors.append(model.wv[word])
                word_names.append(word)
        vectors_pd = pd.DataFrame(vectors)
        vectors_pd.index = word_names
        return vectors_pd.drop_duplicates()
    
    
    def train_and_get_emb(self, words, window, min_count,seed, sg,vector_size, norm,epochs):
        self.train_model(window, min_count,seed, sg,vector_size, norm, epochs)
        df = self.get_embeddings(words)
        nov = NormOfVector(embeddings = df)
        nov.check_if_embeddings_are_normalized()
        return df  
    


class Similarity2:
    def __init__(self, embeddings_df):
        self.embeddings_df = embeddings_df
    
    
    def get_cosine_similarity_of_all_words(self):
        word_names = self.embeddings_df.index
        embeddings_matrix = self.embeddings_df.values
        cos_sim_matrix = cosine_similarity(embeddings_matrix)
        mask = np.triu(np.ones(cos_sim_matrix.shape), k=1).astype(bool)
        indices = np.column_stack(np.where(mask))
        similarities = cos_sim_matrix[mask]
        df_final = pd.DataFrame({
            "similarity": similarities,
            "First": word_names[indices[:, 0]],
            "Last": word_names[indices[:, 1]]})
        return df_final
    
    
    def get_cosine_similarity_of_all_words2(self):
        
        def calculate_cosine_similarity(embeddings_matrix, word_names, mask, indices):
            cos_sim_matrix = cosine_similarity(embeddings_matrix)
            similarities = cos_sim_matrix[mask]
            return similarities, word_names[indices[:, 0]], word_names[indices[:, 1]]
        
        word_names = self.embeddings_df.index
        embeddings_matrix = self.embeddings_df.values
        mask = np.triu(np.ones((len(word_names), len(word_names))), k=1).astype(bool)
        indices = np.column_stack(np.where(mask))
        num_jobs = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_jobs)(
            delayed(calculate_cosine_similarity)(
                embeddings_matrix, word_names, mask, indices
            ) for _ in range(num_jobs))
        similarities = np.concatenate([r[0] for r in results])
        first_words = np.concatenate([r[1] for r in results])
        last_words = np.concatenate([r[2] for r in results])
        df_final = pd.DataFrame({
            "similarity": similarities,
            "First": first_words,
            "Last": last_words
        })
        return df_final

    
    def get_euclidean_similarity_of_all_words(self):
        word_names = self.embeddings_df.index
        embeddings_matrix = self.embeddings_df.values
        euclidean_sim_matrix = np.zeros((len(word_names), len(word_names)))
        for i in range(len(word_names)):
            for j in range(i + 1, len(word_names)):
                euclidean_sim_matrix[i, j] = euclidean(embeddings_matrix[i], embeddings_matrix[j])
        mask = np.triu(np.ones(euclidean_sim_matrix.shape), k=1).astype(bool)
        indices = np.column_stack(np.where(mask))
        similarities = euclidean_sim_matrix[mask]
        df_final = pd.DataFrame({
            "similarity": similarities,
            "First": word_names[indices[:, 0]],
            "Last": word_names[indices[:, 1]]
        })
        return df_final

     
    

class Similarity:
    def __init__(self, embeddings_df):
        self.embeddings_df = embeddings_df
    
    def get_cosine_similarity_of_all_words(self):
        cos_sim = []
        word_combinations = []
        word_names = self.embeddings_df.index.tolist()
        for i in range(len(word_names)):
            for j in range(i+1, len(word_names)):
                word_comb = f"{word_names[i]},{word_names[j]}"
                word_combinations.append(word_comb)
                vec1 = self.embeddings_df.loc[word_names[i]].values.reshape(1, -1)
                vec2 = self.embeddings_df.loc[word_names[j]].values.reshape(1, -1)
                similarity = cosine_similarity(vec1, vec2)[0][0]
                cos_sim.append(similarity)
        df_final = pd.DataFrame({"similarity": cos_sim, "words": word_combinations})
        df_final[["First", "Last"]] = df_final["words"].str.split(",", expand=True)
        df_final = df_final[["similarity", "First", "Last"]]
        return df_final
    
    def get_euclidean_similarity_of_all_words(self):
        euclidean_sim = []
        word_combinations = []

        word_names = self.embeddings_df.index.tolist()

        for i in range(len(word_names)):
            for j in range(i+1, len(word_names)):
                word_comb = f"{word_names[i]},{word_names[j]}"
                word_combinations.append(word_comb)

                vec1 = self.embeddings_df.loc[word_names[i]].values
                vec2 = self.embeddings_df.loc[word_names[j]].values

                similarity = euclidean(vec1, vec2)
                euclidean_sim.append(similarity)

        df_final = pd.DataFrame({"similarity": euclidean_sim, "words": word_combinations})
        df_final[["First", "Last"]] = df_final["words"].str.split(",", expand=True)
        df_final = df_final[["similarity", "First", "Last"]]

        return df_final

    
    
def plot_emb(embeddings, with_desc = True):
    embedding_array = embeddings.values
    num_dimensions = embedding_array.shape[1]
    if num_dimensions == 2:
        x = embedding_array[:, 0]
        y = embedding_array[:, 1]
        plt.figure(figsize=(12, 12))
        plt.scatter(x, y)
        plt.title("Visualization of Word Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")    
        if with_desc:
            for i, word in enumerate(embeddings.index):
                plt.annotate(word, (x[i], y[i]), alpha=0.7)    
        plt.show()
    else:
        tsne_model = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne_model.fit_transform(embedding_array)
        x = tsne_embeddings[:, 0]
        y = tsne_embeddings[:, 1]
        plt.figure(figsize=(12, 12))
        plt.scatter(x, y)
        plt.title("TSNE Visualization of Word Embeddings")
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")
        if with_desc:
            for i, word in enumerate(embeddings.index):
                plt.annotate(word, (x[i], y[i]), alpha=0.7)
        plt.show()

