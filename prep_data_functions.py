import pandas as pd
import ast
import matplotlib.pyplot as plt 
import numpy as np
import itertools
import random
import seaborn as sns
from scipy.spatial.distance import cdist
import gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


class ExperimentalDF:
    def __init__(self, num_documents=None, document_length=None):
        self.num_documents = num_documents
        self.document_length = document_length
        

    def define_first_df(self):
        """
        Defines the first data frame with unique values.

        Parameters:
        - num_documents: Number of documents in the corpus. Optional if df is provided.
        - document_length: Number of words in each document (all documents have the same length). Optional if df is provided.
        """
        data = {'Document': [], 'Word': []}

        for i in range(self.num_documents):
            for j in range(self.document_length):
                data['Document'].append(f'Doc{i+1}')
                data['Word'].append(f'Word{j+1 + i*self.document_length}')

        df = pd.DataFrame(data)
        return df
    
    def define_same_context_df(self):
        """
        Vytvoří experimentální datový rámec s odlišnou logikou.
        Každý dokument obsahuje poslední slovo předchozího dokumentu.

        Parametry:
        - num_documents: Počet dokumentů v korpusu.
        - document_length: Počet slov v každém dokumentu (všechny dokumenty mají stejnou délku).

        Návrat:
        - Tokenizovaný korpus.
        """
        data = {'Document': [], 'Word': []}

        for i in range(self.num_documents):
            current_document = f'Doc{i+1}'
            for j in range(self.document_length):
                data['Document'].append(current_document)
                # Každý dokument obsahuje všechna slova od Word1 do WordX
                data['Word'].append(f'Word{j+1 + i*self.document_length}')

            # Přidá poslední slovo předchozího dokumentu jako první slovo následujícího dokumentu
            next_document = f'Doc{i+2}' if i+2 <= self.num_documents else 'Doc1'
            data['Document'].append(next_document)
            data['Word'].append(f'Word{self.document_length + i*self.document_length} Word1 {next_document}')



        df = pd.DataFrame(data)
        
        def process_word_column(word_column):
            if 'Doc' in word_column:
                return word_column.split('Doc')[0].split()[0]
            else:
                return word_column

        # Aplikace funkce na sloupec 'Word'
        df['Word'] = df['Word'].apply(process_word_column)
        
        df = df[df["Document"] != "Doc1"]
        
        return df





    def duplicate_co_occurrences(self, dict_index_count, df=None):
        """
        Duplicates co-occurrences by duplicating chosen documents.

        Parameters:
        - dict_index_count: Dictionary specifying how many times each document should be duplicated.
                            Key: Document index, Value: Number of duplications.
        - df: Optional external dataframe to duplicate co-occurrences. If not provided, the class df will be used.
        """
        if df is None:
            df = self.define_first_df()

        duplicated_df = pd.DataFrame(columns=df.columns)
        for doc_index, count in dict_index_count.items():
            selected_docs = df[df['Document'] == f'Doc{doc_index}']       
            for i in range(count):
                selected_docs['Document'] = f'Doc{doc_index}_{i+2}'  # Vytvoření nového indexu pro duplikované dokumenty
                duplicated_df = pd.concat([duplicated_df, selected_docs], ignore_index=True)
        df = pd.concat([df, duplicated_df], ignore_index=True)
        return df
    
    def duplicate_co_occurrences2(self, dict_index_count, df=None):
        """
        Duplicates co-occurrences by duplicating chosen documents.

        Parameters:
        - dict_index_count: Dictionary specifying how many times each document should be duplicated.
                            Key: Document index, Value: Number of duplications.
        - df: Optional external dataframe to duplicate co-occurrences. If not provided, the class df will be used.
        """
        if df is None:
            df = self.define_first_df()

        def duplicate_rows(group):
            doc_index = int(group['Document'].iloc[0].split('Doc')[-1])
            count = dict_index_count.get(doc_index, 0)
            duplicated_rows = [group] * count
            for i, dup_row in enumerate(duplicated_rows):
                dup_row['Document'] = f'Doc{doc_index}_{i+2}'
            return pd.concat(duplicated_rows, ignore_index=True)

        duplicated_df = df.groupby('Document', group_keys=False).apply(duplicate_rows)
        df = pd.concat([df, duplicated_df], ignore_index=True)
        return df


    def aggregate_documents(self, df):
        """
        Aggregates the words in each document.
        """
        df = df.groupby('Document')['Word'].agg(', '.join).reset_index().rename(columns={'Word': 'document'})
        df.set_index('Document', inplace=True)
        
        return df

    def tokenize_data(self, string_col, df):
        """
        Tokenizes the values in the specified string column.

        Parameters:
        - string_col: Name of the string column to tokenize.

        Returns:
        - List of tokenized documents.
        """
        corpus = df[string_col].tolist()
        lst_corpus = []
        for string in corpus:
            lst_words = string.split(",")
            lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
            k = []
            for i in lst_grams:
                j = i.replace(' ', '')
                k.append(j)
            lst_grams = k
            lst_corpus.append(lst_grams)
        return lst_corpus
    
    def create_experiment_wo_dupl(self):
        """
        Creates the experimental data frame by utilizing all functions except duplication. 

        Parameters:
        - num_documents: Number of documents in the corpus. Optional if df is provided.
        - document_length: Number of words in each document (all documents have the same length). Optional if df is provided.
        - tokenize: Boolean indicating whether to tokenize the 'document' column. Default is False.

        Returns:
        - Tokenized corpus if tokenize=True, otherwise None.
        """
        df = self.define_first_df()
        df = self.aggregate_documents(df)
        df = self.tokenize_data('document', df)
        return df
        
    def create_experiment(self, dict_index_count, df=None):
        """
        Creates the experimental data frame by utilizing all functions.

        Parameters:
        - dict_index_count: Dictionary specifying how many times each document should be duplicated.
                            Key: Document index, Value: Number of duplications.
        - tokenize: Boolean indicating whether to tokenize the 'document' column. Default is False.
        - df: Optional external dataframe. If provided, num_documents and document_length will be ignored.
        - num_documents: Number of documents in the corpus. Optional if df is provided.
        - document_length: Number of words in each document (all documents have the same length). Optional if df is provided.

        Returns:
        - Tokenized corpus if tokenize=True, otherwise None.
        """
        if df is None:
            df = self.define_first_df()
        df = self.duplicate_co_occurrences(dict_index_count, df)
        df = self.aggregate_documents(df)
        df = self.tokenize_data('document', df)
        return df
    
    def create_experiment_same_context(self, dict_index_count, df=None):
        if df is None:
            df = self.define_same_context_df()
        df = self.duplicate_co_occurrences2(dict_index_count, df)
        df = self.aggregate_documents(df)
        df = self.tokenize_data('document', df)
        return df

    
