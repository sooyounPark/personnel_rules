import re
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import networkx as nx
from tqdm import tqdm
from spacy.matcher import Matcher

from spacy.tokens import Span
pd.set_option('display.max_colwidth', 200)
nlp = spacy.load('en_core_web_sm')
candidate_sentences = pd.read_csv('wiki_sentences_v2.csv')


def get_entities(sent):
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""
    prv_tok_text = ""
    prefix = ""
    modifier = ""

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
        #############################################################

    return [ent1.strip(), ent2.strip()]

entity_pairs = []
for i in tqdm(candidate_sentences["sentence"]):
    entity_pairs.append(get_entities(i))

    entity_pairs[10:20]
def get_relations(sent):
    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': '?'},
               {'DEP': 'agent', 'OP': '?'},
               {'POS': 'ADJ', 'OP': '?'}]
    matcher.add("matching_1", [pattern])  # 수정된 부분
    matches = matcher(doc)
    print('matches:', matches)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]
    return(span.text)

get_relations("John completed the task")


relations = [get_relations(i) for i in tqdm(candidate_sentences['sentence'])]

source = [i[0] for i in entity_pairs]
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
plt.show()

G=nx.from_pandas_edgelist(kg_df[kg_df['edge'] == "completed by"], 'source', 'target',
                          edge_attr=True,create_using=nx.MultiDiGraph())

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G,k=0.5)
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap =plt.cm.Blues, pos=pos)
specific_edges = [(row['source'], row['target']) for idx, row in kg_df.iterrows() if row['edge'] == 'completed by']
nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: 'completed by' for edge in specific_edges})
plt.show()


G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="produced by"], "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()