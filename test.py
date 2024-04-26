import spacy
from spacy.matcher import Matcher
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth',200)

candidate_sentences = pd.read_csv('wiki_sentences_v2.csv')
candidate_sentences.shape


# 텍스트 예시
text = "Barack Obama was born in Hawaii. He was the 44th president of the United States."

# spaCy를 사용하여 문장을 파싱
doc = nlp(text)

# Matcher 객체 생성
matcher = Matcher(nlp.vocab)

# 패턴 정의: "PERSON born in PLACE"
pattern = [{"ENT_TYPE": "PERSON"}, {"LOWER": "born"}, {"ENT_TYPE": "GPE"}]

# Matcher에 패턴 추가
matcher.add("BORN_PATTERN", [pattern])

# 매칭된 결과 저장할 리스트 생성
matches = matcher(doc)

# 그래프 생성
G = nx.Graph()

for match_id, start, end in matches:
    span = doc[start:end]
    if "PERSON" in [ent.label_ for ent in span.ents] and "GPE" in [ent.label_ for ent in span.ents]:
        person_entity = [ent.text for ent in span.ents if ent.label_ == "PERSON"][0]
        place_entity = [ent.text for ent in span.ents if ent.label_ == "GPE"][0]
        G.add_node(person_entity)
        G.add_node(place_entity)
        G.add_edge(person_entity, place_entity)

# 그래프 시각화
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=10, font_weight="bold")
plt.show()
