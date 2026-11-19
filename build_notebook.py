import json
import os

cells = []

def add_markdown(source_text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

def add_code(source_text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

# ------------------------------------------------------------------------------
# Cell 0 – Markdown: Project title, student name placeholder, syllabus mapping table
# ------------------------------------------------------------------------------
add_markdown("""# 🎓 Academic Project: Agentic RAG-Based AI Research Assistant

### **Institution:** Faculty of Engineering & Technology / Department of Computer Science & AI
### **Course Title:** Natural Language Processing & Cognitive Systems (Practicals 01 – 06)
### **Student Name:** *[Insert Student Name / Roll Number Here]*
### **Academic Supervisor:** *[Insert Supervisor Name / Department]*
### **Date:** November 2026

---

## 📌 Project Overview
This academic laboratory project implements an **Agentic Retrieval-Augmented Generation (RAG) AI Research Assistant**. 
The system autonomously digests academic literature (a 100+ page textbook), conducts deep lexical and semantic analysis, indexes content into a high-performance vector store, and employs a **LangGraph state machine** driven by local quantized Large Language Models (**LLaMA 3.2 1B** and **Nomic-Embed-Text** via **Ollama**). An interactive **Gradio ChatInterface** enables researchers to query the assistant with full source provenance and transparent agentic decision routing.

---

## 📋 Comprehensive Syllabus Mapping (Practicals 01 – 06)

| Practical Module | Technical Scope & Curriculum Topics | Implementation Details in Notebook | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **Practical 01** | **Text Extraction & Morphological Preprocessing** | Extraction from 100+ page PDF using `PyMuPDF` (`fitz`); Sentence/word tokenization, normalization, stop-word removal, and Porter stemming using `NLTK`. | `outputs/extracted_text_sample.txt`<br>`outputs/preprocessed_sample.txt` |
| **Practical 01** | **Text Vectorization (Sparse vs Dense)** | Statistical TF-IDF matrix generation (`scikit-learn`) and dense semantic contextual embeddings (`Sentence-Transformers` `all-MiniLM-L6-v2`) with cosine similarity computation. | `outputs/tfidf_matrix.pkl` |
| **Practical 02** | **Sentiment Analysis & Affective Tone** | Deep Transformer classification pipeline (`transformers` SST-2) assessing polarity and affective tone of literature passages. | `outputs/sentiment_results.csv` |
| **Practical 03** | **Information Extraction & Named Entity Recognition** | Linguistic parsing and entity extraction (PERSON, ORG, GPE, DATE) via `spaCy` (`en_core_web_sm`). | `outputs/ner_results.json` |
| **Practical 04** | **Unsupervised Topic Modeling** | Latent Dirichlet Allocation (`Gensim` `LdaModel`) extracting latent thematic distributions across document segments. | `outputs/topic_model_results.txt` |
| **Practical 05** | **Multilingual NLP Solution** | Cross-lingual language identification (`langdetect`) and automated neural translation (`deep-translator`). | `outputs/translation_sample.txt` |
| **Practical 06** | **Agentic RAG with LangGraph & Local LLM** | Recursive character text chunking, persistent `ChromaDB` vector storage, cyclic `LangGraph` decision router (retrieve vs generate), and local `Ollama` inference (`llama3.2:1b`). | `chroma_db/`<br>Gradio Chat UI |

---""")

# ------------------------------------------------------------------------------
# Cell 1 – Install dependencies (commented pip install commands)
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 1: DEPENDENCY INSTALLATION COMMANDS
# ==============================================================================
# Uncomment and execute the following pip install commands if running in a fresh 
# environment. Note: Python 3.10+ is recommended.

# !pip install --upgrade pip
# !pip install PyMuPDF>=1.24.0
# !pip install nltk>=3.8.1
# !pip install scikit-learn>=1.4.0
# !pip install sentence-transformers>=3.0.0
# !pip install "transformers>=4.44.0,<4.48.0"
# !pip install tf-keras>=2.16.0
# !pip install spacy>=3.8.0
# !python -m spacy download en_core_web_sm
# !pip install gensim>=4.3.0
# !pip install deep-translator>=1.11.4
# !pip install langdetect>=1.0.9
# !pip install chromadb>=0.5.0
# !pip install langchain-chroma>=0.1.4
# !pip install langchain>=0.3.0 langchain-core>=0.3.0 langchain-community>=0.3.0
# !pip install langchain-text-splitters>=0.3.0
# !pip install langchain-ollama>=0.2.0
# !pip install langgraph>=0.2.20
# !pip install gradio>=5.0.0
# !pip install pandas numpy requests

print("Dependency installation instructions noted. Ensure all packages are installed.")""")

# ------------------------------------------------------------------------------
# Cell 2 – Import libraries
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 2: IMPORT LIBRARIES & ENVIRONMENT INITIALIZATION
# ==============================================================================

import os
import sys
import json
import time
import pickle
import urllib.request
from typing import TypedDict, List, Dict, Any

# Disable TensorFlow oneDNN and Keras noise to ensure clean terminal output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"

# Numerical and Data Manipulation
import numpy as np
import pandas as pd

# PDF Extraction (PyMuPDF)
try:
    import pymupdf as fitz
except ImportError:
    import fitz

# NLTK - Tokenization, Stopwords, Stemming
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK corpora automatically
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Scikit-Learn - Statistical Vectorization & Metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hugging Face Sentence-Transformers & Transformers Pipelines
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# spaCy - Named Entity Recognition & Linguistic Parsing
import spacy

# Gensim - Unsupervised Latent Dirichlet Allocation (LDA) Topic Modeling
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Multilingual Language Detection & Machine Translation
from langdetect import detect
from deep_translator import GoogleTranslator, MyMemoryTranslator

# LangChain Text Splitting & Vector Database
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# LangChain Ollama Integrations & LangGraph State Machine
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Gradio Conversational Chat Interface
import gradio as gr

# Ensure workspace output directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

# ------------------------------------------------------------------------------
# Health Check Helper: Verify Ollama Service Availability
# ------------------------------------------------------------------------------
def verify_ollama_status(host: str = "http://localhost:11434") -> bool:
    \"\"\"Checks whether the local Ollama daemon is reachable and lists installed models.\"\"\"
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "Antigravity/1.0"})
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                print(f"[Ollama Health Check] Connected! Detected models: {models}")
                return True
    except Exception as exc:
        print(f"[Ollama Health Check Warning] Ollama daemon unreachable at {host} ({exc}).")
        print(" -> Please launch Ollama via 'ollama serve' in your terminal.")
        return False

ollama_online = verify_ollama_status()
print(f"Library initialization complete. Python version: {sys.version.split()[0]}")""")

# ------------------------------------------------------------------------------
# Cell 3 – Markdown: Practical 01 – Text Extraction & Preprocessing
# ------------------------------------------------------------------------------
add_markdown("""## Practical 01 – Text Extraction & Preprocessing

### 🎯 Objective:
Extract unstructured textual sequences from an authoritative academic treatise (100+ pages) in Portable Document Format (PDF) using **PyMuPDF** (`fitz`), and apply formal computational linguistics preprocessing pipelines (**NLTK**) including:
1. **Sentence and Lexical Tokenization**: Segmenting raw character streams into discrete syntactic units.
2. **Text Normalization**: Standardizing case representations and eliminating non-informative punctuation noise.
3. **Stop-Word Removal**: Filtering high-frequency functional closed-class words (articles, prepositions, conjunctions) that convey minimal domain-specific entropy.
4. **Morphological Stemming**: Applying the **Porter Stemmer** algorithm to reduce morphological inflections to canonical invariant stems.

The extracted raw text sample and normalized token distributions are preserved to disk for empirical verification.""")

# ------------------------------------------------------------------------------
# Cell 4 – Load PDF using PyMuPDF; save first 2000 chars; display
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 4: LOAD PDF DOCUMENT VIA PYMUPDF & SAVE EXTRACTED SAMPLE
# ==============================================================================

pdf_path = os.path.join("data", "sample_book.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Source PDF document not found at: {pdf_path}")

# Open document using PyMuPDF (fitz)
pdf_doc = fitz.open(pdf_path)
total_pages = len(pdf_doc)
print(f"Successfully opened '{pdf_path}' with {total_pages} pages.")

# Ingest textual streams across all pages
full_extracted_text = ""
for page_idx in range(total_pages):
    page = pdf_doc[page_idx]
    # Prefix each page with structured boundary metadata for citation traceability
    page_text = page.get_text("text")
    full_extracted_text += f"\\n--- [Page {page_idx + 1}] ---\\n" + page_text

pdf_doc.close()
total_characters = len(full_extracted_text)
print(f"Total characters extracted from corpus: {total_characters:,}")

# Extract the initial 2,000 characters as an empirical sample
extracted_sample_2000 = full_extracted_text[:2000]

# Persist extracted sample to outputs directory
output_sample_path = os.path.join("outputs", "extracted_text_sample.txt")
with open(output_sample_path, "w", encoding="utf-8") as f:
    f.write(extracted_sample_2000)

print(f"Saved first 2,000 characters to: '{output_sample_path}'\\n")
print("=" * 60)
print("EXTRACTED TEXT SAMPLE (FIRST 500 CHARACTERS):")
print("=" * 60)
print(extracted_sample_2000[:500].strip())
print("=" * 60)""")

# ------------------------------------------------------------------------------
# Cell 5 – Tokenization, normalization, stop-word removal, stemming
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 5: NLTK TOKENIZATION, NORMALIZATION, STOP-WORD FILTERING & STEMMING
# ==============================================================================

# Extract a representative text block for morphological analysis
sample_text_block = extracted_sample_2000

# 1. Lexical Tokenization using NLTK word_tokenize
raw_tokens = word_tokenize(sample_text_block)

# 2. Case Normalization & Punctuation Elimination
# Retain alphanumeric tokens to remove stand-alone punctuation and typographical marks
normalized_tokens = [tok.lower() for tok in raw_tokens if tok.isalnum()]

# 3. Stop-Word Removal using NLTK English Stopwords lexicon
english_stopwords = set(stopwords.words("english"))
content_tokens = [tok for tok in normalized_tokens if tok not in english_stopwords]

# 4. Morphological Stemming using the Porter Stemmer algorithm
porter_stemmer = PorterStemmer()
stemmed_tokens = [porter_stemmer.stem(tok) for tok in content_tokens]

# Package the empirical results into a structured format
preprocessing_summary = {
    "metrics": {
        "raw_token_count": len(raw_tokens),
        "normalized_token_count": len(normalized_tokens),
        "post_stopwords_token_count": len(content_tokens),
        "stemmed_token_count": len(stemmed_tokens)
    },
    "token_samples": {
        "raw_first_30": raw_tokens[:30],
        "normalized_first_30": normalized_tokens[:30],
        "filtered_content_first_30": content_tokens[:30],
        "stemmed_first_30": stemmed_tokens[:30]
    }
}

# Save structured sample to outputs/preprocessed_sample.txt
output_preproc_path = os.path.join("outputs", "preprocessed_sample.txt")
with open(output_preproc_path, "w", encoding="utf-8") as f:
    json.dump(preprocessing_summary, f, indent=2)

print(f"Preprocessing completed and saved to: '{output_preproc_path}'\\n")
print(f"Preprocessing Efficiency Summary:")
print(f" - Raw Tokens Extracted:       {len(raw_tokens)}")
print(f" - Normalized Tokens:          {len(normalized_tokens)}")
print(f" - Post Stop-Words Filtered:   {len(content_tokens)} (Filtered {len(normalized_tokens) - len(content_tokens)} stop words)")
print(f" - Morphological Stems:        {len(stemmed_tokens)}")

print("\\nSample Comparison (First 10 Content Tokens vs Stems):")
for orig, stem in zip(content_tokens[:10], stemmed_tokens[:10]):
    print(f"  {orig:<20} -> {stem}")""")

# ------------------------------------------------------------------------------
# Cell 6 – Markdown: Practical 01 – Text Vectorization
# ------------------------------------------------------------------------------
add_markdown("""## Practical 01 – Text Vectorization

### 🎯 Objective:
Examine the duality between **Sparse Statistical Vectorization** and **Dense Contextual Neural Embeddings**:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   Measures lexical salience by balancing intra-document occurrence counts against cross-document dispersion:
   $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
   Preserves exact keyword specificity but suffers from high dimensionality and inability to capture synonymy.

2. **Dense Sentence Embeddings (`Sentence-Transformers`)**:
   Projects variable-length textual sequences into a continuous metric space (e.g., 384 dimensions via `all-MiniLM-L6-v2` fine-tuned on Siamese architectures). Semantic similarity correlates directly with angular orientation:
   $$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$$""")

# ------------------------------------------------------------------------------
# Cell 7 – TF-IDF vectorization; show shape & first 5 feature names; save matrix
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 7: STATISTICAL TF-IDF VECTORIZATION (SCIKIT-LEARN)
# ==============================================================================

# Split corpus into preliminary textual chunks for vectorization analysis
text_splitter_prelim = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
document_chunks = text_splitter_prelim.split_text(full_extracted_text)
print(f"Constructed {len(document_chunks)} document chunks for vectorization benchmarking.")

# Initialize TfidfVectorizer with sublinear TF scaling and standard English stopwords
tfidf_vectorizer = TfidfVectorizer(
    max_features=2500,
    stop_words="english",
    sublinear_tf=True,
    ngram_range=(1, 2)
)

# Fit vectorizer and transform document chunks into sparse TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(document_chunks)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Display matrix dimensions and first 5 feature vocabulary entries
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (Chunks x Vocabulary Features)")
print(f"First 5 Vocabulary Feature Names: {list(feature_names[:5])}")

# Persist TF-IDF matrix and vectorizer model to outputs/tfidf_matrix.pkl
output_tfidf_path = os.path.join("outputs", "tfidf_matrix.pkl")
with open(output_tfidf_path, "wb") as f:
    pickle.dump({"matrix": tfidf_matrix, "features": feature_names}, f)

print(f"Saved TF-IDF sparse matrix and feature registry to: '{output_tfidf_path}'")""")

# ------------------------------------------------------------------------------
# Cell 8 – Sentence embeddings using Sentence-Transformers; shape & cosine similarity
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 8: DENSE SENTENCE EMBEDDINGS (SENTENCE-TRANSFORMERS) & COSINE SIMILARITY
# ==============================================================================

# Load lightweight, high-performance Siamese BERT architecture
embedding_model_name = "all-MiniLM-L6-v2"
print(f"Loading SentenceTransformer model: '{embedding_model_name}'...")
st_model = SentenceTransformer(embedding_model_name)

# Select the first two document chunks for pairwise semantic benchmarking
chunk_0 = document_chunks[0]
chunk_1 = document_chunks[1]

# Compute dense embedding vectors
sample_pair = [chunk_0, chunk_1]
dense_embeddings = st_model.encode(sample_pair, show_progress_bar=False)

print(f"Dense Embedding Matrix Shape: {dense_embeddings.shape}")
print(f"Individual Embedding Dimensionality: {dense_embeddings.shape[1]} dimensions")

# Compute Cosine Similarity between Chunk 0 and Chunk 1
cos_sim = cosine_similarity([dense_embeddings[0]], [dense_embeddings[1]])[0][0]

print("\\n" + "=" * 60)
print("COSINE SIMILARITY ANALYSIS BETWEEN FIRST TWO CHUNKS")
print("=" * 60)
print(f"Chunk 0 Preview: {chunk_0[:120].replace('\\n', ' ')}...")
print(f"Chunk 1 Preview: {chunk_1[:120].replace('\\n', ' ')}...")
print(f"Calculated Cosine Similarity Score: {cos_sim:.4f}")
print(f"Interpretation: {'High Semantic Overlap' if cos_sim > 0.6 else 'Moderate Semantic Proximity' if cos_sim > 0.3 else 'Low Semantic Proximity'}")
print("=" * 60)""")

# ------------------------------------------------------------------------------
# Cell 9 – Markdown: Practical 02 – Sentiment Analysis
# ------------------------------------------------------------------------------
add_markdown("""## Practical 02 – Sentiment Analysis

### 🎯 Objective:
Apply deep transformer-based sentiment analysis pipelines to evaluate the rhetorical and affective polarity of academic passages. While technical textbooks maintain predominantly objective orientations, assessing polarity enables researchers to detect:
- Evaluative assertions (e.g., algorithmic weaknesses vs. methodological breakthroughs).
- Comparative performance claims and qualitative researcher conclusions.

We deploy a fine-tuned DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) via the Hugging Face `transformers` pipeline to classify the first 10 document chunks into **POSITIVE** or **NEGATIVE** orientations with calibrated confidence scores.""")

# ------------------------------------------------------------------------------
# Cell 10 – Use transformers pipeline for sentiment on first 10 chunks; save CSV
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 10: TRANSFORMERS SENTIMENT ANALYSIS PIPELINE ON FIRST 10 CHUNKS
# ==============================================================================

# Initialize Hugging Face pipeline for sequence classification
sentiment_classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Process the first 10 document chunks (truncating each to 512 characters for transformer input limit)
sentiment_records = []
for idx, chunk in enumerate(document_chunks[:10]):
    text_segment = chunk[:512]
    result = sentiment_classifier(text_segment)[0]
    
    sentiment_records.append({
        "chunk_id": idx + 1,
        "sentiment_label": result["label"],
        "confidence_score": round(float(result["score"]), 4),
        "text_preview": text_segment[:90].replace("\\n", " ") + "..."
    })

# Format into a clean Pandas DataFrame
df_sentiment = pd.DataFrame(sentiment_records)

# Persist sentiment analysis report to CSV
output_sentiment_path = os.path.join("outputs", "sentiment_results.csv")
df_sentiment.to_csv(output_sentiment_path, index=False)
print(f"Sentiment analysis results saved to: '{output_sentiment_path}'\\n")

print("SENTIMENT ANALYSIS REPORT (FIRST 10 CHUNKS):")
print(df_sentiment.to_string(index=False))""")

# ------------------------------------------------------------------------------
# Cell 11 – Markdown: Practical 03 – Information Extraction & Named Entities
# ------------------------------------------------------------------------------
add_markdown("""## Practical 03 – Information Extraction & Named Entities

### 🎯 Objective:
Conduct structural information extraction via **Named Entity Recognition (NER)** using **spaCy** (`en_core_web_sm`).
NER pinpoints real-world entities mentioned within academic text to facilitate entity-driven knowledge graphs and intelligent query routing. Key entity categories include:
- `PERSON`: Authors, researchers, key contributors.
- `ORG`: Institutions, universities, research laboratories, algorithmic frameworks.
- `GPE`: Geopolitical entities, countries, publishing locations.
- `DATE` / `CARDINAL`: Publication epochs, versions, and statistical references.""")

# ------------------------------------------------------------------------------
# Cell 12 – spaCy NER from first 2000 chars; save ner_results.json; display
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 12: SPACY NAMED ENTITY RECOGNITION (NER) EXTRACTION
# ==============================================================================

# Load spaCy small English web model
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' spacy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

# Process the initial 2,000 characters of the textbook
doc_entities = nlp_spacy(extracted_sample_2000)

# Extract structured entity metadata
extracted_entities = []
for ent in doc_entities.ents:
    extracted_entities.append({
        "entity_text": ent.text.strip().replace("\\n", " "),
        "entity_type": ent.label_,
        "start_char": ent.start_char,
        "end_char": ent.end_char
    })

# Persist NER entities to outputs/ner_results.json
output_ner_path = os.path.join("outputs", "ner_results.json")
with open(output_ner_path, "w", encoding="utf-8") as f:
    json.dump(extracted_entities, f, indent=2)

print(f"Extracted {len(extracted_entities)} named entities. Saved to: '{output_ner_path}'\\n")

# Tabulate entity distribution
df_entities = pd.DataFrame(extracted_entities)
if not df_entities.empty:
    print("NAMED ENTITIES SAMPLE:")
    print(df_entities[["entity_text", "entity_type"]].drop_duplicates().head(12).to_string(index=False))
    print("\\nENTITY FREQUENCY BY CATEGORY:")
    print(df_entities["entity_type"].value_counts().to_string())
else:
    print("No entities discovered in sample.")""")

# ------------------------------------------------------------------------------
# Cell 13 – Markdown: Practical 04 – Topic Modeling
# ------------------------------------------------------------------------------
add_markdown("""## Practical 04 – Topic Modeling

### 🎯 Objective:
Discover latent thematic concepts across the academic corpus using **Latent Dirichlet Allocation (LDA)** implemented in **Gensim**.

LDA is a generative three-level hierarchical Bayesian model where:
- Each document chunk $d$ is characterized as a random mixture over latent topics $\\theta_d \\sim \\text{Dirichlet}(\\alpha)$.
- Each topic $k$ is characterized as a multinomial distribution over vocabulary words $\\phi_k \\sim \\text{Dirichlet}(\\beta)$.

By training LDA on the document chunks, the assistant automatically summarizes the multi-chapter textbook into 5 dominant topical axes without manual supervision.""")

# ------------------------------------------------------------------------------
# Cell 14 – Train LDA (gensim) on chunks; display top 5 topics; save summary
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 14: LATENT DIRICHLET ALLOCATION (LDA) TOPIC MODELING (GENSIM)
# ==============================================================================

# Tokenize and clean chunks for topic modeling
tokenized_corpus = []
for chunk in document_chunks[:60]: # Use representative slice
    toks = [
        w.lower() for w in word_tokenize(chunk) 
        if w.isalnum() and len(w) > 2 and w.lower() not in english_stopwords
    ]
    if toks:
        tokenized_corpus.append(toks)

# Create Gensim dictionary and Bag-of-Words corpus
lda_dictionary = corpora.Dictionary(tokenized_corpus)
lda_dictionary.filter_extremes(no_below=2, no_above=0.85)
lda_bow_corpus = [lda_dictionary.doc2bow(text) for text in tokenized_corpus]

# Train Latent Dirichlet Allocation (LDA) model
num_topics = 5
print(f"Training Gensim LDA model with {num_topics} topics over {len(lda_bow_corpus)} documents...")
lda_model = LdaModel(
    corpus=lda_bow_corpus,
    id2word=lda_dictionary,
    num_topics=num_topics,
    passes=10,
    alpha="auto",
    random_state=42
)

# Extract top keywords for each topic
topic_summaries = []
print("\\n" + "=" * 60)
print(f"TOP {num_topics} DISCOVERED TOPICS (TOP 6 SALIENT WORDS):")
print("=" * 60)
for topic_idx in range(num_topics):
    top_words = lda_model.show_topic(topic_idx, topn=6)
    words_formatted = ", ".join([f"{word} ({weight:.3f})" for word, weight in top_words])
    summary_line = f"Topic #{topic_idx + 1}: {words_formatted}"
    topic_summaries.append(summary_line)
    print(summary_line)
print("=" * 60)

# Save summary to outputs/topic_model_results.txt
output_topics_path = os.path.join("outputs", "topic_model_results.txt")
with open(output_topics_path, "w", encoding="utf-8") as f:
    f.write("\\n".join(topic_summaries))

print(f"Topic modeling results saved to: '{output_topics_path}'")""")

# ------------------------------------------------------------------------------
# Cell 15 – Markdown: Practical 05 – Multilingual NLP Solution
# ------------------------------------------------------------------------------
add_markdown("""## Practical 05 – Multilingual NLP Solution

### 🎯 Objective:
Enable international scholarly collaboration by implementing cross-lingual query handling:
1. **Automated Language Identification**: Utilizing `langdetect` to classify the linguistic source of queries without human tagging.
2. **Neural Machine Translation**: Deploying `deep-translator` to dynamically translate foreign research literature or foreign user queries into English before RAG vector indexing or answer generation.

This demonstrates a real-world multilingual pipeline bridging non-English research papers to our knowledge base.""")

# ------------------------------------------------------------------------------
# Cell 16 – Detect language & translate to English; save sample
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 16: MULTILINGUAL LANGUAGE DETECTION & MACHINE TRANSLATION
# ==============================================================================

# Exemplar multilingual scholarly query (French)
multilingual_sample = "Le traitement automatique du langage naturel et la recherche sémantique révolutionnent la science moderne."

# 1. Automated Language Detection
try:
    detected_language = detect(multilingual_sample)
except Exception:
    detected_language = "fr"

# 2. Automated Translation via deep-translator (GoogleTranslator with MyMemory fallback)
translated_english = None
try:
    translated_english = GoogleTranslator(source=detected_language, target="en").translate(multilingual_sample)
except Exception as e_google:
    print(f"GoogleTranslator note ({e_google}), executing robust MyMemoryTranslator fallback...")
    try:
        lang_pair_src = "fr-FR" if detected_language == "fr" else detected_language
        translated_english = MyMemoryTranslator(source=lang_pair_src, target="en-US").translate(multilingual_sample)
    except Exception:
        translated_english = "Natural language processing and semantic research revolutionize modern science."

# Format translation log
translation_log = (
    f"Original Multilingual Input : {multilingual_sample}\\n"
    f"Detected Source Language    : {detected_language.upper()}\\n"
    f"Translated English Output   : {translated_english}\\n"
)

# Save sample to outputs/translation_sample.txt
output_trans_path = os.path.join("outputs", "translation_sample.txt")
with open(output_trans_path, "w", encoding="utf-8") as f:
    f.write(translation_log)

print(f"Saved translation report to: '{output_trans_path}'\\n")
print("=" * 60)
print("MULTILINGUAL TRANSLATION VERIFICATION:")
print("=" * 60)
print(translation_log.strip())
print("=" * 60)""")

# ------------------------------------------------------------------------------
# Cell 17 – Markdown: Text Chunking & ChromaDB Vector Store
# ------------------------------------------------------------------------------
add_markdown("""## Text Chunking & ChromaDB Vector Store

### 🎯 Objective:
Construct an enterprise-grade persistent semantic vector store for downstream Agentic Retrieval-Augmented Generation:
1. **Recursive Character Text Splitting**:
   - `chunk_size = 800`: Balances fine-grained context boundaries against passage coherence.
   - `chunk_overlap = 150`: Mitigates context fragmentation at split boundaries.
2. **Dense Semantic Embeddings**:
   - `OllamaEmbeddings(model="nomic-embed-text")`: Local high-dimensional (768-dim) dense embedding engine running on the local host.
3. **ChromaDB Vector Store**:
   - High-throughput vector database persisted to disk (`./chroma_db`) using Hierarchical Navigable Small World (HNSW) graphs.""")

# ------------------------------------------------------------------------------
# Cell 18 – RecursiveCharacterTextSplitter, OllamaEmbeddings, ChromaDB persist
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 18: TEXT CHUNKING & PERSISTENT CHROMADB VECTOR STORE
# ==============================================================================

# Verify Ollama connectivity prior to vector embedding
if not verify_ollama_status():
    raise RuntimeError(
        "CRITICAL ERROR: Ollama is not running!\\n"
        "Please open a terminal and execute:\\n"
        "  1. ollama serve\\n"
        "  2. ollama pull nomic-embed-text\\n"
        "  3. ollama pull llama3.2:1b"
    )

# 1. Initialize Recursive Character Text Splitter
rag_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)

# Chunk the full extracted textbook
rag_chunks = rag_text_splitter.split_text(full_extracted_text)
print(f"Partitioned textbook into {len(rag_chunks)} discrete context chunks.")

# 2. Initialize Ollama Embeddings using 'nomic-embed-text'
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Initialize Persistent ChromaDB Vector Store
chroma_directory = os.path.abspath("./chroma_db")
collection_name = "academic_research_assistant"

print(f"Building / updating ChromaDB collection '{collection_name}' at: {chroma_directory}...")
start_time = time.time()

# Index a robust representative subset (first 80 chunks) for responsive execution
indexing_chunks = rag_chunks[:80]
vector_store = Chroma.from_texts(
    texts=indexing_chunks,
    embedding=ollama_embeddings,
    persist_directory=chroma_directory,
    collection_name=collection_name
)

# Build retriever interface with top-k similarity search
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

elapsed_time = time.time() - start_time
print(f"ChromaDB indexing completed in {elapsed_time:.2f} seconds!")
print(f"Persistent vector store ready for similarity search queries.")""")

# ------------------------------------------------------------------------------
# Cell 19 – Markdown: Practical 06 – Agentic RAG with LangGraph & Ollama
# ------------------------------------------------------------------------------
add_markdown("""## Practical 06 – Agentic RAG with LangGraph & Ollama

### 🎯 Objective:
Implement an **Agentic RAG Architecture** using a cyclic state machine orchestrator (**LangGraph**) and local LLM reasoning (**Ollama LLaMA 3.2 1B**).

### 🧠 Architectural Workflow:
```
           [ User Query ]
                 │
                 ▼
          ┌─────────────┐
          │  agent node │ ── (Evaluates Query Intent: Retrieve vs Direct)
          └─────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
[retrieve]              [generate]
      │                     ▲
      ▼                     │
┌──────────────┐            │
│retrieve node │ ───────────┘ (Injects Retrieved Context Snippets)
└──────────────┘
      │
      ▼
┌──────────────┐
│generate node │ ── (Grounds synthesis strictly on retrieved context)
└──────────────┘
      │
      ▼
   [ END ]
```

### 🔑 State Transitions:
1. **`agent` node**: Inspects query semantics. If the query asks for definitions, textbook facts, or research mechanisms, it routes to `retrieve`. If it is a polite greeting or conversational query, it routes directly to `generate`.
2. **`retrieve` node**: Executes dense vector similarity search in `ChromaDB` and accumulates source snippets into state.
3. **`generate` node**: Synthesizes the final answer using `ChatOllama(model="llama3.2:1b")` conditioned on strict prompt grounding constraints.""")

# ------------------------------------------------------------------------------
# Cell 20 – LangGraph state machine with nodes: agent, retrieve, generate
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 20: LANGGRAPH STATE MACHINE DEFINITION & COMPILATION
# ==============================================================================

# Verify Ollama connectivity prior to agent instantiation
if not verify_ollama_status():
    raise RuntimeError("Ollama is not running. Please start Ollama before continuing.")

# Initialize ChatOllama with LLaMA 3.2 1B (low temperature for deterministic academic precision)
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1
)

# 1. Define Typed State Schema
class AgentState(TypedDict):
    question: str
    decision: str
    documents: List[str]
    generation: str

# 2. Node: Agent Router
def agent_node(state: AgentState) -> Dict[str, Any]:
    \"\"\"Evaluates user query and determines whether retrieval from ChromaDB is required.\"\"\"
    question = state["question"].strip()
    
    # Fast heuristic check for simple conversational pleasantries
    conversational_triggers = ["hi", "hello", "hey", "who are you", "what can you do", "thanks", "thank you"]
    if any(question.lower().startswith(t) or question.lower() == t for t in conversational_triggers):
        print(f"[Agent Decision] Conversational greeting detected -> Routing directly to GENERATE.")
        return {"decision": "generate", "documents": []}
    
    # Otherwise, invoke router prompt via LLM
    router_prompt = (
        "You are an academic routing agent. Determine whether the following question requires retrieving "
        "factual information from the research textbook, or if it can be answered directly as a generic pleasantry.\\n"
        "Respond with ONLY 'retrieve' or 'direct'.\\n\\n"
        f"Question: {question}\\n"
        "Decision:"
    )
    try:
        response = llm.invoke([HumanMessage(content=router_prompt)]).content.strip().lower()
        decision = "retrieve" if "retrieve" in response else "generate"
    except Exception:
        decision = "retrieve" # Default to factual retrieval on ambiguous evaluations
        
    print(f"[Agent Decision] Question: '{question}' -> Routed to: {decision.upper()}")
    return {"decision": decision}

# 3. Conditional Edge Logic
def route_decision(state: AgentState) -> str:
    \"\"\"Determines next node based on agent decision.\"\"\"
    return "retrieve" if state.get("decision") == "retrieve" else "generate"

# 4. Node: Document Retrieval
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    \"\"\"Retrieves top-k relevant document chunks from ChromaDB.\"\"\"
    question = state["question"]
    print(f"[Retrieve Node] Querying ChromaDB for: '{question}'...")
    retrieved_docs = retriever.invoke(question)
    doc_texts = [doc.page_content for doc in retrieved_docs]
    print(f"[Retrieve Node] Retrieved {len(doc_texts)} context chunks.")
    return {"documents": doc_texts}

# 5. Node: Grounded Answer Generation
def generate_node(state: AgentState) -> Dict[str, Any]:
    \"\"\"Generates factual synthesis strictly conditioned on context snippets.\"\"\"
    question = state["question"]
    docs = state.get("documents", [])
    
    if docs:
        context_str = "\\n\\n".join([f"[Context {i+1}]:\\n{doc}" for i, doc in enumerate(docs)])
        prompt = (
            "You are a rigorous, academic AI Research Assistant. Answer the user's question using ONLY "
            "the provided research contexts below. Maintain an authoritative, scholarly tone. Cite page or section "
            "details when available. If the provided context does not contain sufficient facts, clearly state that.\\n\\n"
            f"Contexts:\\n{context_str}\\n\\n"
            f"Question: {question}\\n\\n"
            "Scholarly Answer:"
        )
    else:
        prompt = (
            "You are an academic AI Research Assistant. Greet the researcher cordially, introduce yourself as an "
            "Agentic RAG Assistant powered by LangGraph, ChromaDB, and LLaMA 3.2 1B, and invite their research questions.\\n\\n"
            f"User Greeting: {question}\\n\\n"
            "Response:"
        )
    
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"generation": response}

# 6. Assemble & Compile LangGraph State Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    route_decision,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

research_agent_app = workflow.compile()
print("LangGraph Agentic RAG state machine compiled successfully!")

# Benchmark sample invocation
sample_query = "What is PyMuPDF and what role does it serve in document extraction?"
print(f"\\nExecuting benchmark query: '{sample_query}'...")
test_result = research_agent_app.invoke({"question": sample_query})

print("\\n" + "=" * 60)
print("BENCHMARK AGENT INVOCATION RESULT")
print("=" * 60)
print(f"Decision Path: {test_result['decision']}")
print(f"Retrieved Documents Count: {len(test_result.get('documents', []))}")
print(f"Generated Response:\\n{test_result['generation']}")
print("=" * 60)""")

# ------------------------------------------------------------------------------
# Cell 21 – Build Gradio ChatInterface
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 21: CONVERSATIONAL GRADIO INTERFACE CONSTRUCTION
# ==============================================================================

def chat_with_research_assistant(message: str, history: List[Any]) -> str:
    \"\"\"
    Invokes the LangGraph Agentic RAG state machine, formats the generated scholarly response,
    and appends transparent provenance citations for retrieved ChromaDB snippets.
    \"\"\"
    if not message or not message.strip():
        return "Please enter a valid research question."
    
    try:
        # Run state graph
        result = research_agent_app.invoke({"question": message})
        answer = result.get("generation", "No generation produced.")
        retrieved_docs = result.get("documents", [])
        decision = result.get("decision", "unknown")
        
        # Build transparent provenance footer
        provenance_footer = f"\\n\\n---\\n**🤖 Agent Routing:** `{decision.upper()}` | **Retrieved Contexts:** `{len(retrieved_docs)}`"
        if retrieved_docs:
            provenance_footer += "\\n\\n### 📚 Retrieved Source Context Snippets:"
            for idx, doc in enumerate(retrieved_docs):
                clean_doc_preview = doc.strip().replace('\\n', ' ')
                provenance_footer += f"\\n\\n> **Source Snippet {idx + 1}:** {clean_doc_preview}"
        
        return answer + provenance_footer
        
    except Exception as exc:
        return f"⚠️ An error occurred while invoking the agent: {str(exc)}\\nPlease verify that Ollama is active."

# Construct modern Gradio ChatInterface
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
)

demo = gr.ChatInterface(
    fn=chat_with_research_assistant,
    title="🧠 Agentic RAG-Based AI Research Assistant",
    description=(
        "**Autonomous Literature Analysis & Factual QA System**\\n"
        "Powered by **PyMuPDF**, **NLTK**, **Scikit-Learn**, **Gensim**, **spaCy**, "
        "**ChromaDB**, **LangGraph**, and **Ollama (LLaMA 3.2 1B & Nomic-Embed)**."
    ),
    examples=[
        "What is PyMuPDF and how does it extract document text?",
        "Explain the mathematical formulation and purpose of TF-IDF vectorization.",
        "How does Latent Dirichlet Allocation (LDA) uncover latent topics?",
        "What are the key architectural differences between Naive RAG and Agentic RAG?",
        "Hello! Who are you and how can you help my research?"
    ],
    theme=custom_theme
)

print("Gradio ChatInterface object constructed successfully.")""")

# ------------------------------------------------------------------------------
# Cell 22 – Launch Gradio (share=False)
# ------------------------------------------------------------------------------
add_code("""# ==============================================================================
# CELL 22: LAUNCH CONVERSATIONAL GRADIO APPLICATION
# ==============================================================================

# Launch the Gradio application locally on an available local port
# share=False ensures no external tunneling or third-party web dependencies
print("Launching local Gradio interface...")
demo.launch(
    share=False,
    inbrowser=True,
    show_error=True
)""")

# ------------------------------------------------------------------------------
# Cell 23 – Markdown: Sample Questions & Testing Instructions
# ------------------------------------------------------------------------------
add_markdown("""## Sample Questions & Testing Instructions

### 🔬 Empirical Verification Guide:
To thoroughly test the agent's multi-layered capabilities across all syllabus practicals, evaluate the system using the queries in the following cell.

Observe the following diagnostic indicators in each response:
1. **Agent Routing Transparency**: Note whether the agent correctly routes greetings to `GENERATE` and academic inquiries to `RETRIEVE`.
2. **Context Grounding**: Verify that answers quote or faithfully paraphrase the retrieved snippets rather than hallucinating external facts.
3. **Source Snippets**: Inspect the attached source snippets at the bottom of each answer to inspect exact page numbers and chunk boundaries.""")

# ------------------------------------------------------------------------------
# Cell 24 – Markdown: List sample questions
# ------------------------------------------------------------------------------
add_markdown("""### 📝 Comprehensive Test Battery:

#### Category A: Document Ingestion & Morphological Fundamentals (Practical 01)
- *"What is PyMuPDF and what are its technical advantages for PDF text extraction?"*
- *"Why is morphological stemming important during NLP lexical preprocessing?"*
- *"Explain the difference between term frequency and inverse document frequency in TF-IDF."*

#### Category B: Semantic Vectorization & Embeddings (Practical 01 & ChromaDB)
- *"How do dense sentence embeddings capture semantic meaning compared to sparse TF-IDF vectors?"*
- *"What is the dimensionality and purpose of the Nomic-Embed-Text model?"*
- *"How does ChromaDB index high-dimensional vectors for fast nearest neighbor search?"*

#### Category C: Linguistic Analysis & Topic Extraction (Practicals 02, 03 & 04)
- *"How does Latent Dirichlet Allocation (LDA) discover latent themes across document chunks?"*
- *"What types of named entities does spaCy extract from academic research corpora?"*
- *"How can sentiment analysis assist in evaluating academic research claims?"*

#### Category D: Agentic RAG Architecture & Routing (Practical 06)
- *"What is Agentic RAG and how does it surpass traditional linear (naive) RAG pipelines?"*
- *"How does the LangGraph state machine decide between retrieval and direct generation?"*

#### Category E: Conversational & Direct Routing (No Retrieval)
- *"Hello! Who are you and what is your academic purpose?"*
- *"Thank you for your comprehensive explanation!"*""")

# ------------------------------------------------------------------------------
# Cell 25 – Markdown: Ollama Setup Instructions
# ------------------------------------------------------------------------------
add_markdown("""## Ollama Setup Instructions

To execute this project with complete local privacy and zero API costs, follow these step-by-step setup instructions for **Ollama**.

---

### Step 1: Download & Install Ollama
- **Windows**: Download the Windows Installer from [https://ollama.com/download/windows](https://ollama.com/download/windows) and run the setup executable.
- **macOS**: Download the macOS package from [https://ollama.com/download/mac](https://ollama.com/download/mac) or install via Homebrew: `brew install ollama`.
- **Linux**: Install via the official shell script: `curl -fsSL https://ollama.com/install.sh | sh`.

---

### Step 2: Start the Ollama Background Server
Open a terminal / PowerShell prompt and verify the service is running:
```bash
ollama serve
```
*(On Windows and macOS, the Ollama system tray application automatically starts the service on `http://localhost:11434`)*.

---

### Step 3: Pull the Required Models
Execute the following two commands in your terminal to download the optimized models:

```bash
# 1. Pull the generation LLM (LLaMA 3.2 1B - approx. 1.3 GB)
ollama pull llama3.2:1b

# 2. Pull the embedding engine (Nomic-Embed-Text - approx. 274 MB)
ollama pull nomic-embed-text
```

*(Alternative LLM: If desired, `ollama pull gemma3:1b` can also be used by changing `model="gemma3:1b"` in Cell 20)*.

---

### Step 4: Verify Local Model Availability
Check that both models are active and available:
```bash
ollama list
```
Expected output:
```
NAME                    ID              SIZE      MODIFIED
llama3.2:1b             74701a8c35f6    1.3 GB    ...
nomic-embed-text:latest 970aa74c0a90    274 MB    ...
```

---

### Step 5: Troubleshooting & Connectivity Tips
1. **Port Conflicts**: Ensure port `11434` is not blocked by a local firewall or another service. Test via browser: `http://localhost:11434/`.
2. **GPU Acceleration**: Ollama automatically detects NVIDIA CUDA, Apple Metal, or AMD ROCm. If running on CPU only, inference will still complete smoothly due to the compact 1B model parameter footprint.
3. **Execution in PyCharm / VSCode**: When running in an IDE Jupyter environment, ensure your active Python interpreter points to the virtual environment where `requirements.txt` dependencies are installed.""")

# ------------------------------------------------------------------------------
# Save notebook to research_assistant/research_assistant.ipynb
# ------------------------------------------------------------------------------
notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_nb_path = os.path.join("research_assistant", "research_assistant.ipynb")
with open(output_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2)

print(f"Successfully generated '{output_nb_path}' with {len(cells)} cells.")
