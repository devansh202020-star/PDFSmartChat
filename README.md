# PDF Text Summarization and Q&A Chatbot

A comprehensive NLP application built with Streamlit that demonstrates advanced natural language processing techniques for PDF document analysis.

## 🎓 Academic Project Overview

This project showcases a complete NLP pipeline implementing multiple techniques covered in natural language processing coursework, including text preprocessing, vectorization, summarization, sentiment analysis, and topic modeling.

## ✨ Key Features

### 1. **PDF Text Extraction**
- Upload single or multiple PDF documents
- Robust text extraction using PyPDF2
- Document management with easy switching between files

### 2. **Text Preprocessing Pipeline**
- **Tokenization**: Sentence and word-level tokenization using NLTK
- **Stopword Removal**: Filtering common words using NLTK stopwords corpus
- **Lemmatization**: Converting words to base forms using WordNet Lemmatizer
- Visual demonstration of preprocessing effects

### 3. **Document Statistics**
- Character, word, and sentence counts
- Unique vocabulary analysis
- Word frequency visualization
- Preprocessing impact comparison

### 4. **Keyword Extraction**
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency analysis
- Top N keyword extraction with importance scores
- Visual representation of keyword weights

### 5. **Extractive Summarization**
- **Technique**: TF-IDF-based sentence ranking
- Selects most important sentences from original document
- Configurable summary length
- Download functionality for summaries

### 6. **Abstractive Summarization**
- **Technique**: Groq's Llama 3.3 70B language model
- Generates new summary by understanding and rephrasing content
- Configurable summary length
- Download functionality for summaries

### 7. **Q&A Chatbot**
- **Semantic Search**: TF-IDF + Cosine Similarity for context retrieval
- **Language Model**: Groq Llama for answer generation
- Context-aware responses based on document content
- Chat history with export functionality

### 8. **Sentiment Analysis**
- **Technique**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Overall document sentiment scoring
- Sentence-level sentiment breakdown
- Positive, negative, and neutral classification
- Sentiment distribution visualization

### 9. **Topic Modeling**
- **Technique**: Latent Dirichlet Allocation (LDA)
- Unsupervised topic discovery
- Configurable number of topics
- Top words per topic with weights
- Visual representation of topic distributions

### 10. **Comparative Document Analysis**
- Multi-document statistics comparison
- **TF-IDF Similarity Matrix**: Cosine similarity between documents
- Identification of most/least similar document pairs
- Side-by-side word frequency comparison

## 🔧 NLP Techniques Demonstrated

| Technique | Library/Method | Application |
|-----------|---------------|-------------|
| Tokenization | NLTK | Sentence and word segmentation |
| Stopword Removal | NLTK | Noise reduction |
| Lemmatization | NLTK WordNet | Word normalization |
| TF-IDF Vectorization | scikit-learn | Feature extraction, keyword extraction |
| Cosine Similarity | scikit-learn | Document similarity, semantic search |
| Extractive Summarization | TF-IDF sentence ranking | Document summarization |
| Abstractive Summarization | Groq Llama (Transformer) | AI-powered summarization |
| Sentiment Analysis | VADER | Emotion detection |
| Topic Modeling | LDA (scikit-learn) | Unsupervised topic discovery |
| Semantic Search | TF-IDF + Cosine Similarity | Question answering |

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- Groq API Key

### Dependencies
```bash
pip install streamlit groq pypdf2 nltk scikit-learn numpy pandas vaderSentiment
```

### Environment Setup
1. Set your Groq API key as an environment variable:
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```

2. NLTK data will be downloaded automatically on first run

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

## 📊 Usage Guide

### Basic Workflow
1. **Upload PDF(s)**: Use the sidebar to upload one or more PDF documents
2. **Extract Text**: Click "Extract Text from All PDFs" button
3. **Explore Features**: Navigate through different tabs to analyze your documents
4. **Switch Documents**: Use the dropdown to switch between uploaded documents
5. **Download Results**: Export summaries and chat history as needed

### Best Practices
- Use documents with substantial text content for better analysis
- Try multiple documents for comparative analysis features
- Experiment with different settings (summary length, number of topics, etc.)
- Use the Q&A chatbot to explore specific aspects of your documents

## 🎯 Academic Relevance

This project demonstrates understanding of:
- **Text Preprocessing**: Complete pipeline from raw text to clean tokens
- **Feature Extraction**: Multiple vectorization techniques (TF-IDF, Count Vectorization)
- **Supervised Learning**: Classification via sentiment analysis
- **Unsupervised Learning**: Topic modeling with LDA
- **Deep Learning**: Transformer-based text generation
- **Information Retrieval**: Semantic search and similarity metrics
- **NLP Applications**: Real-world use cases combining multiple techniques

## 🔬 Technical Architecture

### Frontend
- **Streamlit**: Interactive web interface
- **Pandas/NumPy**: Data manipulation and visualization
- **Matplotlib**: Charts and graphs

### NLP Backend
- **NLTK**: Text preprocessing and tokenization
- **scikit-learn**: Vectorization, LDA, cosine similarity
- **VADER**: Sentiment analysis
- **Groq API**: Large language model integration

### Document Processing
- **PyPDF2**: PDF text extraction
- **Session State**: Document management and caching

## 📈 Performance Considerations

- Caching for NLTK resources and Groq client
- Efficient vectorization with scikit-learn
- Defensive error handling for edge cases
- Graceful degradation for short/sparse documents

## 🔐 Security

- API keys stored as environment variables
- No secrets exposed in code
- Secure API communication with Groq

## 🎓 Educational Value

This project serves as a comprehensive demonstration of:
1. **End-to-end NLP pipeline** from raw PDFs to actionable insights
2. **Multiple NLP paradigms**: rule-based, statistical, and neural approaches
3. **Practical applications**: summarization, Q&A, sentiment analysis, topic modeling
4. **Software engineering**: modular design, error handling, user interface

## 📝 License

This is an academic project for educational purposes.

## 👥 Author

Devansh Gupta 
