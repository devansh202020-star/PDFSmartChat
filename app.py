import streamlit as st
import os
from groq import Groq
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Map each package to its correct loader path
    nltk_resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }
    
    for package, path in nltk_resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_data()

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in environment variables!")
        return None
    return Groq(api_key=api_key)

client = get_groq_client()

# Text Preprocessing Functions
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep periods for sentences
        text = re.sub(r'[^\w\s\.]', '', text)
        return text.strip()
    
    def tokenize_sentences(self, text):
        """Tokenize text into sentences"""
        return sent_tokenize(text)
    
    def tokenize_words(self, text):
        """Tokenize text into words"""
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words and token.isalnum()]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, remove_stops=True, lemmatize=True):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        # Tokenize
        tokens = self.tokenize_words(text)
        # Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        return tokens

# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Document Statistics
def get_document_statistics(text, preprocessor):
    """Calculate various document statistics"""
    sentences = preprocessor.tokenize_sentences(text)
    words = preprocessor.tokenize_words(text)
    
    # Apply full preprocessing pipeline (stopword removal + lemmatization)
    preprocessed_tokens = preprocessor.preprocess(text, remove_stops=True, lemmatize=True)
    
    stats = {
        "Total Characters": len(text),
        "Total Words": len(words),
        "Total Sentences": len(sentences),
        "Unique Words": len(set(words)),
        "Unique Words (after preprocessing)": len(set(preprocessed_tokens)),
        "Average Words per Sentence": round(len(words) / len(sentences), 2) if sentences else 0,
        "Average Word Length": round(sum(len(w) for w in words) / len(words), 2) if words else 0
    }
    
    return stats, preprocessed_tokens

# Keyword Extraction using TF-IDF
def extract_keywords(text, preprocessor, top_n=10):
    """Extract top keywords using TF-IDF"""
    if not text or not text.strip():
        return []
    
    sentences = preprocessor.tokenize_sentences(text)
    
    if len(sentences) < 1:
        return []
    
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        if len(feature_names) == 0:
            return []
        
        # Get average TF-IDF scores
        avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        keyword_scores = [(feature_names[i], avg_scores[i]) for i in range(len(feature_names))]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores
    except Exception as e:
        return []

# Extractive Summarization using TF-IDF
def extractive_summarization(text, preprocessor, num_sentences=5):
    """Generate extractive summary using TF-IDF"""
    if not text or not text.strip():
        return "Error: No text to summarize."
    
    sentences = preprocessor.tokenize_sentences(text)
    
    if len(sentences) == 0:
        return "Error: No sentences found in the text."
    
    if len(sentences) <= num_sentences:
        return text
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Fit and transform sentences
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Check if we got valid features
        if tfidf_matrix.shape[1] == 0:
            return "Error: Unable to extract meaningful features from text."
        
        # Calculate sentence scores (sum of TF-IDF values)
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Get top sentences
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentence_indices.sort()  # Maintain original order
        
        # Create summary
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        return summary
    except Exception as e:
        return f"Error generating extractive summary: {str(e)}"

# Abstractive Summarization using Groq
def abstractive_summarization(text, max_length=300):
    """Generate abstractive summary using Groq Llama model"""
    if not client:
        return "Groq client not initialized. Please check your API key."
    
    try:
        # Truncate text if too long
        max_input = 6000
        if len(text) > max_input:
            text = text[:max_input] + "..."
        
        prompt = f"""Please provide a comprehensive summary of the following text. 
        The summary should capture the main ideas, key points, and important details.
        Keep the summary concise but informative (around {max_length} words).
        
        Text:
        {text}
        
        Summary:"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating abstractive summary: {str(e)}"

# Q&A Chatbot with Semantic Search
def answer_question(question, document_text, preprocessor):
    """Answer questions about the document using semantic search and Groq"""
    if not client:
        return "Groq client not initialized. Please check your API key."
    
    if not document_text or not document_text.strip():
        return "Error: No document text available."
    
    if not question or not question.strip():
        return "Error: Please provide a question."
    
    try:
        # Split document into sentences for semantic search
        sentences = preprocessor.tokenize_sentences(document_text)
        
        if len(sentences) == 0:
            return "Error: No sentences found in the document."
        
        # Use TF-IDF to find most relevant sentences
        all_texts = sentences + [question]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Check if vectorization produced valid features
        if tfidf_matrix.shape[1] == 0:
            return "Error: Unable to process the document for question answering."
        
        # Calculate cosine similarity between question and sentences
        question_vector = tfidf_matrix[-1]
        sentence_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(question_vector, sentence_vectors)[0]
        
        # Get top 5 most relevant sentences
        num_context_sentences = min(5, len(sentences))
        top_indices = similarities.argsort()[-num_context_sentences:][::-1]
        relevant_context = ' '.join([sentences[i] for i in top_indices])
        
        # Generate answer using Groq with context
        prompt = f"""Based on the following context from a document, please answer the question.
        If the answer cannot be found in the context, say so.
        
        Context:
        {relevant_context}
        
        Question: {question}
        
        Answer:"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=512,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error answering question: {str(e)}"

# Sentiment Analysis
@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment(text, preprocessor):
    """Analyze sentiment of document sections"""
    if not text or not text.strip():
        return None
    
    analyzer = get_sentiment_analyzer()
    sentences = preprocessor.tokenize_sentences(text)
    
    if len(sentences) == 0:
        return None
    
    # Analyze each sentence
    sentence_sentiments = []
    for sentence in sentences:
        scores = analyzer.polarity_scores(sentence)
        sentence_sentiments.append({
            'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })
    
    # Overall document sentiment
    overall = analyzer.polarity_scores(text)
    
    # Categorize sentences
    positive_sentences = [s for s in sentence_sentiments if s['compound'] >= 0.05]
    negative_sentences = [s for s in sentence_sentiments if s['compound'] <= -0.05]
    neutral_sentences = [s for s in sentence_sentiments if -0.05 < s['compound'] < 0.05]
    
    return {
        'overall': overall,
        'sentence_sentiments': sentence_sentiments,
        'positive_sentences': positive_sentences,
        'negative_sentences': negative_sentences,
        'neutral_sentences': neutral_sentences
    }

# Topic Modeling using LDA
def perform_topic_modeling(text, preprocessor, num_topics=5, num_words=10):
    """Perform topic modeling using Latent Dirichlet Allocation"""
    if not text or not text.strip():
        return None
    
    sentences = preprocessor.tokenize_sentences(text)
    
    if len(sentences) < num_topics:
        return None
    
    try:
        # Create document-term matrix
        vectorizer = CountVectorizer(max_features=1000, stop_words='english', min_df=2)
        doc_term_matrix = vectorizer.fit_transform(sentences)
        
        if doc_term_matrix.shape[1] == 0:
            return None
        
        # Apply LDA
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
        lda_model.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_num': topic_idx + 1,
                'words': top_words,
                'weights': [topic[i] for i in top_words_idx]
            })
        
        return {
            'topics': topics,
            'num_topics': num_topics
        }
    except Exception as e:
        return None

# Document Comparison
def compare_documents(documents_dict, preprocessor):
    """Compare multiple documents"""
    if len(documents_dict) < 2:
        return None
    
    results = {}
    
    # Document names (always set)
    doc_names = list(documents_dict.keys())
    results['doc_names'] = doc_names
    
    # Calculate statistics for each document
    for doc_name, doc_text in documents_dict.items():
        stats, tokens = get_document_statistics(doc_text, preprocessor)
        results[doc_name] = {
            'stats': stats,
            'tokens': tokens,
            'text': doc_text
        }
    
    # TF-IDF similarity matrix
    doc_texts = [documents_dict[name] for name in doc_names]
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        
        # Check if we got valid features
        if tfidf_matrix.shape[1] > 0:
            similarity_matrix = cosine_similarity(tfidf_matrix)
            results['similarity_matrix'] = similarity_matrix
        else:
            results['similarity_matrix'] = None
    except Exception as e:
        results['similarity_matrix'] = None
    
    return results

# Streamlit App
def main():
    st.set_page_config(
        page_title="PDF Text Summarization & Q&A Chatbot",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Text Summarization and Q&A Chatbot")
    st.markdown("Upload a PDF document to extract text, generate summaries, and ask questions about the content.")
    
    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = {}  # {filename: text}
    if 'active_document' not in st.session_state:
        st.session_state.active_document = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = TextPreprocessor()
    
    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("📤 Upload PDF Documents")
        uploaded_files = st.file_uploader("Choose PDF file(s)", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Extract Text from All PDFs"):
                with st.spinner("Extracting text from PDFs..."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        text = extract_text_from_pdf(uploaded_file)
                        if text:
                            st.session_state.documents[uploaded_file.name] = text
                            success_count += 1
                    
                    if success_count > 0:
                        st.session_state.active_document = list(st.session_state.documents.keys())[0]
                        st.session_state.chat_history = []
                        st.success(f"✅ Extracted text from {success_count} document(s)!")
                    else:
                        st.error("Failed to extract text from PDFs")
        
        # Document selector
        if st.session_state.documents:
            st.markdown("---")
            st.header("📚 Loaded Documents")
            selected_doc = st.selectbox(
                "Select active document:",
                options=list(st.session_state.documents.keys()),
                index=list(st.session_state.documents.keys()).index(st.session_state.active_document) if st.session_state.active_document else 0
            )
            
            if selected_doc != st.session_state.active_document:
                st.session_state.active_document = selected_doc
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("🗑️ Clear All Documents"):
                st.session_state.documents = {}
                st.session_state.active_document = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Settings")
        num_summary_sentences = st.slider("Extractive Summary Sentences", 3, 10, 5)
        abstractive_length = st.slider("Abstractive Summary Length (words)", 100, 500, 250)
        top_keywords = st.slider("Number of Keywords to Extract", 5, 20, 10)
    
    # Main content area
    if st.session_state.documents and st.session_state.active_document:
        # Get current document text
        document_text = st.session_state.documents[st.session_state.active_document]
        
        # Display active document info
        st.info(f"📄 Active Document: **{st.session_state.active_document}** ({len(st.session_state.documents)} document(s) loaded)")
        
        # Create tabs for different features
        if len(st.session_state.documents) > 1:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Document Statistics", 
                "🔑 Keywords", 
                "📝 Extractive Summary", 
                "🤖 Abstractive Summary",
                "💬 Q&A Chatbot",
                "😊 Sentiment Analysis",
                "🎯 Topic Modeling",
                "🔄 Compare Documents"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Document Statistics", 
                "🔑 Keywords", 
                "📝 Extractive Summary", 
                "🤖 Abstractive Summary",
                "💬 Q&A Chatbot",
                "😊 Sentiment Analysis",
                "🎯 Topic Modeling"
            ])
            tab8 = None
        
        preprocessor = st.session_state.preprocessor
        
        # Tab 1: Document Statistics
        with tab1:
            st.header("Document Statistics")
            st.markdown("**NLP Techniques:** Tokenization, Stopword Removal, Lemmatization")
            
            stats, preprocessed_tokens = get_document_statistics(
                document_text, 
                preprocessor
            )
            
            # Display statistics in columns
            col1, col2, col3 = st.columns(3)
            stats_items = list(stats.items())
            
            with col1:
                for key, value in stats_items[:3]:
                    st.metric(key, value)
            with col2:
                for key, value in stats_items[3:5]:
                    st.metric(key, value)
            with col3:
                for key, value in stats_items[5:]:
                    st.metric(key, value)
            
            # Word frequency distribution
            st.subheader("Top 15 Most Common Words (after stopword removal + lemmatization)")
            word_freq = Counter(preprocessed_tokens).most_common(15)
            
            if word_freq:
                df_freq = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                st.bar_chart(df_freq.set_index('Word'))
            
            # Show preprocessed text sample
            with st.expander("View Preprocessed Text Sample (first 100 lemmatized tokens)"):
                sample_tokens = preprocessed_tokens[:100]
                st.write(" ".join(sample_tokens))
                st.info("💡 Preprocessing pipeline: Tokenization → Stopword Removal → Lemmatization")
        
        # Tab 2: Keywords
        with tab2:
            st.header("Keyword Extraction")
            st.markdown("**NLP Technique:** TF-IDF (Term Frequency-Inverse Document Frequency)")
            
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(
                    document_text, 
                    preprocessor, 
                    top_n=top_keywords
                )
            
            if keywords:
                st.subheader(f"Top {len(keywords)} Keywords")
                
                # Create DataFrame for better visualization
                df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'TF-IDF Score'])
                df_keywords['TF-IDF Score'] = df_keywords['TF-IDF Score'].round(4)
                
                # Display as table
                st.dataframe(df_keywords, use_container_width=True)
                
                # Display as bar chart
                st.bar_chart(df_keywords.set_index('Keyword'))
            else:
                st.info("Not enough content to extract keywords.")
        
        # Tab 3: Extractive Summary
        with tab3:
            st.header("Extractive Summarization")
            st.markdown("**NLP Technique:** TF-IDF vectorization with sentence ranking")
            st.markdown("This method selects the most important sentences from the original document based on TF-IDF scores.")
            
            if 'extractive_summary' not in st.session_state:
                st.session_state.extractive_summary = None
            
            if st.button("Generate Extractive Summary", key="extractive"):
                with st.spinner("Generating extractive summary..."):
                    summary = extractive_summarization(
                        document_text,
                        preprocessor,
                        num_sentences=num_summary_sentences
                    )
                    st.session_state.extractive_summary = summary
                    st.rerun()
            
            if st.session_state.extractive_summary:
                st.subheader("Summary")
                st.write(st.session_state.extractive_summary)
                
                # Show statistics
                st.info(f"Original: {len(preprocessor.tokenize_sentences(document_text))} sentences → Summary: {len(preprocessor.tokenize_sentences(st.session_state.extractive_summary))} sentences")
                
                # Download button
                st.download_button(
                    label="📥 Download Extractive Summary",
                    data=st.session_state.extractive_summary,
                    file_name="extractive_summary.txt",
                    mime="text/plain"
                )
        
        # Tab 4: Abstractive Summary
        with tab4:
            st.header("Abstractive Summarization")
            st.markdown("**NLP Technique:** Transformer-based language model (Groq Llama)")
            st.markdown("This method generates a new summary using AI, potentially rephrasing and combining information.")
            
            if 'abstractive_summary' not in st.session_state:
                st.session_state.abstractive_summary = None
            
            if st.button("Generate Abstractive Summary", key="abstractive"):
                with st.spinner("Generating abstractive summary using Groq Llama..."):
                    summary = abstractive_summarization(
                        document_text,
                        max_length=abstractive_length
                    )
                    st.session_state.abstractive_summary = summary
                    st.rerun()
            
            if st.session_state.abstractive_summary:
                st.subheader("AI-Generated Summary")
                st.write(st.session_state.abstractive_summary)
                
                # Download button
                st.download_button(
                    label="📥 Download Abstractive Summary",
                    data=st.session_state.abstractive_summary,
                    file_name="abstractive_summary.txt",
                    mime="text/plain"
                )
        
        # Tab 5: Q&A Chatbot
        with tab5:
            st.header("Q&A Chatbot")
            st.markdown("**NLP Techniques:** Semantic search (TF-IDF + Cosine Similarity) + Language Model")
            st.markdown("Ask questions about your document. The system finds relevant context and generates answers using AI.")
            
            # Display chat history
            for i, (q, a) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**Q{i+1}:** {q}")
                    st.markdown(f"**A{i+1}:** {a}")
                    st.markdown("---")
            
            # Question input
            question = st.text_input("Ask a question about the document:", key="question_input")
            
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1:
                ask_button = st.button("Ask", type="primary")
            with col2:
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col3:
                if st.session_state.chat_history:
                    # Create chat history export text
                    chat_export = "Q&A Chat History\n" + "="*50 + "\n\n"
                    for i, (q, a) in enumerate(st.session_state.chat_history):
                        chat_export += f"Q{i+1}: {q}\n\n"
                        chat_export += f"A{i+1}: {a}\n\n"
                        chat_export += "-"*50 + "\n\n"
                    
                    st.download_button(
                        label="📥 Export Chat History",
                        data=chat_export,
                        file_name="chat_history.txt",
                        mime="text/plain"
                    )
            
            if ask_button and question:
                with st.spinner("Searching document and generating answer..."):
                    answer = answer_question(
                        question,
                        document_text,
                        preprocessor
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
        
        # Tab 6: Sentiment Analysis
        with tab6:
            st.header("Sentiment Analysis")
            st.markdown("**NLP Technique:** VADER Sentiment Analysis")
            st.markdown("Analyze the emotional tone of the document using lexicon-based sentiment analysis.")
            
            if st.button("Analyze Sentiment", key="sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment_results = analyze_sentiment(document_text, preprocessor)
                    
                    if sentiment_results:
                        # Overall sentiment
                        st.subheader("Overall Document Sentiment")
                        overall = sentiment_results['overall']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Compound Score", f"{overall['compound']:.3f}")
                        with col2:
                            st.metric("Positive", f"{overall['pos']:.2%}")
                        with col3:
                            st.metric("Negative", f"{overall['neg']:.2%}")
                        with col4:
                            st.metric("Neutral", f"{overall['neu']:.2%}")
                        
                        # Sentiment interpretation
                        if overall['compound'] >= 0.05:
                            st.success("📈 Overall sentiment: **Positive**")
                        elif overall['compound'] <= -0.05:
                            st.error("📉 Overall sentiment: **Negative**")
                        else:
                            st.info("➡️ Overall sentiment: **Neutral**")
                        
                        st.markdown("---")
                        
                        # Sentence breakdown
                        st.subheader("Sentiment Distribution")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Positive Sentences", len(sentiment_results['positive_sentences']))
                        with col2:
                            st.metric("Negative Sentences", len(sentiment_results['negative_sentences']))
                        with col3:
                            st.metric("Neutral Sentences", len(sentiment_results['neutral_sentences']))
                        
                        # Show most positive and negative sentences
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Most Positive Sentences")
                            positive_sorted = sorted(sentiment_results['positive_sentences'], 
                                                   key=lambda x: x['compound'], reverse=True)[:5]
                            for sent in positive_sorted:
                                st.success(f"**{sent['compound']:.2f}** - {sent['sentence']}")
                        
                        with col2:
                            st.subheader("Most Negative Sentences")
                            negative_sorted = sorted(sentiment_results['negative_sentences'], 
                                                   key=lambda x: x['compound'])[:5]
                            for sent in negative_sorted:
                                st.error(f"**{sent['compound']:.2f}** - {sent['sentence']}")
                        
                        # Sentiment score chart
                        st.subheader("Sentiment Score Distribution")
                        sentiment_df = pd.DataFrame(sentiment_results['sentence_sentiments'])
                        st.line_chart(sentiment_df['compound'])
                    else:
                        st.warning("Unable to analyze sentiment for this document.")
        
        # Tab 7: Topic Modeling
        with tab7:
            st.header("Topic Modeling")
            st.markdown("**NLP Technique:** Latent Dirichlet Allocation (LDA)")
            st.markdown("Discover latent topics in the document using unsupervised machine learning.")
            
            num_topics = st.slider("Number of Topics", 3, 10, 5)
            
            if st.button("Perform Topic Modeling", key="topics"):
                with st.spinner("Performing topic modeling..."):
                    topic_results = perform_topic_modeling(document_text, preprocessor, num_topics=num_topics)
                    
                    if topic_results:
                        st.subheader(f"Discovered {topic_results['num_topics']} Topics")
                        
                        for topic in topic_results['topics']:
                            with st.expander(f"📌 Topic {topic['topic_num']}", expanded=True):
                                st.markdown("**Top Words:**")
                                
                                # Create a DataFrame for visualization
                                topic_df = pd.DataFrame({
                                    'Word': topic['words'],
                                    'Weight': topic['weights']
                                })
                                
                                # Display as bar chart
                                st.bar_chart(topic_df.set_index('Word'))
                                
                                # Also show as text
                                st.write(", ".join(topic['words'][:10]))
                    else:
                        st.warning("Unable to perform topic modeling. The document may be too short or lacks sufficient variety.")
        
        # Tab 8: Comparative Analysis (only if multiple documents)
        if tab8 is not None:
            with tab8:
                st.header("Comparative Document Analysis")
                st.markdown("**NLP Techniques:** TF-IDF vectorization + Cosine Similarity for document comparison")
                st.markdown("Compare multiple documents to identify similarities and differences.")
                
                if st.button("Compare All Documents", key="compare"):
                    with st.spinner("Comparing documents..."):
                        comparison_results = compare_documents(st.session_state.documents, preprocessor)
                        
                        if comparison_results:
                            # Document statistics comparison
                            st.subheader("Document Statistics Comparison")
                            
                            stats_data = []
                            for doc_name in comparison_results['doc_names']:
                                stats = comparison_results[doc_name]['stats']
                                stats_data.append({
                                    'Document': doc_name,
                                    **stats
                                })
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Similarity matrix
                            if comparison_results.get('similarity_matrix') is not None:
                                st.markdown("---")
                                st.subheader("Document Similarity Matrix")
                                st.markdown("Values range from 0 (completely different) to 1 (identical)")
                                
                                similarity_df = pd.DataFrame(
                                    comparison_results['similarity_matrix'],
                                    index=comparison_results['doc_names'],
                                    columns=comparison_results['doc_names']
                                )
                                
                                # Display heatmap-style
                                st.dataframe(similarity_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1), 
                                           width='stretch')
                                
                                # Find most and least similar pairs
                                st.markdown("---")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("Most Similar Documents")
                                    max_sim = 0
                                    max_pair = (None, None)
                                    for i in range(len(comparison_results['doc_names'])):
                                        for j in range(i+1, len(comparison_results['doc_names'])):
                                            sim = comparison_results['similarity_matrix'][i][j]
                                            if sim > max_sim:
                                                max_sim = sim
                                                max_pair = (comparison_results['doc_names'][i], 
                                                          comparison_results['doc_names'][j])
                                    
                                    if max_pair[0]:
                                        st.success(f"**{max_pair[0]}** ↔ **{max_pair[1]}**")
                                        st.metric("Similarity Score", f"{max_sim:.2%}")
                                
                                with col2:
                                    st.subheader("Least Similar Documents")
                                    min_sim = 1
                                    min_pair = (None, None)
                                    for i in range(len(comparison_results['doc_names'])):
                                        for j in range(i+1, len(comparison_results['doc_names'])):
                                            sim = comparison_results['similarity_matrix'][i][j]
                                            if sim < min_sim:
                                                min_sim = sim
                                                min_pair = (comparison_results['doc_names'][i], 
                                                          comparison_results['doc_names'][j])
                                    
                                    if min_pair[0]:
                                        st.info(f"**{min_pair[0]}** ↔ **{min_pair[1]}**")
                                        st.metric("Similarity Score", f"{min_sim:.2%}")
                            else:
                                st.markdown("---")
                                st.warning("⚠️ Unable to compute similarity matrix. Documents may be too short or contain insufficient unique vocabulary for TF-IDF analysis.")
                            
                            # Word frequency comparison
                            st.markdown("---")
                            st.subheader("Top Words Comparison")
                            
                            cols = st.columns(len(comparison_results['doc_names']))
                            for idx, doc_name in enumerate(comparison_results['doc_names']):
                                with cols[idx]:
                                    st.markdown(f"**{doc_name}**")
                                    tokens = comparison_results[doc_name]['tokens']
                                    word_freq = Counter(tokens).most_common(10)
                                    if word_freq:
                                        freq_df = pd.DataFrame(word_freq, columns=['Word', 'Count'])
                                        st.dataframe(freq_df, use_container_width=True)
                        else:
                            st.warning("Need at least 2 documents to perform comparison.")
    
    else:
        # Welcome message
        st.info("👈 Please upload a PDF document using the sidebar to get started!")
        
        st.markdown("## Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Document Analysis")
            st.markdown("- Comprehensive text statistics")
            st.markdown("- Word frequency analysis")
            st.markdown("- Text preprocessing demonstration")
            
            st.markdown("### 📝 Summarization")
            st.markdown("- **Extractive:** TF-IDF based sentence selection")
            st.markdown("- **Abstractive:** AI-powered summary generation")
        
        with col2:
            st.markdown("### 🔑 Keyword Extraction")
            st.markdown("- TF-IDF vectorization")
            st.markdown("- Most important terms identification")
            
            st.markdown("### 💬 Q&A Chatbot")
            st.markdown("- Semantic search with cosine similarity")
            st.markdown("- Context-aware AI responses")
            st.markdown("- Interactive question answering")
        
        st.markdown("---")
        st.markdown("### 🎓 NLP Concepts Demonstrated")
        st.markdown("""
        - **Text Preprocessing:** Tokenization, stopword removal, lemmatization
        - **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Similarity Metrics:** Cosine similarity for semantic search
        - **Extractive Summarization:** Sentence ranking and selection
        - **Abstractive Summarization:** Transformer-based text generation
        - **Information Retrieval:** Semantic search for question answering
        - **Sentiment Analysis:** VADER lexicon-based sentiment scoring
        - **Topic Modeling:** Latent Dirichlet Allocation (LDA) for unsupervised topic discovery
        - **Document Clustering:** Multiple document support for comparative analysis
        """)

if __name__ == "__main__":
    main()
