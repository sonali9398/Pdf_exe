from flask import Flask, request, redirect, session, send_file, render_template_string
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber
import pandas as pd

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

def get_pdf_text_and_tables(pdf_docs):
    text_dict = {}
    tables_dict = {}
    for pdf in pdf_docs:
        text = ""
        tables = []
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text()
                    tables += page.extract_tables()
            text_dict[pdf.filename] = text
            tables_dict[pdf.filename] = tables
        except Exception as e:
            print(f"Error processing {pdf.filename}: {e}")
            continue
    return text_dict, tables_dict

def get_text_chunks(text_dict):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_dict = {}
    for pdf_name, text in text_dict.items():
        chunks = text_splitter.split_text(text)
        chunks_dict[pdf_name] = chunks
    return chunks_dict

def get_vector_store(chunks_dict):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store_dict = {}

    for pdf_name, chunks in chunks_dict.items():
        batch_size = 100
        vector_store = None
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings)
            else:
                new_vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings)
                vector_store.merge_from(new_vector_store)
        vector_store.save_local(f"faiss_index_{pdf_name}")
        vector_store_dict[pdf_name] = vector_store

def get_conversational_chain():
    prompt_template = """
    you are experienced marine Human Resource Manager your task is I'll provide you resumes. You are an expert in matching this job description and giving the percentage:
    Task Instructions for Expert Assistant:
        1) Multiple PDF Handling:
            1) Ensure all provided PDF documents are processed.
            2) Do not miss any PDF when extracting information.
        2) Exact Extraction:
            1) Extract information exactly as it appears in the PDFs.
            2) Do not infer, summarize, or create any information outside of what is explicitly provided.
        3) Sequential Processing:
            1) Process each PDF in the order provided.
            2) Ensure information from each PDF is kept distinct.
        4) Clear Indication of Availability:
            1) If the answer is unavailable in any of the provided PDFs, clearly state: "The answer is not available in the provided PDFs."
        5) Consistent Formatting:
            1) Maintain consistent formatting.
            2) Clearly indicate which PDF each piece of information is extracted from.

    Example Response:
        PDF 1:
        information 
        PDF 2:
        information
        
        "The answer is not available in the provided PDFs."

        Your task:
        Follow the instructions above.
        Extract and provide the exact text from each PDF as specified.


    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def keyword_search(text, keywords):
    results = []
    for keyword in keywords:
        if keyword.lower() in text.lower():
            results.append(keyword)
    return results

def user_input(user_question, text_dict, tables_dict):
    keywords = user_question.split()
    found_keywords = {pdf_name: keyword_search(text, keywords) for pdf_name, text in text_dict.items()}
    
    response_text = ""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    for pdf_name in text_dict.keys():
        try:
            new_db = FAISS.load_local(f"faiss_index_{pdf_name}", embeddings, allow_dangerous_deserialization=True)
        except ValueError as e:
            print(f"Error loading FAISS index for {pdf_name}: {e}")
            continue
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response_text += f"PDF Name: {pdf_name}\n{response['output_text']}\n\n"
        if "table" in user_question.lower():
            tables = tables_dict.get(pdf_name, [])
            for table in tables:
                if table:  # Check if the table is not empty
                    df = pd.DataFrame(table[1:], columns=table[0])
                    response_text += f"Table from {pdf_name}:\n{df.to_string(index=False)}\n\n"

    print(response_text)
    return response_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_question = request.form['user_question']
        if 'pdf_files' not in request.files:
            return redirect(request.url)
        pdf_files = request.files.getlist('pdf_files')
        text_dict, tables_dict = get_pdf_text_and_tables(pdf_files)
        session['text_dict'] = text_dict
        session['tables_dict'] = tables_dict
        chunks_dict = get_text_chunks(text_dict)
        get_vector_store(chunks_dict)
        response_text = user_input(user_question, text_dict, tables_dict)
        return render_template_string(open("index.html").read(), response_text=response_text)

    return render_template_string(open("index.html").read())

if __name__ == "__main__":
    app.run(debug=True)
