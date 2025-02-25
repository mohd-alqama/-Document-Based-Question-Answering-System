from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context only, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide any wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
    

def main():
    load_dotenv()

    pdf_read = PdfReader(r"D:\attention.pdf")

    # extracting texts
    if pdf_read is not None:
        text = ""
        for page in pdf_read.pages:
            text += page.extract_text()

        # splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)

        # embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # user 
        user_question = input("Ask a Question from the PDF: ")

        if user_question:   
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
            docs = vector_store.similarity_search(user_question)

            chain = get_conversational_chain()
            response = chain(
                {"input_documents":docs, "question": user_question}
                , return_only_outputs=True)
            
            print(response["output_text"])
    

if __name__ == "__main__":
    main()