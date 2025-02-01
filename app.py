import streamlit as st
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import pipeline
import torch
import base64
import textwrap
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Create offload directory if it doesn't exist
os.makedirs('model_cache', exist_ok=True)

# model and tokenizer loading
checkpoint = "LaMini-T5-738M"
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)

# Determine if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model with appropriate settings
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32,
    device_map="auto" if device == "cuda" else None,
    offload_folder="model_cache",
    local_files_only=False
)

if device == "cpu":
    base_model = base_model.to(device)

@st.cache_resource
def llm_pipeline(is_detailed=False):
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=512 if is_detailed else 256,
        do_sample=True,
        temperature=0.5 if is_detailed else 0.3,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm(is_detailed=False):
    llm = llm_pipeline(is_detailed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever(
        search_kwargs={"k": 5 if is_detailed else 2}
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_both_answers(instruction, progress_bar):
    try:
        progress_bar.progress(0, "Starting to generate answers...")
        time.sleep(0.5)  # Simulate initial loading
        
        # Generate concise answer
        progress_bar.progress(20, "Generating concise answer...")
        qa_concise = qa_llm(is_detailed=False)
        concise_instruction = f"Provide a brief and concise answer in 2-3 sentences: {instruction}"
        concise_response = qa_concise(concise_instruction)
        concise_answer = concise_response['result']
        concise_metadata = concise_response.get('source_documents', [])
        
        progress_bar.progress(50, "Processing concise answer...")
        time.sleep(0.5)  # Simulate processing time
        
        # Generate detailed answer
        progress_bar.progress(60, "Generating detailed answer...")
        qa_detailed = qa_llm(is_detailed=True)
        detailed_instruction = f"Provide a detailed and comprehensive answer with examples and explanations: {instruction}"
        detailed_response = qa_detailed(detailed_instruction)
        detailed_answer = detailed_response['result']
        detailed_metadata = detailed_response.get('source_documents', [])
        
        progress_bar.progress(90, "Finalizing responses...")
        time.sleep(0.5)  # Simulate final processing
        
        progress_bar.progress(100, "Complete!")
        
        return {
            'concise': {'answer': concise_answer, 'metadata': concise_metadata},
            'detailed': {'answer': detailed_answer, 'metadata': detailed_metadata}
        }
    except Exception as e:
        st.error(f"Error processing answers: {str(e)}")
        return None

def display_answer(answer_data, show_detailed):
    if answer_data is None:
        st.error("Failed to generate answers. Please try again.")
        return
        
    # Display the appropriate answer based on toggle state
    answer = answer_data['detailed'] if show_detailed else answer_data['concise']
    
    st.success("Answer:")
    st.write(answer['answer'])
    
    # Display source documents
    if answer['metadata']:
        st.info("Source Documents:")
        for i, doc in enumerate(answer['metadata'], 1):
            with st.expander(f"Source {i}"):
                st.markdown(f"**Page:** {doc.metadata.get('page', 'Unknown')}")
                st.markdown(f"**Content:**")
                st.write(textwrap.fill(doc.page_content, width=80))
                st.markdown(f"**Source Path:** {doc.metadata.get('source', 'Unknown')}")

def main():
    st.title("Search Your PDF📄📚")

    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI-powered Question and Answering app that responds to questions about your PDF file.
            Toggle between concise and detailed answers using the switch below.
            """
        )

    # Use st.chat_input for interactive conversation input
    question = st.chat_input("Enter your question:")

    if 'answers' not in st.session_state:
        st.session_state.answers = None

    if question:
        if not question.strip():
            st.warning("Please enter a valid question!")
        else:
            st.info(f"Your Question: {question}")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            st.session_state.answers = process_both_answers(question, progress_bar)
            
            # Clear the progress bar after completion
            time.sleep(0.5)  # Show completed progress briefly
            progress_bar.empty()
    
    # Add the toggle switch for detailed/concise display
    if st.session_state.answers is not None:
        show_detailed = st.toggle(
            "Show Detailed Answer",
            False,
            help="Switch between concise and detailed answers"
        )
        
        # Display current mode
        st.info(f"Currently showing: {'Detailed' if show_detailed else 'Concise'} Answer")
        
        # Display the appropriate answer
        display_answer(st.session_state.answers, show_detailed)

if __name__ == '__main__':
    main()  
