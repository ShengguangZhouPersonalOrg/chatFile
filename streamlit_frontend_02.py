from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms import CTransformers
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import NotebookLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
import os


# initialization
# llama2 LLM
def init_model():
    PATH = "/Users/shengguang/Desktop/llama-main/llama-2-7b-chat.ggmlv3.q8_0.bin"
    llm = CTransformers(model=PATH,
                        model_type = 'llama',
                        max_new_token = 4096,
                        temperature = 0.01,
                        f16_kv=True,
                        streaming = True
                        )
    # embedding
    embeddings_model = HuggingFaceEmbeddings(model_name = 'shibing624/text2vec-base-chinese')
    return llm,embeddings_model
llm,embeddings_model = init_model()


# backend - load file,not directly used, called by doc loader.
# modify for adding more file types
supported_format = ['docx','doc','pdf','txt','csv','xlsx','ipynb','pptx']
def read_file(file):
    # streamlit version, add this line for streamlit
    temp_file_path = os.path.join('/tmp', file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(file.getvalue())

    if temp_file_path.endswith(('.docx','doc')):
        loader = Docx2txtLoader(temp_file_path)
    elif temp_file_path.endswith('.pdf'):
        loader = PyPDFLoader(temp_file_path)
    elif temp_file_path.endswith('.txt'):
        loader = TextLoader(temp_file_path)
    elif temp_file_path.endswith('.csv'):
        loader = CSVLoader(temp_file_path)
    elif temp_file_path.endswith('.xlsx'):
        loader = UnstructuredExcelLoader(temp_file_path)
    elif temp_file_path.endswith('.ipynb'):
        loader = NotebookLoader(temp_file_path,include_outputs=True,
                                max_output_length=100,remove_newline=True)
    elif temp_file_path.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(temp_file_path)
    else:
        st.write('暂时不支持该文件格式')

    content = loader.load()
    os.remove(temp_file_path)
    return content

# Loader for files
def doc_loader(files):
    # upload a list
    merged_loader = [read_file(file) for file in files]
    return merged_loader

# file split, called by file_split_thru, not called directly
def file_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators = ['\n\n', '\n', '(?<=\. )', ' ', '']
    )
    chunks = text_splitter.split_documents(text)
    return chunks

# Split thru a list of Documents
def file_split_thru(loader):
    for i in range(len(loader)):
        loader[i] = file_splitter(loader[i])
    return loader

# create embedding
def create_vectorDB(doc_loader):
    try:
        db_doc = FAISS.from_documents(file_splitter(doc_loader[0]),embeddings_model)
        if len(doc_loader)>1:
            for i in range(1,len(doc_loader)):
                db_i = FAISS.from_documents(file_splitter(doc_loader[i]),embeddings_model)
                db_doc.merge_from(db_i)
    except:
        print('Wrong file')
    return db_doc

# embedding merge
def embedding_merge(db1,db2):
    return db1.merge_from(db2)

# embedding delete
def embedding_delete(db,index):
    return db.delete([db.index_to_docstore_id[index]])


# QA retrieval chain
def QA_chain(vectorstore):
    # Define memory
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      input_key='question',
                                      output_key='answer',
                                      return_messages=True)
    # QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),  # k can be optimized
                                                     return_source_documents=True,
                                                     verbose=True,
                                                     memory=memory)
    return qa_chain


# prompt calling LLM
def prompt_call(prompt_input,callback):
    # prompt template
    prompt_template = '''
     {user_input}
    '''
    prompt = PromptTemplate(input_variables = ['user_input'],template = prompt_template)
    final_prompt = prompt.format(user_input = prompt_input)
    # call by session_state so it fits different cases
    output = st.session_state.conversation({"question": final_prompt},callbacks = callback)
    return output

# conversation_chain
def conversation_chain(input_prompt,callback):
    # Define memory
    memory = ConversationBufferMemory()
    # conversation chain
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory)
    template = """You are a helpful assistant. You do not respond as 'Human' or pretend to be 'Human'. You only respond once as 'AI'. 
    You also won't say those things that you are thinking as well, such as *smiling*, *excitedly*. """
    conversation_output = conversation.run(template + input_prompt, callbacks = callback)
    return conversation_output




# frontend - done by streamlit
# initialization theme
def theme():
    # st.set_page_config(page_title="💬 ChatPDF")
    st.title('💬 ChatPDF')
    with st.chat_message("assistant"):
        initial_message = "您好，请问有什么可以帮助您的?"
        st.markdown(initial_message)
theme()

# initialization for session state
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "db" not in st.session_state:
        st.session_state.db = None
init_session()

# Display chat, done by passing into message list as a dict
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input widget and reaction
if prompt := st.chat_input("输入问题"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user","content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Conversation Chain
    if st.session_state.conversation is None:
        with st.chat_message("assistant"):
            message_assistent = st.empty()
            st_callback = StreamlitCallbackHandler(message_assistent)
            output = message_assistent.markdown(conversation_chain(prompt,callback = [st_callback]))
        st.session_state.messages.append({"role": "assistant", "content": output})

    # case for chatPDF
    else:
        with st.chat_message("assistant"):
            message_assistent = st.empty()
            st_callback = StreamlitCallbackHandler(message_assistent)
            output = message_assistent.markdown(prompt_call(prompt,callback = [st_callback])['answer'])
            # retrieval_output = prompt_call(prompt)['answer']
        st.session_state.messages.append({"role": "assistant", "content": output})

# sidebar
with st.sidebar:
    # uploading file
    upload_file = st.file_uploader(label='**上传文件**',accept_multiple_files=True,
                                   type = supported_format,
                                   help = '拖拽或选择文件上传，等待模型加载完毕后即可与文件对话')
    # Check if there are uploaded files
    if upload_file:
        # Keep a count of previously uploaded files
        if "prev_file_count" not in st.session_state:
            st.session_state.prev_file_count = 0

        current_file_count = len(upload_file)

        # New files have been added
        if current_file_count > st.session_state.prev_file_count:
            with st.spinner('正在读取中'):
                new_files = upload_file[st.session_state.prev_file_count:]
                loader = doc_loader(new_files)
                splitted_loader = file_split_thru(loader)
                new_db = create_vectorDB(splitted_loader)
                # If there's an existing DB, merge. Otherwise, assign the new DB.
                if st.session_state.db is not None:
                    embedding_merge(st.session_state.db, new_db)
                else:
                    st.session_state.db = new_db

            st.session_state.conversation = QA_chain(st.session_state.db)
        # TODO: Handle file removal, update DB accordingly.
        # Update the previous file count
        st.session_state.prev_file_count = current_file_count
