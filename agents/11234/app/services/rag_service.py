import requests
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.schema.runnable import RunnableMap
from app.config import API_URL, HEADERS, INDEX_PATH
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, convert_system_message_to_human=True,rate_limiter=rate_limiter).with_retry(stop_after_attempt=4)

def query(payload):
    """Query the Hugging Face model"""
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error querying HuggingFace API: {str(e)}")
        # Return a default response format to avoid errors
        return [{"generated_text": "Error processing request. ###Answer### I couldn't process this request."}]

def qna(question, kb):
    """Simple question answering without RAG chain"""
    t = kb.similarity_search(question)
    context = ''
    metadata = []
    for i in t:
        context = context + i.page_content
        metadata.append(i.metadata)
    
    prompt = f"""Generate the answer using provided context and question.
    Answer should be clear and informative.
    The answer should be in the same language as the question.
    Generate the answer according to question and context.
    If question and context doesn't match then answer should be 'Sorry I don't know. Please ask question relevant to document.
    Don't add any other extract information.
    Output should be only answer dont return the context.
    Answer should be end with ###
    context : {context}
    Question :{question}
    ###Answer### : """

    output = query({
        "inputs": prompt,
    })
    
    try:
        index = output[0]['generated_text'].index('###Answer###')
        answer = output[0]['generated_text'][index+13:]
    except (IndexError, KeyError, ValueError):
        answer = "Sorry, I couldn't generate an answer. Please try again."
    
    return answer, output[0]['generated_text'] if output and len(output) > 0 else "", metadata

def format_docs(docs):
    """Format documents for prompt"""
    return "\n\n".join(f"Context: {doc.page_content}\n Metadata: {doc.metadata}\n" for doc in docs)

def create_rag_chain(embeddings_provider, username=None):
    """
    Create a retrieval chain that filters results by username.
    
    Args:
        embeddings_provider: The embeddings model to use
        username: If provided, will only retrieve documents uploaded by this user
    """
    try:
        # Load the knowledge base
        knowledge_base = FAISS.load_local(INDEX_PATH, index_name='AZ', embeddings=embeddings_provider)
        
        # Create a username-filtered retriever if username is provided
        if username:
            print(f"Creating user-specific retriever for {username}")
            # Create a filter to only get documents from this user
            # retriever = knowledge_base.as_retriever(
            #     search_kwargs={
            #         "k": 4,  # Number of documents to retrieve
            #         "filter": lambda doc: doc.metadata.get("username") == username
            #     }
            # )
            # retriever = knowledge_base.as_retriever(
            #     search_kwargs={'k': 6, 'filter': {"username": username}}
                
            # )
            retriever = knowledge_base.as_retriever(search_kwargs={"k": 4})
        else:
            # No filtering if no username provided (fallback)
            print("Creating retriever without user filtering")
            retriever = knowledge_base.as_retriever(search_kwargs={"k": 4})
        
        prompt_template = """
As an AI Assistant designed to handle customer service inquiries, your task is to generate responses based on the provided context, question, and chat history.

- **Response Generation**: 
  Provide a clear, concise, and informative answer that directly addresses the user's question. Ensure the response is in the same language as the question.
  The tone and style of your answer should align with the tone of the question, for easy understanding.

- **Context & Question Alignment**:
  If the context and question do not align or match, do not return the reference document, or leave it empty.
  If the question does not match the context, respond with: "Sorry, I don't know. Please ask a question relevant to the document."

- **Metadata**:
  When generating the response based on the context, include the metadata of the context used to generate the answer. Label it as "Reference Document" and ensure the metadata is presented in a clear, legible format.

- **Chat History**:
  Carefully review the chat history to ensure your response aligns smoothly with the previous conversation and helps maintain a coherent flow.

- **Exclusion of Supplementary Information**:
  Do not include any extra information, nor should you return the context or metadata in the response unless explicitly asked for. Focus only on the answer.

- **Greetings & Day-to-Day Questions Handling**:
  If the question is a greeting or a casual day-to-day life question (e.g., "How are you?", "Good morning", "Hello!", "How will you help me?"), respond appropriately without referencing the document or providing metadata. Only return the greeting or general response.

Context: {context}
Question: "{question}"
Chat History: {chat_history}
"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ] 
        )
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        rag_chain_from_docs = (
            {
                "context": lambda input: format_docs(input["documents"]),
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableMap(
            {"documents": history_aware_retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        ) | {
            "documents": lambda input: [doc.metadata for doc in input["documents"]],
            "answer": rag_chain_from_docs,
        }
        print("RAG chain created successfully rag service")
        # print(llm.invoke("What is the capital of France?"))
        # print(rag_chain_with_source.invoke({"input": "hello", 
        #         "chat_history": [{"role": "user", "content": "hello"}]}))
        return rag_chain_with_source, knowledge_base
    except Exception as e:
        print(f"Error creating RAG chain: {str(e)}")
        return None, None

def create_rag_chain_streaming(embeddings_provider, username=None):
    """
    Create a streaming retrieval chain that filters results by username.
    
    Args:
        embeddings_provider: The embeddings model to use
        username: If provided, will only retrieve documents uploaded by this user
    
    Returns:
        A streaming RAG chain that yields tokens as they're generated
    """
    try:
        # Load the knowledge base
        knowledge_base = FAISS.load_local(INDEX_PATH, index_name='AZ', embeddings=embeddings_provider)
        
        # Create a username-filtered retriever if username is provided
        if username:
            print(f"Creating user-specific streaming retriever for {username}")
            retriever = knowledge_base.as_retriever(
                search_kwargs={'k': 6, 'filter': {"username": "admin"}}
            )
        else:
            # No filtering if no username provided (fallback)
            print("Creating streaming retriever without user filtering")
            retriever = knowledge_base.as_retriever(search_kwargs={"k": 4})
        
        prompt_template = """
As an AI Assistant designed to handle customer service inquiries, your task is to generate responses based on the provided context, question, and chat history.

- **Response Generation**: 
  Provide a clear, concise, and informative answer that directly addresses the user's question. Ensure the response is in the same language as the question.
  The tone and style of your answer should align with the tone of the question, for easy understanding.

- **Context & Question Alignment**:
  If the context and question do not align or match, do not return the reference document, or leave it empty.
  If the question does not match the context, respond with: "Sorry, I don't know. Please ask a question relevant to the document."

- **Metadata**:
  When generating the response based on the context, include the metadata of the context used to generate the answer. Label it as "Reference Document" and ensure the metadata is presented in a clear, legible format.

- **Chat History**:
  Carefully review the chat history to ensure your response aligns smoothly with the previous conversation and helps maintain a coherent flow.

- **Exclusion of Supplementary Information**:
  Do not include any extra information, nor should you return the context or metadata in the response unless explicitly asked for. Focus only on the answer.

- **Greetings & Day-to-Day Questions Handling**:
  If the question is a greeting or a casual day-to-day life question (e.g., "How are you?", "Good morning", "Hello!", "How will you help me?"), respond appropriately without referencing the document or providing metadata. Only return the greeting or general response.

Context: {context}
Question: "{question}"
Chat History: {chat_history}
"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ] 
        )
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Create streaming chain that yields tokens
        def create_streaming_chain():
            async def stream_response(input_data):
                # Get documents first
                documents = await history_aware_retriever.ainvoke({
                    "input": input_data["input"],
                    "chat_history": input_data["chat_history"]
                })
                
                # Format the context
                context = format_docs(documents)
                
                # Create the prompt
                formatted_prompt = prompt.format(
                    context=context,
                    question=input_data["input"],
                    chat_history=input_data["chat_history"]
                )
                
                # Stream the response
                full_response = ""
                async for chunk in llm.astream(formatted_prompt):
                    if chunk.content:
                        full_response += chunk.content
                        yield {
                            "token": chunk.content,
                            "full_response": full_response,
                            "documents": [doc.metadata for doc in documents],
                            "finished": False
                        }
                
                # Send final response
                yield {
                    "token": "",
                    "full_response": full_response,
                    "documents": [doc.metadata for doc in documents],
                    "finished": True
                }
            
            return stream_response
        
        return create_streaming_chain(), knowledge_base
        
    except Exception as e:
        print(f"Error creating streaming RAG chain: {str(e)}")
        return None, None
