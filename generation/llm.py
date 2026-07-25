import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())

# Cached instances (lazy initialization)
_llm = None
_retriever = None
_embeddings = None
_vectorstore = None

def get_llm(api_key: str | None = None):
    """Get or create the LLM instance.

    If api_key is provided explicitly (per-session), bypass the module cache so
    different users' keys don't bleed into each other.
    """
    global _llm
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")
    if api_key:
        # ponytail: no per-key cache; ChatGroq is a lightweight config object
        return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, api_key=key)
    if _llm is None:
        _llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, api_key=key)
    return _llm

def get_retriever(k: int = 5):
    """Get or create the FAISS retriever."""
    global _retriever, _embeddings, _vectorstore
    if _retriever is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_db')
        _vectorstore = FAISS.load_local(db_path, _embeddings, allow_dangerous_deserialization=True)
        _retriever = _vectorstore.as_retriever(search_kwargs={"k": k})
    return _retriever

def get_prompt():
    """Get the prompt template."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    return ChatPromptTemplate.from_messages([
        ("system", f"""Today is {now}. You are the project-intelligence analyst for a software team. \
Your job is to reconstruct what actually happened on the project from its scattered communications \
and answer the manager's question with precision.

Ground every claim in the message log below. Follow these rules:
- Answer ONLY from the provided messages. If the log doesn't cover the question, say so plainly \
("The communications don't mention that") rather than guessing.
- Be specific: name the people, dates, numbers, decisions, and blockers involved. Vague summaries \
are worthless to a manager.
- When something is a decision, milestone, blocker, or its resolution, say which — managers think \
in those terms.
- Attribute claims to who said them when it matters ("Priya flagged the latency regression on the 21st").
- Be concise. Lead with the answer, then the supporting detail. No preamble, no "based on the logs".

Message log:
{{context}}"""),
        ("human", "{question}")
    ])

def retrieve_context(question: str, k: int = 5):
    """Retrieve relevant documents and return them with formatted context."""
    retriever = get_retriever(k)
    docs = retriever.invoke(question)
    
    context_parts = []
    for doc in docs:
        meta = doc.metadata
        context_parts.append(
            f"[{meta.get('timestamp', 'Unknown')}] {meta.get('author', 'Unknown')} via {meta.get('source', 'Unknown')}:\n{doc.page_content}"
        )
    context_string = "\n\n".join(context_parts)
    
    return docs, context_string

def stream_answer(question: str, history: list[dict] | None = None, api_key: str | None = None):
    """Like ask_question but yields text chunks for st.write_stream."""
    docs, context_string = retrieve_context(question)
    llm = get_llm(api_key)

    if history:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        sys_template = get_prompt().messages[0].prompt.template
        msgs = [SystemMessage(content=sys_template.replace("{context}", context_string))]
        for h in history[:-1]:
            if h["role"] == "user":
                msgs.append(HumanMessage(content=h["text"]))
            else:
                msgs.append(AIMessage(content=h["text"]))
        msgs.append(HumanMessage(content=question))
        for chunk in llm.stream(msgs):
            yield chunk.content
    else:
        chain = get_prompt() | llm
        for chunk in chain.stream({"context": context_string, "question": question}):
            yield chunk.content


def ask_question(question: str, history: list[dict] | None = None, api_key: str | None = None) -> str:
    """Ask a question and get an answer using RAG.

    history: list of {"role": "user"|"assistant", "text": str} from previous turns.
    api_key: per-session Groq key; bypasses module cache when provided.
    """
    docs, context_string = retrieve_context(question)
    llm = get_llm(api_key)

    if history:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        sys_template = get_prompt().messages[0].prompt.template
        msgs = [SystemMessage(content=sys_template.replace("{context}", context_string))]
        for h in history[:-1]:
            if h["role"] == "user":
                msgs.append(HumanMessage(content=h["text"]))
            else:
                msgs.append(AIMessage(content=h["text"]))
        msgs.append(HumanMessage(content=question))
        response = llm.invoke(msgs)
    else:
        chain = get_prompt() | llm
        response = chain.invoke({
            "context": context_string,
            "question": question
        })

    return response.content

# 5. Interactive Q&A loop
if __name__ == "__main__":
    print("=" * 50)
    print("Project Assistant (RAG-powered)")
    print("Ask questions about the project. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\nSearching relevant context and thinking...\n")
        answer = ask_question(question)
        print("Answer:")
        print(answer)
