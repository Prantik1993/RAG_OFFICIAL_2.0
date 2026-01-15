from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from src.config import Config
from src.logger import get_logger
from src.exceptions import RAGChainError

logger = get_logger("RAGEngine")


try:
    FlashrankRerank.model_rebuild()
except Exception:
    pass
logger = get_logger("RAGEngine")
class RAGEngine:
    def __init__(self, vectorstore):
        try:
            logger.info("Initializing RAG Engine...")
            self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0,api_key=Config.OPENAI_API_KEY)
            
            # --- 1. RERANKING SETUP ---
            # Step A: Fetch 20 documents (broad search)
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
            
            # Step B: Rerank them to get the top 5 (accurate search)
            # Flashrank runs locally and is free
            logger.info("Loading Reranker (Flashrank)...")
            compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
            
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )
            
            self.store = {} # Memory storage
        except Exception as e:
            logger.critical(f"Failed to initialize RAG Engine: {e}")
            raise RAGChainError(f"RAG init failed: {e}")

    def _validate_input(self, user_input: str):
        """
        --- 2. PROMPT GUARDRAILS ---
        Basic checks to prevent spam or token overload.
        """
        if not user_input or not user_input.strip():
            raise ValueError("Input cannot be empty.")
        if len(user_input) > 2000:
            raise ValueError("Input is too long (max 2000 chars).")

    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_chain(self):
        try:
            # 1. Define how documents should look to the LLM
            # This tells the LLM: "Here is text from Page 5: ..."
            document_prompt = ChatPromptTemplate.from_template(
                "Content: {page_content}\nSource: Page {page}\n"
            )

            # 2. The System Prompt (Instructions)
            qa_system_prompt = (
                "You are a GDPR legal expert. "
                "Use the following retrieved context to answer the question. "
                "If you don't know the answer, say so. "
                "\n\n"
                "CITATION RULES:\n"
                "- You MUST cite the page number for every fact you mention.\n"
                "- Use the format: [Page X].\n"
                "- Example: 'Fines can be up to 20 million [Page 83].'"
                "\n\n"
                "{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # 3. Create the Chain with the document prompt
            question_answer_chain = create_stuff_documents_chain(
                self.llm, 
                qa_prompt,
                document_prompt=document_prompt,
                document_variable_name="context"
            )
            
            # 4. Connect everything
            context_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question... (rephrase logic)..."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_retriever = create_history_aware_retriever(self.llm, self.retriever, context_prompt)
            rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

            return RunnableWithMessageHistory(
                rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        except Exception as e:
            logger.error(f"Error building RAG chain: {e}")
            raise RAGChainError(f"Chain creation failed: {e}")