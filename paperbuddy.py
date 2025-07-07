from dotenv import load_dotenv
from faiss import IndexFlatL2
import json
from langchain.document_loaders import ArxivLoader
from langchain.document_transformers import LongContextReorder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
import logging
from operator import itemgetter
import os
import pickle


EMBEDDING_MODEL_NAME = 'nvidia/nv-embed-v1'
CHAT_MODEL_NAME = 'meta/llama-3.3-70b-instruct'
PAPERBUDDY_SAVE_DIR = './paperbuddy_data'
CONV_STORE_IDX_PATH = f'{PAPERBUDDY_SAVE_DIR}/conv_store_faiss_index'
DOC_STORE_IDX_PATH = f'{PAPERBUDDY_SAVE_DIR}/doc_store_faiss_index'
ARXIV_PAPER_IDS_PATH = f'{PAPERBUDDY_SAVE_DIR}/arxiv_paper_ids.pkl'


class PaperBuddy:
    def __init__(self, arxiv_paper_ids, load_stores=False):
        self.arxiv_paper_ids = arxiv_paper_ids

        # get logger
        self.logger = logging.getLogger('PaperBuddy_Logger')
        self.logger.setLevel(logging.INFO)

        # load environment variables
        if os.path.isfile('./.env'):
            load_dotenv()
        self.logger.info('Loaded environment variables from .env file.')

        # specify embedding model and chat model
        embedding_model = NVIDIAEmbeddings(model=EMBEDDING_MODEL_NAME)
        chat_model = ChatNVIDIA(model=CHAT_MODEL_NAME, temperature=0)
        self.logger.info(f'Using {EMBEDDING_MODEL_NAME} embedding model.')
        self.logger.info(f'Using {CHAT_MODEL_NAME} chat model.')

        # get chunker
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=100,
            separators=['\n\n', '\n', '.', '!', '?', ''])

        # utility function for loading papers into document store
        def _add_papers_to_doc_store(arxiv_paper_ids):
            self.logger.info(
                f'Adding arXiv papers with IDs {arxiv_paper_ids} to document '
                + 'store')
            # load arXiv papers from their IDs
            papers = list(map(
                lambda i: ArxivLoader(query=i).load(), arxiv_paper_ids))

            # remove the References section from the papers
            for paper in papers:
                content = json.dumps(paper[0].page_content)
                if 'References' in content:
                    paper[0].page_content = content[
                        :content.rfind('References')]

            # split papers into chunks, filter out very small chunks
            papers_chunks = [
                chunker.split_documents(paper) for paper in papers]
            papers_chunks = [
                [c for c in paper_chunks if len(c.page_content) > 50] for
                paper_chunks in papers_chunks]
            self.logger.info('Split papers into chunks.')

            # create metadata chunks
            available_papers = 'Available Papers: '
            metadata_chunks = []
            for paper_chunks in papers_chunks:
                paper_name = paper_chunks[0].metadata['Title']
                available_papers += paper_name + ', '
                metadata_chunks.append(
                    f'Metadata for {paper_name}: '
                    + str(paper_chunks[0].metadata))
            metadata_chunks = [available_papers[:-2]] + metadata_chunks
            self.logger.info('Created paper metadata chunks.')

            # add chunks to docstore
            self.doc_store.merge_from(
                FAISS.from_texts(metadata_chunks, embedding_model))
            for paper_chunks in papers_chunks:
                self.doc_store.merge_from(
                    FAISS.from_documents(paper_chunks, embedding_model))
            self.logger.info(
                'Added paper and metadata chunks to document store.')

        # create/load conversation and document stores
        if load_stores:
            # load conversation store from saved FAISS index
            self.conv_store = FAISS.load_local(
                CONV_STORE_IDX_PATH, embedding_model,
                allow_dangerous_deserialization=True)
            self.logger.info(
                'Loaded PaperBuddy conversation store from saved FAISS '
                + 'index.')

            # load document store from saved FAISS index
            self.doc_store = FAISS.load_local(
                DOC_STORE_IDX_PATH, embedding_model,
                allow_dangerous_deserialization=True)
            self.logger.info(
                'Loaded PaperBuddy document store from saved FAISS index.')

            # if there are new arXiv paper IDs, update the arXiv paper IDs
            # list and add the new papers to the document store
            with open(ARXIV_PAPER_IDS_PATH, 'rb') as f:
                saved_paper_ids = list(pickle.load(f))
            new_paper_ids = list(filter(
                lambda i: i not in set(saved_paper_ids), arxiv_paper_ids))
            self.arxiv_paper_ids = saved_paper_ids + new_paper_ids
            _add_papers_to_doc_store(new_paper_ids)
        else:
            # create in-memory conversation store, that stores the embeddings
            # (of chunks of the conversation) generated by the embedding model
            # similarity is calculated via Euclidean distance (L2 norm)
            embed_dims = len(
                embedding_model.embed_query('lorem ipsum dolor'))
            self.conv_store = FAISS(
                embedding_function=embedding_model,
                index=IndexFlatL2(embed_dims),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=False)
            self.logger.info('Created FAISS conversation store.')

            # create in-memory document store, that stores the embeddings
            # (of chunks of the papers) generated by the embedding model
            # similarity is calculated via Euclidean distance (L2 norm)
            self.doc_store = FAISS(
                embedding_function=embedding_model,
                index=IndexFlatL2(embed_dims),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=False)
            self.logger.info('Created FAISS document store.')

            # split papers into chunks, add to document store
            _add_papers_to_doc_store(self.arxiv_paper_ids)

        # specify prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', '''
             You are an assistant named PaperBuddy that answers questions
             about research papers.

             The user asked: {input}

             Relevant info retrieved from conversation history:

             {conv_history}

             Relevant info retrieved from paper(s):

             {context}

             Answer only based on the retrieved info, and cite the sources used
             with "Sources:" followed by a bulleted list at the end of your
             answer. If you are citing papers, use only the paper name,
             without any additional text. If you are citing conversation
             history, say "Conversation history".

             Do not cite sources that you did not use in your answer. Only cite
             sources you used in your answer. This applies to both papers and
             conversation history

             Answer only the question asked, succinctly. Do not include any
             irrelevant information in your answer.

             Your response should be as concise and precise as possible. It
             should only include the answer to the question with any details
             that are necessary for the user to understand the answer. Do not
             include unnecessary information in your answer. Do not make your
             answer longer once it has answered the question.
             '''),
            ('user', '{input}')
        ])

        # utility function for saving input and output to conversation store
        def _save_input_output(state):
            # chunk input and output
            split_input = list(map(
                lambda x: x + ' [Quote from user]',
                chunker.split_text(state['input'])))
            split_output = list(map(
                lambda x: x[:x.rfind('References:\n')]
                + ' [Quote from PaperBuddy]',
                chunker.split_text(state['output'].content)))
            self.logger.info(
                'Split user input and PaperBuddy output into chunks.')
            self.conv_store.add_texts(split_input + split_output)
            self.logger.info('Added chunks to conversation store.')
            return state['output']

        # utility function for formatting context string from retrieved
        # chunks
        def _get_context_str(chunks, paper_chunks=False):
            context_str = ''
            for chunk in chunks:
                # source suffix does not need to be specified for conversation
                # chunks or paper metadata chunks, since they already have the
                # sources specified
                source_suffix = ''
                if paper_chunks:
                    # if chunk has metadata with title, it is a chunk
                    # of text directly from a paper, and needs a source suffix
                    if hasattr(chunk, 'metadata'):
                        metadata = chunk.metadata
                        if 'Title' in metadata.keys():
                            paper_title = chunk.metadata['Title']
                            source_suffix = f' [Quote from {paper_title}]'
                context_str += \
                    chunk.page_content + source_suffix + '\n\n'
            return context_str

        # utility function for logging
        def _log_message(input, message):
            self.logger.info(message)
            return input

        # create retrieval augmented generation (RAG) chain
        long_reorder = RunnableLambda(LongContextReorder().transform_documents)
        retrieval_chain = (
            {'input': (lambda x: x)}
            | RunnableAssign({
                'conv_history': itemgetter('input')
                | self.conv_store.as_retriever(search_kwargs={'k': 3})
                | long_reorder
                | _get_context_str
                | RunnableLambda(lambda x: _log_message(
                    x, 'Retrieved relevant conversation history.'))})
            | RunnableAssign({
                'context': itemgetter('input')
                | self.doc_store.as_retriever(search_kwargs={'k': 12})
                | long_reorder
                | RunnableLambda(lambda x: _get_context_str(
                    x, paper_chunks=True))
                | RunnableLambda(lambda x: _log_message(
                    x, 'Retrieved relevant context from papers.'))}))
        generation_chain = (
            RunnableLambda(lambda x: x)
            | RunnableAssign({'output': prompt_template | chat_model})
            | RunnableLambda(lambda x: _log_message(
                    x, 'Generated chat model response to user input with '
                    + 'retrieved conversation history and context.'))
            | RunnableLambda(lambda x: _save_input_output(x))
            | StrOutputParser())
        self.rag_chain = retrieval_chain | generation_chain

    def prompt(self, prompt):
        return self.rag_chain.invoke(prompt)

    def save_data(self):
        # create directory where data will be saved, if it does not already
        # exist
        if not os.path.isdir(PAPERBUDDY_SAVE_DIR):
            os.mkdir(PAPERBUDDY_SAVE_DIR)

        # save arXiv paper IDs
        with open(ARXIV_PAPER_IDS_PATH, 'wb') as f:
            pickle.dump(self.arxiv_paper_ids, f)
        self.logger.info('Saved arXiv paper IDs.')

        # save conversation store FAISS index
        self.conv_store.save_local(CONV_STORE_IDX_PATH)
        self.logger.info(
            'Saved FAISS index for PaperBuddy conversation store.')

        # save document store FAISS index
        self.doc_store.save_local(DOC_STORE_IDX_PATH)
        self.logger.info('Saved FAISS index for PaperBuddy document store.')
