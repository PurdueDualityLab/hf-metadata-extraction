
# pre process

## load

## split

### langchain markdown header text splitter
    
https://www.pinecone.io/learn/chunking-strategies/
rather than embedding model cards (readme) as input documents, we hope to apply chunking and embed the overall context and relationship of the structure of the markdown document. We thus transform the markdown document model cards into 

Here is an simple markdown document

ex

Through markdown_splitter.split_text() we split model cards into list of DOCUMENTS (https://docs.langchain.com/docs/components/schema/document)

ex

Through character_text_splitter we then split when identifying "\n\n"

ex

## store

## retrive

### contextual compression

#### FAISS
    Facebook AI Similarity Search (Faiss) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.

# generate

## LLM

## prompt

## schema

# post process






during chunking process we found out that many lesser well documnented models have the following text appended to the start of the model card:

\<!-- This model card has been generated automatically according to the information the Trainer had access to. You should probably proofread and complete it, then remove this comment. -->

>>>>>>> 
