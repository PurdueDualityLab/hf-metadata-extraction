
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

# System Design

Foundational LLM models are trained on very general domain corpora, making them less effective for domain-specific tasks. 
To enhance our LLM extraction, we based our system off of the RAG (retrieval-augmented generation) framework. 
The RAG framework consists of retrieving relevant information from knowledge base,
that could be used as context for LLM. Through retrieving relevant information, it is able to ground the LLM and reduce hallucinations. 
By inputing more relevant information into the LLM, we hope to reduce token use and ultimately reduce financial and computationl cost. Our goal is thus to input in a model card and extract relevant metadata from it.

https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/jumpstart/jumpstart-fm-rag.jpg


## Overall Pipeline

Our overall pipeline consists of 5 major steps: document loading, splitting,  retrieval, compression and extraction. Most of which are done through langchain's framework

https://blog.langchain.dev/content/images/size/w1000/2023/04/screenshot-2023-04-20-at-4.52.12-pm.png


(p.s. not my image hence the red box, will make image soon)

### Document Loading

We load the model card using Huggingface API, this allows us to access the model cards, and their tags. The model card could then be split through chunking, while important tags could be fed into the LLM through prompting for more complete background knowledge.

### Splitting 

https://www.pinecone.io/learn/chunking-strategies/ \
Rather than embedding the model cards (readme files) of the models as input documents, we apply chunking and embed the overall context and relationship of the structure of the markdown document. We thus transform the markdown document model cards into a list of Langchain.schema.document.Document. The document object contains the page contents and current header and subheaders the contents are in

ex: \
Document(page_content: "Markdown is widely used in blogging, instant messaging, online forums", metadata = {'Header 1': 'intro', 'Header 2': 'history'})

Using this method, we can retain the full information of the card while honoring the structure of the document for better embedding.

### Retrieval

We store our document splits in vectorstores after embedding them using OpenAIEmbeddings. OpenAIEmbeddings measures the relatedness of text strings. This allows us to provide a prompt and query through the relevat document splits to find relevnt documents for the prompt using Meta's FAISS (Facebook AI Similarity Search). The FAISS library provides us an easy way to quickly search through embeddings of document that are similar to our prompt. Retrieval of relevant documents help by leaving out irrelevant information that could be parsed into a LLM that could distract it from outputing accurate results, and reduces the overall financial cost of large input prompts into a LLM.

### Contextual Compression

Once we retrived our relevant document splits. We compress these splits into more condensed documents using OpenAI llm. This is different from retrieval, where in retrieval, we query for sections in the original model card that would be relevant to our extraction, in compression, we compress the relevant sections into small more concise text strings. These concise text strings aim to improve accuracy of the model while reducing overall cost.

### Extraction

The extraction is done through Langchain's extraction chain. The extraction chain allows for extraction with a desired schema using an OpenAI function call, allowing for a json structure extraction output. Through proper prompting, relevant document queries, and output schema, we are able to extract desired metadata from the model card.

## Prompting

Prompting is important as they are the instructions that a LLM bases its actions off of. In our pipeline there exists 3 different steps that require prompting: retrieval, compression, and extraction

### Retrieval Prompting

(still tinkering)

### Compression Prompting

(still tinkering)

### Extraciton Prompting

(still tinkering)




during chunking process we found out that many lesser well documnented models have the following text appended to the start of the model card:

\<!-- This model card has been generated automatically according to the information the Trainer had access to. You should probably proofread and complete it, then remove this comment. -->

>>>>>>> 

