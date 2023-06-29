Check our SIGIR2021 short paper: https://dl.acm.org/doi/10.1145/3404835.3463048

This checkpoint is a variant of monot5 (T5 pointwise re-ranking model). 
Specifically, we fuse the "P2Q (i.e. doc2query)" and "Rank (i.e. passage ranking)" to learn the **discriminative** view (Rank) and **geneartive** view (P2Q). 

We found that under the specific **mixing ratio** of these two task, the effectiveness of passage re-ranking improves on par with monot5-3B models.
Hence, you can try to do both the task with this checkpoint by the following input format:
- P2Q: Document: *\<here is a document or a passage\>* Translate Document to Query:
- Rank: Query: *\<here is a query\>* Document: *\<here is a document or a passage\>* Relevant: 

which the outputs will be like:
- P2Q: *\<relevant query of the given text\>*
- Rank: *true* or *false*

```
Note that we usually use the logit values of *true*/ *false* token from T5 reranker as our query-passage relevant scores
Note the above tokens are all case-sensitive.
```