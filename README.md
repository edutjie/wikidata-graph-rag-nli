# Towards an Open NLI LLM-based System for KGs: A Case Study of Wikidata

![Pipeline](/images/pipeline-latest.png "Pipeline")

The proposed GraphRAG pipeline is illustrated above. The explanation of each step is as follows:

- Given the user's question, the LLM extracts relevant entities that exist within the question. For example, if the question is about Humans born in Indonesia, the LLM response would be `['Human', 'Indonesia']`.
- For every extracted entity, the keyword is used to retrieve the top-5 related Wikidata entities using the Wikidata API. The retrieved data includes the URI (ID), label, and description of the entity.
- The LLM determines the most suitable entity from the list of entities retrieved before, based on the context of the user's question and the description of the entity.
- The LLM generates the most appropriate SPARQL query to answer the user's question, given the determined entities and the pre-defined top-100 frequently used properties in general within Wikidata. The SPARQL query must contain the given properties and entities only, to reduce hallucination.
- The generated SPARQL query will be executed against the Wikidata endpoint, and the result will be passed as the context for the LLM to generate a relevant response to answer the user's question.

## Instructions

To run the app locally, please run:

```shell
streamlit run main.py
```

Deployed app: [here](https://wikidata-graph-rag-nli.streamlit.app/)
