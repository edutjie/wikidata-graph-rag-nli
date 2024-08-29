import re, requests
from SPARQLWrapper import SPARQLWrapper, JSON

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate, HuggingFaceHub, HuggingFacePipeline, LLMChain
from langchain.output_parsers import (
    ResponseSchema,
    StructuredOutputParser,
    PydanticOutputParser,
)

from pydantic import BaseModel, Field
from typing import List, Optional


class WikidataGraphRAG:
    def __init__(
        self,
        hf_token: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cpu",
        local: str = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local = local
        self.sparqlwd = SPARQLWrapper(
            "https://query.wikidata.org/sparql",
            agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        )
        if self.local:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, token=hf_token
            )
        else:
            self.tokenizer = None
            self.model = None

    # https://www.jcchouinard.com/wikidata-api-python/
    def _fetch_wikidata(self, params: dict[str, str]) -> any:
        url = "https://www.wikidata.org/w/api.php"
        try:
            return requests.get(url, params=params)
        except:
            return "There was and error"

    def _get_wikidata_entities(self, entity: str, lang: str = "en") -> str:
        # Which parameters to use
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "search": entity,
            "language": lang,
        }

        # Fetch API
        data = self._fetch_wikidata(params)

        # show response as JSON
        data = data.json()
        return [
            {
                "id": item["id"],
                "label": item["label"],
                "description": item.get("description", ""),
            }
            for item in data["search"][:5]
        ]

    def execute_sparql_to_wikidata(self, q: str):
        self.sparqlwd.setQuery(q)
        self.sparqlwd.setReturnFormat(JSON)
        try:
            results = self.sparqlwd.query().convert()
            results_cleaned = []
            for result in results["results"]["bindings"]:
                tmp = dict()
                for header in results["head"]["vars"]:
                    tmp[header] = result[header]["value"]
                results_cleaned.append(tmp)
            return results_cleaned
        except Exception as e:
            print(e)
            return None

    def _extract_sparql_query(self, text):
        # Regex to match SPARQL query in the string
        regex = r"```sparql(.*?)```"

        # Search for the pattern in the text
        match = re.search(regex, text, re.DOTALL)

        # Return the matched query or None if not found
        return match.group(1).strip() if match else None

    def extract_entity(
        self, question: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    ) -> list[str]:
        template = """## INSTRUCTIONS
- Extract the entities from the given question!
- These entities usage is to find the most appropriate entity ID from wikidata to be used in SPARQL queries.
- If there is no entity in the question, return empty list.
- ONLY return the entities. DO NOT return anything else.
- DO NOT include adjectives like 'Highest', 'Lowest', 'Biggest', etc in the entity.
- DO NOT provide any extra information, for instance explanation inside a brackets like '(population)', '(area)', '(place)', '(artist)', etc
- DO NOT include any explanations or apologies in your responses.
- Remove all stop words, including conjunctions like 'and' and prepositions like 'in' and 'on' from the extracted entity.
- Make the entity singular, not plural. For instance, if the entity is foods, then transform it into food.

## OUTPUT FORMAT INSTRUCTIONS
{format_instructions}

## EXAMPLES
- Question: how much is 1 tablespoon of water?
Entity: ```json{{"entities": ["Tablespoon"]}}```

- Question: how are glacier caves formed?
Entity: ```json{{"entities": ["Glacier cave"]}}```

- Question: how much are the harry potter movies worth?
Entity: ```json{{"entities": ["Harry Potter"]}}```

- Question: how big is auburndale florida?
Entity: ```json{{"entities": ["Auburndale", "Florida"]}}```

- Question: what country is jakarta in?
Entity: ```json{{"entities": ["Jakarta"]}}```

- Question: how deep can be drill for deep underwater?
Entity: ```json{{"entities": ["Deepwater drilling"]}}```

- Question: how many continents there are in indonesia?
Entity: ```json{{"entities": ["Continent", "Indonesia"]}}```

- Question: how fast is it?
Entity: ```json{{"entities": []}}```

- Question: Largest cities of the world
Entity: ```json{{"entities": ["city"]}}```

- Question: Popular surnames among fictional characters
Entity: ```json{{"entities": ["Fictional character"]}}```

- Question: WWII battle durations
Entity: ```json{{"entities": ["WWII", "battle"]}}```

## QUESTION
- Question: {question}
Entity: """

        response_schemas = [
            ResponseSchema(
                name="entities",
                description="entities extracted from user's question",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )

        model_kwargs = {"device": self.device}
        if self.local:
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            llm = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response = llm_chain.run(question=question).split("Entity: ")[-1]
        return output_parser.parse(response).get("entities", [])

    def get_entity_ids(
        self,
        question: str,
        entities: list[str],
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ) -> list[dict[str, str]]:
        retrieved_wikidata_matched_entities = dict()
        for entity in entities:
            retrieved_wikidata_matched_entities[entity] = self._get_wikidata_entities(
                entity
            )

        template = """## INSTRUCTIONS
- For each entity given, find the most appropriate entity ID from the list of wikidata entities given to be used in SPARQL queries to answer the given question!
- You MUST return the ID and label of all the entity in the list of entities given in "Entities". For example: if the entities give is "A" and "B", then you have to return the ID and label of "A" and "B".
- ONLY return ONE set of entity ID for each entity, DO NOT hallucinate and pick more than one entity ID sets. For example: there are [C, D, E] for "A" and there are [F, G, H] for "B", ONLY return ONE for each entity, such as C for "A" and F for "B".
- ONLY return the entity IDs, labels, and descriptions from the list of wikidata entities given. DO NOT return anything else and DO NOT hallucinate.
- DO NOT include any explanations or apologies in your responses.

## OUTPUT FORMAT INSTRUCTIONS
{format_instructions}

## EXAMPLES
- Question: Humans born in New York City
Entities: ["New York City", "Human"]
Wikidata Entities: ```json{{"New York City": [{{"id": "Q60",
"label": "New York City",
"description": "most populous city in the United States"}},
{{"id": "Q99673783",
"label": "New York City",
"description": "New York City as depicted in Star Trek"}},
{{"id": "Q7013127", "label": "New York City", "description": "band"}},
{{"id": "Q111668100",
"label": "New York City",
"description": "Song by Tee Cloud"}},
{{"id": "Q114518687",
"label": "New York City",
"description": "episode of Drinking Made Easy (S1 E10)"}}],
"Human": [{{"id": "Q5",
"label": "human",
"description": "any member of Homo sapiens, unique extant species of the genus Homo, from embryo to adult"}},
{{"id": "Q15978631",
"label": "Homo sapiens",
"description": "species of mammal"}},
{{"id": "Q67372736",
"label": "personal",
"description": "grammatical gender"}},
{{"id": "Q73755406",
"label": "human",
"description": "human species as depicted in the Teenage Mutant Ninja Turtles universe"}},
{{"id": "Q2408214",
"label": "Human Entertainment",
"description": "Japanese video game developer and publisher"}}]}}```
Entity IDs: ```json{{
        "ids": [
            {{"id": "Q60", "label": "New York City", "description": "most populous city in the United States"}},
            {{"id": "Q5", "label": "human", "description": "any member of Homo sapiens, unique extant species of the genus Homo, from embryo to adult"}},
        ]
    }}```

- Question: Popular surnames among fictional characters
Entities: ["Fictional character"]
Wikidata Entities: ```json{{"Fictional character": [{{"id": "Q95074",
"label": "fictional character",
"description": "fictional human or non-human character in a narrative work of art"}},
{{"id": "Q14514600",
"label": "group of fictional characters",
"description": "set of fictional characters"}},
{{"id": "Q65924737",
"label": "character in a fictitious work",
"description": "fictional character that is considered fictional even in a fictional story"}},
{{"id": "Q27960097",
"label": "character poster",
"description": "advertisement poster focusing on a fictional character of a work"}},
{{"id": "Q100708514",
"label": "fictional character in a musical work",
"description": "fictional character only appearing in musical works"}}]}}```
Entity IDs: ```json{{
        "ids": [
            {{"id": "Q95074", "label": "fictional character", "description": "fictional human or non-human character in a narrative work of art"}}
        ]
    }}```

- Question: WWII battle durations
Entities: ["WWII", "battle"]
Wikidata Entities: ```json{{"WWII": [{{"id": "Q362",
"label": "World War II",
"description": "1939–1945 global conflict"}},
{{"id": "Q1470020",
"label": "National World War II Memorial",
"description": "war memorial in Washington, D.C., United States"}},
{{"id": "Q7957296",
"label": "WWII",
"description": "1982 studio album by Waylon Jennings and Willie Nelson"}},
{{"id": "Q444116",
"label": "WWII Axis collaboration in France",
"description": "policy stance"}},
{{"id": "Q327039",
"label": "timeline of World War II",
"description": "list of significant events occurring during World War II"}}],
"battle": [{{"id": "Q178561",
"label": "battle",
"description": "part of a war which is well defined in duration, area and force commitment"}},
{{"id": "Q737593",
"label": "Battle",
"description": "town and civil parish in the local government district of Rother in East Sussex, England"}},
{{"id": "Q1330167",
"label": "Fairey Battle",
"description": "light bomber family by Fairey"}},
{{"id": "Q16479866", "label": "Battle", "description": "family name"}},
{{"id": "Q105826326",
"label": "battle",
"description": "act of struggling to achieve or fight against something"}}]}}```
Entity IDs: ```json{{
        "ids": [
            {{"id": "Q362", "label": "World War II", "description": "1939–1945 global conflict"}},
            {{"id": "Q178561", "label": "battle", "description": "part of a war which is well defined in duration, area and force commitment"}},
        ]
    }}```

## QUESTION
- Question: {question}
Entities: {entities}
Wikidata Entities: ```json{retrieved_wikidata_matched_entities}```
Entity IDs: """

        class EntityIdItem(BaseModel):
            id: str = Field(description="id of the entity")
            label: str = Field(description="label of the entity")
            description: Optional[str] = Field(
                None, description="description of the entity"
            )

        class EntityIds(BaseModel):
            ids: List[EntityIdItem]

        output_parser = PydanticOutputParser(pydantic_object=EntityIds)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "question",
                "entities",
                "retrieved_wikidata_matched_entities",
            ],
            partial_variables={"format_instructions": format_instructions},
        )

        model_kwargs = {"device": self.device, "max_new_tokens": 1000}
        if self.local:
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            llm = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response = llm_chain.run(
            question=question,
            entities=entities,
            retrieved_wikidata_matched_entities=retrieved_wikidata_matched_entities,
        ).split("Entity IDs: ")[-1]

        try:
            parsed_response = output_parser.parse(response)
            return parsed_response.dict().get("ids", [])
        except:
            response = response.replace("'", '"')
            parsed_response = output_parser.parse(response)
            return parsed_response.dict().get("ids", [])

    def generate_sparql(
        self,
        question: str,
        entity_ids: list[dict[str, str]],
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        verbose: bool = False,
    ) -> list[dict[str, str]]:
        template = """## INSTRUCTIONS
- Generate SPARQL queries to answer the given question!
- To generate the SPARQL, you can utilize the information from the given Entity IDs. You do not have to use it, but if it can help you to determine the ID of the entity, you can use it.
- You will also be provided with the 100 most used properties with its ID. You are only able to generate SPARQL query from these properties. If it requires property that is not provided, then generate empty query like ```sparql```.
- You can also determine the IDs of the entites that aren't provided with your knowledge.
- Generate the SPARQL with chain of thoughts.
- DO NOT include any apologies in your responses.
- ONLY generate the Thoughts and SPARQL query once! DO NOT try to generate the Question!
- When using a property such as P17 (country), you DO NOT need to verify explicitly whether it is Q6256 entity (country).
- DO NOT use LIMIT, ORDER BY, FILTER in the SPARQL query when not explicitly asked in the question!
- DO NOT aggregation function like COUNT, AVG, etc in the SPARQL query when not asked in the question!
- Always use 'en' language for labels as default unless explicitly asked to use another language.
- Be sure to generate a SPARQL query that is valid and return all the asked information in the question.
- Make the query as simple as possible!
- DO NOT hallucinate the thoughts and query!

## CONTEXT
- entity IDs: ```{entity_ids}```
- 100 most used properties with its ID:
```json
[
  {{
    "label": "cites work",
    "id": "P2860",
    "description": "citation from one creative or scholarly work to another",
    "aliases": "bibliographic citation, citation"
  }},
  {{
    "label": "series ordinal",
    "id": "P1545",
    "description": "position of an item in its parent series (most frequently a 1-based index), generally to be used as a qualifier (different from \"rank\" defined as a class, and from \"ranking\" defined as a property for evaluating a quality).",
    "aliases": "#, index, number, rank, n\u00b0, n\u00ba, \u2116, num., ordinal, ordinal number, position in series, section number, series number, sort order, sorting order, unit number"
  }},
  {{
    "label": "author name string",
    "id": "P2093",
    "description": "stores unspecified author or editor name for publications; use if Wikidata item for author (P50) or editor (P98) does not exist or is not known. Do not use both.",
    "aliases": "byline, author string, creator name string, editor name string, maker name string, short author name, songwriting credits string"
  }},
  {{
    "label": "instance of",
    "id": "P31",
    "description": "that class of which this subject is a particular example and member; different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain (and an instance of volcanic landform)",
    "aliases": "example of, is an, rdf:type, unique individual of, unitary element of class, \u2208, has type, is of type, item of type, type, is a"
  }},
  {{
    "label": "retrieved",
    "id": "P813",
    "description": "date or point in time that information was retrieved from a database or website (for use in online sources)",
    "aliases": "reference date, access date, accessdate, accessed, consulted, date accessed, date of access, date of retrieval, date retrieved, retrieved at, retrieved date, time of retrieval, viewed on, viewed on date, visited at"
  }},
  {{
    "label": "stated in",
    "id": "P248",
    "description": "to be used in the references field to refer to the information document or database in which a claim is made; for qualifiers use P805; for the type of document in which a claim is made use P3865",
    "aliases": "cited by, cited in, cited from, cited at, cited on, stated on, in statement, originating source, source of claim, stated at"
  }},
  {{
    "label": "reference URL",
    "id": "P854",
    "description": "should be used for Internet URLs as references. Use \"Wikimedia import URL\" (P4656) for imports from WMF sites",
    "aliases": "stated at URL, ref URL, reference Uniform Resource Locator, source URL, url for reference, webref, website, URL, ref"
  }},
  {{
    "label": "PubMed publication ID",
    "id": "P698",
    "description": "identifier\u00a0for journal articles/abstracts in PubMed",
    "aliases": "PMID, PubMed ID, PubMed reference number"
  }},
  {{
    "label": "title",
    "id": "P1476",
    "description": "published name of a work, such as a newspaper article, a literary work, piece of music, a website, or a performance work",
    "aliases": "article, name, full title, headline, known as, original title, titled"
  }},
  {{
    "label": "publication date",
    "id": "P577",
    "description": "date or point in time when a work was first published or released",
    "aliases": "drop date, air date, airdate, be published during, be published in, broadcast date, date of first publication, date of release, date published, first publication, first published, first released, pubdate, publication time, time of publication, was published during, was published in, year of publication, issued, publication, dop, published, initial release, released in, date released, released, release date, launch date, launched, date of publication"
  }},
  {{
    "label": "published in",
    "id": "P1433",
    "description": "larger work that a given work was published in, like a book, journal or music album",
    "aliases": "volume of the book, tome of, tome of the book, volume of, album, work, venue, music album, part of work, article of, chapter of, essay of, on the tracklist of, published in journal, song on, song on album, track of, track on, track on album"
  }},
  {{
    "label": "page(s)",
    "id": "P304",
    "description": "page number of source referenced for statement. Note \"column(s)\" (P3903) and \"folio(s)\" (P7416) for other numbering systems",
    "aliases": "page cited, page referenced, page, p., pages, page number, page numbers, pg., pgs., pp."
  }},
  {{
    "label": "volume",
    "id": "P478",
    "description": "volume of a book or music release in a collection/series or a published collection of journal issues in a serial publication",
    "aliases": "number of series, tome, volume of a book, part number, vol., volume of serial, numbering of part, series numbering, number of part, numbering in series, numbering of volume, volume number"
  }},
  {{
    "label": "issue",
    "id": "P433",
    "description": "issue of a newspaper, a scientific journal or magazine for reference purpose",
    "aliases": "no., number, issue number"
  }},
  {{
    "label": "DOI",
    "id": "P356",
    "description": "serial code used to uniquely identify digital objects like academic papers (use upper case letters only)",
    "aliases": "Digital Object Identifier, doi"
  }},
  {{
    "label": "apparent magnitude",
    "id": "P1215",
    "description": "measurement of the brightness of an astronomic object, as seen from the Earth",
    "aliases": ""
  }},
  {{
    "label": "astronomical filter",
    "id": "P1227",
    "description": "passband used to isolate particular sections of the electromagnetic s|pectrum",
    "aliases": "filter, colour filter, wavelength band"
  }},
  {{
    "label": "author",
    "id": "P50",
    "description": "main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist",
    "aliases": "maker, writer, poet, playwright, creator, written by"
  }},
  {{
    "label": "main subject",
    "id": "P921",
    "description": "primary topic of a work (see also P180: depicts)",
    "aliases": "descriptor (wd), main keyword, primary keyword, /common/topic/subject, content deals with, content describes, content is about, describes, has a subject, has topic, in regards to, is about, main issue, main theme, main thing, mainly about, plot keyword, primary subject, primary topic, regards, topic of work, subject, artistic theme, sitter, regarding, topic, theme, about, keyword, main topic, index term, subject heading"
  }},
  {{
    "label": "catalog code",
    "id": "P528",
    "description": "catalog name of an object, use with qualifier P972",
    "aliases": "register, astronomical catalog, registration number, cat. no., catalog number, cat#, catalogue code, catalogue number, identifier in catalog, enrollment number"
  }},
  {{
    "label": "language of work or name",
    "id": "P407",
    "description": "language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name (for persons use \"native language\" (P103) and \"languages spoken, written or signed\" (P1412))",
    "aliases": "language, audio language, available in, broadcasting language, language of name, language of spoken text, language of the name, language of the reference, language of URL, language of website, language of work, named in language, used language"
  }},
  {{
    "label": "catalog",
    "id": "P972",
    "description": "catalog for the item, or, as a qualifier of P528 \u2013 catalog for which the 'catalog code' is valid",
    "aliases": "exhibition catalogue, catalogue, registry"
  }},
  {{
    "label": "object named as",
    "id": "P1932",
    "description": "use as qualifier to indicate how the object's value was given in the source",
    "aliases": "location name string, object name string, place name string, author named as, sic, stated as, as, [sic], credited as, named as, object named as, object stated as, object value stated as, original wording, originally printed as, printed as, reference wording, source wording, stated author, value stated as"
  }},
  {{
    "label": "country",
    "id": "P17",
    "description": "sovereign state that this item is in (not to be used for human beings)",
    "aliases": "host country, state, land, sovereign state"
  }},
  {{
    "label": "point in time",
    "id": "P585",
    "description": "date something took place, existed or a statement was true; for providing time use the \"refine date\" property (P4241)",
    "aliases": "point time, during, as of, at time, by date, occurred on, time of event, year, date, time, on, when"
  }},
  {{
    "label": "located in the administrative territorial entity",
    "id": "P131",
    "description": "the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity.",
    "aliases": "administrative territory, happens in, in, in administrative unit, in the administrative unit, Indian reservation, is in administrative unit, is in the administrative region of, is in the administrative unit, is in the arrondissement of, is in the borough of, is in the city of, is in the commune of, is in the county of, is in the department of, is in the district of, is in the Indian reservation of, is in the Indian reserve of, is in the local government area of, is in the municipality of, is in the parish of, is in the prefecture of, is in the principal area of, is in the province of, is in the region of, is in the rural city of, is in the settlement of, is in the shire of, is in the state of, is in the territory of, is in the town of, is in the village of, is in the voivodeship of, is in the ward of, is located in, located in administrative unit, located in the administrative unit, located in the territorial entity, location (administrative territorial entity), town, city, locality, region, administrative territorial entity, state, territory"
  }},
  {{
    "label": "start time",
    "id": "P580",
    "description": "time an entity begins to exist or a statement starts being valid",
    "aliases": "introduction, from, began, beginning, building date, from date, from time, introduced, join date, join time, joined, since, start date, started in, starting, starttime"
  }},
  {{
    "label": "PMC publication ID",
    "id": "P932",
    "description": "identifier for a scientific work issued by PubMed Central (without \"PMC\" prefix)",
    "aliases": "PMC ID, PMCID, PubMed Central ID, PubMed Central reference number"
  }},
  {{
    "label": "determination method",
    "id": "P459",
    "description": "how a value is determined, or the standard by which it is declared",
    "aliases": "standard, method, methodology, rationale, justification, determined by, measured by, method of determination"
  }},
  {{
    "label": "occupation",
    "id": "P106",
    "description": "occupation of a person; see also \"field of work\" (Property:P101), \"position held\" (Property:P39)",
    "aliases": "avocation, career, employ, employment, work, profession, job, craft, vocation"
  }},
  {{
    "label": "coordinate location",
    "id": "P625",
    "description": "geocoordinates of the subject. For Earth, please note that only WGS84 coordinating system is supported at the moment",
    "aliases": "wgs84, position, location, geographic coordinates, latitude, co-ordinate location, co-ordinates, co-ords, coordinates, coords, geo, geocoordinates, geographic coordinate, geographical coordinates, geolocation, geotag, gps, gps co-ordinate, gps co-ordinates, gps coordinate, gps coordinates, gps location, location on earth, location on map, longitude, point on a map, point on earth, point on the globe, wgs 84, wgs-84"
  }},
  {{
    "label": "sex or gender",
    "id": "P21",
    "description": "sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)",
    "aliases": "gender or sex, sex, gender, gender expression, biological sex, gender identity"
  }},
  {{
    "label": "subject named as",
    "id": "P1810",
    "description": "name by which a subject is recorded in a database, mentioned as a contributor of a work, or is referred to in a particular context",
    "aliases": "AAP, stated as, authorized access point, credited as, named as, authorized as, authorized form, authorized heading, entity named as, established as, established form, established heading, spelled as, subject stated as"
  }},
  {{
    "label": "epoch",
    "id": "P6259",
    "description": "epoch of an astronomical object coordinate",
    "aliases": ""
  }},
  {{
    "label": "based on heuristic",
    "id": "P887",
    "description": "indicates that the property value is determined based on some heuristic (Q201413); to be used as source",
    "aliases": "heuristic, heuristically based on, heuristic used, inferred from heuristic"
  }},
  {{
    "label": "VIAF ID",
    "id": "P214",
    "description": "identifier for the Virtual International Authority File database [format: up to 22 digits]",
    "aliases": "VIAF cluster ID, VIAF, Virtual International Authority File, VIAF identifier, viaf.org ID"
  }},
  {{
    "label": "right ascension",
    "id": "P6257",
    "description": "astronomical equivalent of longitude",
    "aliases": "Longitude of Ascending Node"
  }},
  {{
    "label": "declination",
    "id": "P6258",
    "description": "astronomical equivalent of latitude",
    "aliases": ""
  }},
  {{
    "label": "SIMBAD ID",
    "id": "P3083",
    "description": "identifier for an astronomical object, in the University of Strasbourg's SIMBAD database",
    "aliases": "SIMBAD"
  }},
  {{
    "label": "given name",
    "id": "P735",
    "description": "first name or another given name of this person; values used with the property should not link disambiguations nor family names",
    "aliases": "name, Christian name, first name, forename, middle name, personal name"
  }},
  {{
    "label": "Google Knowledge Graph ID",
    "id": "P2671",
    "description": "identifier for Google Knowledge Graph API, starting with \"/g/\". For IDs starting with \"/m/\", use Freebase ID (P646)",
    "aliases": "Google knowledge panel, Google Knowledge Panel ID, MREID"
  }},
  {{
    "label": "constellation",
    "id": "P59",
    "description": "the area of the celestial sphere of which the subject is a part (from a scientific standpoint, not an astrological one)",
    "aliases": "part of constellation"
  }},
  {{
    "label": "found in taxon",
    "id": "P703",
    "description": "the taxon in which the item can be found",
    "aliases": "found in species, present in taxon"
  }},
  {{
    "label": "date of birth",
    "id": "P569",
    "description": "date on which the subject was born",
    "aliases": "born, DOB, birth date, birth year, birthdate, birthyear, born on, year of birth"
  }},
  {{
    "label": "Wikimedia import URL",
    "id": "P4656",
    "description": "URL of source to indicate the page or revision of an import source from another Wikimedia project (except actual references, such as Wikisource source texts). Use instead of \"reference URL\" (P854). Permalinks are preferred.",
    "aliases": "import URL from a Wikimedia project, reference URL from Wikimedia, reference URL from Wikipedia, URL imported from Wikimedia, Wikimedia permalink, Wikimedia reference URL, Wikimedia URL, Wikipedia import URL, Wikipedia reference URL, Wikipedia URL, WMF import URL, WMF URL, Wikibooks import URL, Wikidata import URL, Wikifunctions import URL, Wikimedia Commons import URL, Wikinews import URL, Wikiquote import URL, Wikisource import URL, Wikispecies import URL, Wikiversity import URL, Wikivoyage import URL, Wiktionary import URL"
  }},
  {{
    "label": "end time",
    "id": "P582",
    "description": "moment when an entity ceases to exist or a statement stops being valid",
    "aliases": "to, ending, cease date, cease operation, cease time, closed, completed in, dissolved, divorced, end date, enddate, ends, endtime, fall date, left office, stop time, till, until"
  }},
  {{
    "label": "imported from Wikimedia project",
    "id": "P143",
    "description": "source of this claim's value; used in references section by bots or humans importing data from Wikimedia projects",
    "aliases": "imported from wiki, Wikimedia project, Wikipedia project, from Wikimedia project, import from Wikimedia project, import from WM project, imported from, imported from Wikipedia project, imported from WM project"
  }},
  {{
    "label": "parallax",
    "id": "P2214",
    "description": "parallax of nearest stars",
    "aliases": "astronomical parallax"
  }},
  {{
    "label": "full work available at URL",
    "id": "P953",
    "description": "URL of a web page containing the full body of this item",
    "aliases": "available at URL, ebook, full text available at URL, transcription, online version, archived at, article at, available at, document at, ebook URL, full text at, full text available at, full text url, fulltext, online mirror, paper at, text at, text mirror url, URL for full text"
  }},
  {{
    "label": "Commons category",
    "id": "P373",
    "description": "name of the Wikimedia Commons category containing files related to this item (without the prefix \"Category:\")",
    "aliases": "Wikimedia Commons category, category Commons, category on Commons, category on Wikimedia Commons, category Wikipedia Commons, Commons cat, commonscat, Wikimedia Commons cat"
  }},
  {{
    "label": "image",
    "id": "P18",
    "description": "image of relevant illustration of the subject; if available, also use more specific properties (sample: coat of arms image, locator map, flag image, signature image, logo image, collage image)",
    "aliases": "camera picture, still capture, Commons image, image capture, image of exterior, image of subject, Image on Commons, Image on Wikimedia Commons, img, infobox image, graph, illustration, drawing, screen capture, screenshot, still, photograph, portrait, picture, graphic, photo, File:"
  }},
  {{
    "label": "country of citizenship",
    "id": "P27",
    "description": "the object is a country that recognizes the subject as its citizen",
    "aliases": "citizenship, (legal) nationality, citizen of, nation of citizenship, national of, subject of (country)"
  }},
  {{
    "label": "right ascension component of proper motion",
    "id": "P10752",
    "description": "the value \u03bc\u1d45 as a component of the proper motion of a star",
    "aliases": ""
  }},
  {{
    "label": "declination component of proper motion",
    "id": "P10751",
    "description": "the value \u03bc\u1d5f as a component of the proper motion of a star",
    "aliases": ""
  }},
  {{
    "label": "family name",
    "id": "P734",
    "description": "part of full name of person",
    "aliases": "surname, last name"
  }},
  {{
    "label": "CNKI CJFD journal article ID",
    "id": "P6769",
    "description": "identifier for journal articles in China National Knowledge Infrastructure (CNKI)'s China Journal Full-text Database (CJFD)",
    "aliases": "CJFD ID, CJFD journal article ID, CNKI CJFD journal article ID, CNKI journal article ID"
  }},
  {{
    "label": "part of",
    "id": "P361",
    "description": "object of which the subject is a part (if this subject is already part of object A which is a part of object B, then please only make the subject part of object A), inverse property of \"has part\" (P527, see also \"has parts of the class\" (P2670))",
    "aliases": "assembly of, branch of, cadet branch of, collateral branch of, component of, element of, is part of, meronym of, part of-property, section of, subassembly of, subgroup of, subsystem of, system of, in, chain, contained in, contained within, within"
  }},
  {{
    "label": "subclass of",
    "id": "P279",
    "description": "this item is a subclass (subset) of that item; all instances of this item are instances of that item; different from P31 (instance of), e.g.: K2 is an instance of mountain; volcano is a subclass of mountain (and an instance of volcanic landform)",
    "aliases": "variant of, hyperonym, all members of this class also belong to, category of, form of, has superclass, hyponym of, is a class of, is a type of, is necessarily also a, is thereby also a, rdfs:subClassOf, subcategory of, subtype of, type of, way of, whose instances are a subset of those of, whose instances are among, kind of, sort of, sc (abbreviation)"
  }},
  {{
    "label": "radial velocity",
    "id": "P2216",
    "description": "component of the object's velocity that points in the direction of the radius connecting the object and the point",
    "aliases": ""
  }},
  {{
    "label": "ResearchGate publication ID",
    "id": "P5875",
    "description": "identifier of a publication in ResearchGate",
    "aliases": "ResearchGate ID, ResearchGate publication identifier"
  }},
  {{
    "label": "collection",
    "id": "P195",
    "description": "art, museum, archival, or bibliographic collection the subject is part of",
    "aliases": "archives, book series, art collection, editorial collection, series, museum collection, book collection, repository, GLAM, archival holdings, bibliographic collection"
  }},
  {{
    "label": "Freebase ID",
    "id": "P646",
    "description": "identifier for a page in the Freebase database. Format: \"/m/0\" followed by 2 to 7 characters. For IDs starting with \"/g/\", use Google Knowledge Graph ID (P2671)",
    "aliases": "Freebase identifier, MID, FID, Freebase"
  }},
  {{
    "label": "ortholog",
    "id": "P684",
    "description": "orthologous gene in another species (use with 'species' qualifier)",
    "aliases": ""
  }},
  {{
    "label": "copyright status",
    "id": "P6216",
    "description": "copyright status for intellectual creations like works of art, publications, software, etc.",
    "aliases": "copyright restriction"
  }},
  {{
    "label": "GeoNames ID",
    "id": "P1566",
    "description": "identifier in the GeoNames geographical database",
    "aliases": "GeoNames, Geonames Id"
  }},
  {{
    "label": "parent taxon",
    "id": "P171",
    "description": "closest parent taxon of the taxon in question",
    "aliases": "higher taxon, taxon parent"
  }},
  {{
    "label": "taxon name",
    "id": "P225",
    "description": "correct scientific name of a taxon (according to the reference given)",
    "aliases": "correct name (ICNafp), Latin name of a taxon (deprecated), scientific name of a taxon, taxonomic name, valid name (ICZN)"
  }},
  {{
    "label": "taxon rank",
    "id": "P105",
    "description": "level in a taxonomic hierarchy",
    "aliases": "taxonomic rank, rank, type of taxon"
  }},
  {{
    "label": "population",
    "id": "P1082",
    "description": "number of people inhabiting the place; number of people of subject",
    "aliases": "human population, inhabitants"
  }},
  {{
    "label": "location",
    "id": "P276",
    "description": "location of the object, structure or event. In the case of an administrative entity as containing item use P131. For statistical entities use P8138. In the case of a geographic entity use P706. Use P7153 for locations associated with the object",
    "aliases": "whereabouts, held by, neighborhood, locality, region, venue, place, based in, in, from, suburb, neighbourhood, locale, is in, located, located in, location of item, moveable object location, place held"
  }},
  {{
    "label": "place of birth",
    "id": "P19",
    "description": "most specific known birth location of a person/human, animal or fictional character",
    "aliases": "born, born at, born in, location born, birth location, birth place, birthplace, location of birth, POB, birth city"
  }},
  {{
    "label": "distance from Earth",
    "id": "P2583",
    "description": "estimated distance to astronomical objects",
    "aliases": "angular diameter distance, comoving distance, light-travel distance, light-year distance, luminosity distance, parsec distance, proper distance"
  }},
  {{
    "label": "GNS Unique Feature ID",
    "id": "P2326",
    "description": "identifier for geographic entities according to the National Geospatial-Intelligence Agency's GEOnet Names Server",
    "aliases": "GNS UFI, Unique Feature Identifier, GEOnet Names Server Unique Feature, GNS ID, GNS Unique Feature Identifier, Geographic Names Server Unique Feature"
  }},
  {{
    "label": "date of death",
    "id": "P570",
    "description": "date on which the subject died",
    "aliases": "DOD, died, dead, death, death date, died on, year of death, date of the end, deathdate"
  }},
  {{
    "label": "chromosome",
    "id": "P1057",
    "description": "chromosome on which an entity is localized",
    "aliases": "on chromosome, present in chromosome"
  }},
  {{
    "label": "inception",
    "id": "P571",
    "description": "time when an entity begins to exist; for date of official opening use P1619",
    "aliases": "time of inception, written on date, year commenced, year created, year established, year founded, year of creation, year of inception, year written, broke ground, built, commenced on date, commencement date, composed, constructed, construction date, created in, created in year, created on, created on date, creation date, creation year, date commenced, date constructed, date created, date first created, date formed, date founded, date of commencement, date of creation, date of foundation, date of foundation or creation, date of inception, date of origin, day of inception, dedication date, established on date, first created on, formation date, formed at, formed in year, formed on, formed on date, foundation\u202f/\u2009creation date, foundation date, founded, founded on, founded on date, inception date, initiated, instantiation date, made in year, made on, time of foundation or creation, foundation, formation, introduction, formed in, founded in, established, date of establishment, date of founding, establishment date, founding date, introduced, start date, created, first issue, completed, initial release date, launch date, starting date, installated in, installated on, installation date"
  }},
  {{
    "label": "educated at",
    "id": "P69",
    "description": "educational institution attended by subject",
    "aliases": "student at, University attended, alumna of, alumni of, alumnus of, attended school at, college attended, education place, graduate of, graduated from, place of education, school attended, schooled at, schooling place, studied at, went to school at, faculty, education, student of, alma mater, attended"
  }},
  {{
    "label": "GBIF taxon ID",
    "id": "P846",
    "description": "taxon identifier in GBIF",
    "aliases": "GBIF ID, Global Biodiversity Information Facility ID, taxonKey"
  }},
  {{
    "label": "category combines topics",
    "id": "P971",
    "description": "this category combines (intersects) these two or more topics",
    "aliases": "category intersects topics, cct, combines topics, subjects of category, topics of category"
  }},
  {{
    "label": "described at URL",
    "id": "P973",
    "description": "item is described at the following URL",
    "aliases": "about page, d@url, described at webpage, described by URL, described in url, described in URL, described on webpage, descURL, discussed at URL, info URL, rdfs:isDefinedBy, URL described in, web page describing, mentioned at URL, described at Uniform Resource Locator, described at website"
  }},
  {{
    "label": "matched by identifier from",
    "id": "P11797",
    "description": "this external ID was added because the following link was presented in both the external database and the Wikidata item",
    "aliases": "based on matching link, based on matching value, matched by ID from"
  }},
  {{
    "label": "Entrez Gene ID",
    "id": "P351",
    "description": "identifier for a gene per the NCBI Entrez database",
    "aliases": ""
  }},
  {{
    "label": "Elo rating",
    "id": "P1087",
    "description": "quantitative measure of one's game-playing ability, particularly in classical chess",
    "aliases": ""
  }},
  {{
    "label": "GND ID",
    "id": "P227",
    "description": "identifier from an international authority file of names, subjects, and organizations (please don't use type n = name, disambiguation) - Deutsche Nationalbibliothek",
    "aliases": "IndividualisedGnd, GND, Gemeinsame Normdatei ID, GND ID, GND identifier, Integrated Authority File, Universal Authority File, Deutschen Nationalbibliothek ID, DNB author ID, DNB authorities, German National Library ID, GND-IDN"
  }},
  {{
    "label": "languages spoken, written or signed",
    "id": "P1412",
    "description": "language(s) that a person or a people speaks, writes or signs, including the native language(s)",
    "aliases": "language, used language, language used, uses language, language spoken, language written, language of expression, language read, language signed, language(s) spoken, written or signed, languages of expression, languages signed, languages spoken, languages spoken, written, or signed, signed language, signs language, speaks language, spoke language, writes language, wrote language"
  }},
  {{
    "label": "InChIKey",
    "id": "P235",
    "description": "a hashed version of the full standard InChI",
    "aliases": ""
  }},
  {{
    "label": "short name",
    "id": "P1813",
    "description": "short name of a place, organisation, person, journal, wikidata property, etc. Used by some Wikipedia templates.",
    "aliases": "abbreviated name, brief name, short title, shortname, acronym, abbreviation, abbrev, shortened name, initialism"
  }},
  {{
    "label": "Library of Congress authority ID",
    "id": "P244",
    "description": "Library of Congress name authority (persons, families, corporate bodies, events, places, works and expressions) and subject authority identifier [Format: 1-2 specific letters followed by 8-10 digits (see regex). For manifestations, use P1144]",
    "aliases": "LCCN, Authority LCCN, LC Control Number, LC Name Authority File ID, LC Name Authority ID, LC Name ID, LC Subject Heading ID, LC Subject ID, LC/NACO Authority File ID, LC/NACO LCCN, LC/NACO/SACO Authority File ID, LCAuth ID, LCAuth identifier, LCCN (LCSH), LCCN (NACO), LCCN (NAF), LCCN (Name Authority), LCCN (SACO), LCCN (Subject Authority), LCNACO, LCNAF ID, LCSH Authority File ID, LCSH ID, LCSH LCCN, Library of Congress Control Number (Authorities), Library of Congress Control Number (Name Authority), Library of Congress Control Number (Subject Authority), Library of Congress Name Authority File, Library of Congress Subject Headings, Library of Congress Subject Headings ID, LoC Authorities ID, LOC ID, LoC identifier, NACO ID, NACO LCCN, NAF LCCN, SACO ID"
  }},
  {{
    "label": "filename in archive",
    "id": "P7793",
    "description": "qualifier and referencing property to identify the specific file within a compressed archive that is relevant to a statement",
    "aliases": ""
  }},
  {{
    "label": "applies to jurisdiction",
    "id": "P1001",
    "description": "the item (institution, law, public office, public register...) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, ...)",
    "aliases": "jurisdiction linked, applies to place, jurisdiction, applicable area, applicable country, applicable geographic place, applicable jurisdiction, applicable location, applicable place, applicable territorial jurisdiction, applied to jurisdiction, applies to area, applies to geographic area, applies to geographic place, applies to region, applies to territorial jurisdiction, belongs to jurisdiction, country of jurisdiction, of jurisdiction, valid in jurisdiction"
  }},
  {{
    "label": "UniProt protein ID",
    "id": "P352",
    "description": "identifier for a protein per the UniProt database",
    "aliases": "UniProtKB accession number"
  }},
  {{
    "label": "heritage designation",
    "id": "P1435",
    "description": "heritage designation of a cultural or natural site",
    "aliases": "designation, protection, heritage status, heritage designation, legal protection, listed status, listing, protected status"
  }},
  {{
    "label": "copyright license",
    "id": "P275",
    "description": "license under which this copyrighted work is released",
    "aliases": "software license, license, licence, content licence, content license, copyright licence"
  }},
  {{
    "label": "follows",
    "id": "P155",
    "description": "immediately prior item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has replaced the preceding item, e.g. political offices, use \"replaces\" (P1365)]",
    "aliases": "split from, predecessor, preceded by, succeeds, before was, comes after, is after, prequel is, prev, previous element, previous is, sequel of, succeeds to, successor to"
  }},
  {{
    "label": "number of pages",
    "id": "P1104",
    "description": "number of pages in an edition of a written work; see allowed units constraint for valid values to use for units in conjunction with a number",
    "aliases": "pagination, foliation, page count, pages, leaves, pp, leaf count, number of leaves, numpages"
  }},
  {{
    "label": "followed by",
    "id": "P156",
    "description": "immediately following item in a series of which the subject is a part, preferably use as qualifier of P179 [if the subject has been replaced, e.g. political offices, use \"replaced by\" (P1366)]",
    "aliases": "succeeded by, successor, comes before, continued as, next, next element, next is, precedes, prequel of, sequel is, then"
  }},
  {{
    "label": "employer",
    "id": "P108",
    "description": "person or organization for which the subject works or worked",
    "aliases": "organization, organisation, employed by, worked at, worked for, working at, working for, working place, works at, works for"
  }},
  {{
    "label": "has part(s)",
    "id": "P527",
    "description": "part of this subject; inverse property of \"part of\" (P361). See also \"has parts of the class\" (P2670).",
    "aliases": "amalgamation of, assembled from, assembled out of, composed of, consists of, created from, created out of, divided by, had part, has as part, has branch, has component, has ingredients, has part or parts, has parts, have part, holonym of, includes, includes part, incorporates, parts, set of, made up of, comprised of, has part, has member, contains, has ingredient, contain, formed from, formed out of, ingredients"
  }},
  {{
    "label": "ORCID iD",
    "id": "P496",
    "description": "identifier for a person",
    "aliases": "ORCID iD, Open Research Contributor ID, Open Researcher and Contributor ID, ORC ID, ORCiD, ORCID ID"
  }},
  {{
    "label": "described by source",
    "id": "P1343",
    "description": "work where this item is described",
    "aliases": "mentioned in, talked about in, discussed in, defined in, topic in, described in, subject of, entry, found in, subject in, described by biography, described by encyclopedia, described by obituary, described by reference work, described in source, mentioned in news article, reviewed in, source of item, written about in"
  }}
]
```

## EXAMPLES
- Question: Cats
Thoughts:
1. The question asks for information about cats, so I need to identify the relevant entities and properties in Wikidata.
2. First, I need to find items that are classified as cats. In Wikidata, "cat" corresponds to the entity with the identifier Q146.
3. To retrieve items that are instances of cats, I will use the property P31, which stands for "instance of."
4. I should also retrieve the labels of these items in a language the user understands. To do this, I'll utilize the SERVICE wikibase:label to get the label in the user's preferred language. If that language is unavailable, I'll default to a multilingual or English label.
SPARQL Query: ```sparql
SELECT ?item ?itemLabel
WHERE
{{
?item wdt:P31 wd:Q146.
SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }} # Helps get the label in your language, if not, then default for all languages, then en language
}}
```

- Question: Picture of Cats
Thoughts:
1. The query is focused on retrieving an image associated with the concept of "cats" in Wikidata.
2. In Wikidata, the item representing "cats" is identified by Q146.
3. The property P18 is used to denote images, so I'll look for the image associated with Q146.
4. The result will return the image linked to the "cats" item.
SPARQL Query: ```sparql
SELECT ?image WHERE {{
  wd:Q146 wdt:P18 ?image. # Get the image (P18) of Cats (Q146)
}}
```

- Question: Cats, with pictures
Thoughts:
1. The question now asks for information about cats, specifically including their pictures.
2. As before, I need to identify items that are classified as cats using the P31 property with the value Q146.
3. In addition to retrieving the item labels, I need to find the property that holds images associated with these items. In Wikidata, the property P18 is used for images.
4. I will add P18 to the query to retrieve the image associated with each cat item.
5. Finally, I'll include the SERVICE wikibase:label to ensure the labels are returned in the appropriate language, defaulting to multilingual or English if necessary.
SPARQL Query: ```sparql
SELECT ?item ?itemLabel ?pic WHERE {{
  ?item wdt:P31 wd:Q146;
    wdt:P18 ?pic.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }}
}}
```

- Question: Titles of articles about Ukrainian villages on Romanian Wikipedia
Thoughts:
1. The goal is to find articles about villages in Ukraine that exist on the Romanian Wikipedia.
2. First, I need to identify items classified as villages. In Wikidata, villages are represented by Q532.
3. I will then filter these villages to those located in Ukraine, represented by the country code Q212.
4. I need to check if there is a corresponding article for each village on the Romanian Wikipedia. This is done by filtering for schema:isPartOf with the value <https://ro.wikipedia.org/>.
5. Additionally, I will retrieve the titles of these articles on the Romanian Wikipedia (schema:name as page_titleRO).
6. To provide context, I'll also include the labels of these villages in English (LabelEN) and Ukrainian (LabelUK).
7. Finally, I'll limit the query to return up to 300 results.
SPARQL Query: ```sparql
SELECT DISTINCT ?item ?LabelEN ?LabelUK ?page_titleRO WHERE {{
  # item: is a - village
  ?item wdt:P31 wd:Q532 .
  # item: country - Ukraine
  ?item wdt:P17 wd:Q212 .
  # exists article in item that is ro.wiki
  ?article schema:about ?item ; schema:isPartOf <https://ro.wikipedia.org/> ; schema:name ?page_titleRO .
  # wd labels
  ?item rdfs:label ?LabelEN FILTER (lang(?LabelEN) = "en") .
  ?item rdfs:label ?LabelUK FILTER (lang(?LabelUK) = "uk") .
}}
LIMIT 300
```

- Question: Humans who died on August 25, 2001, on the English Wikipedia, ordered by label
Thoughts:
1. The query requires finding humans who died on a specific date: August 25, 2001.
2. In Wikidata, the date of death is represented by the property P570. I need to identify items where this property matches the specified date.
3. The query also focuses on articles available in English Wikipedia. I'll need to retrieve these articles, ensuring they are from the English Wikipedia by filtering with schema:isPartOf.
4. To sort the results by label, I must consider the proper sorting mechanism. I'll use a regex to clean the labels for sorting purposes, accounting for common prefixes in names (e.g., "von," "de") that might affect alphabetical order.
5. I also need to retrieve the item label and description in the appropriate language using the SERVICE wikibase:label.
6. Finally, the results should be ordered by the cleaned label (?sortname) and the original label.
SPARQL Query: ```sparql
SELECT ?item ?articlename ?itemLabel ?itemDescription ?sl
WHERE {{
VALUES ?dod {{"+2001-08-25"^^xsd:dateTime}}
    ?dod ^wdt:P570 ?item .
    ?item wikibase:sitelinks ?sl .
    ?item ^schema:about ?article .
    ?article schema:isPartOf <https://en.wikipedia.org/>;
    schema:name ?articlename .
SERVICE wikibase:label
    {{
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en" .
    ?item rdfs:label ?itemLabel .
    ?item schema:description ?itemDescription .
    }}
BIND(REPLACE(?itemLabel, "^.*(?<! [Vv][ao]n| [Dd][aeiu]| [Dd][e][lns]| [Ll][ae]) (?!([SJ]r\\.?|[XVI]+)$)", "") AS ?sortname)
}} ORDER BY ASC(UCASE(?sortname)) ASC(UCASE(?itemLabel))
```

- Question: The top 10 heaviest humans
Thoughts:
1. The goal is to identify and list the top 10 heaviest humans based on their recorded weight.
2. Humans are represented in Wikidata by the entity Q5.
3. The property P2067 represents the mass of an individual.
4. To extract the relevant data, I'll search for humans (Q5) who have a recorded mass (P2067).
5. The query should order these individuals by their mass in descending order to find the heaviest.
6. I'll limit the results to the top 10 entries.
7. Additionally, I will include the labels for each individual in multiple languages, prioritizing the user's language settings, and falling back to English, Spanish, French, and German.
SPARQL Query: ```sparql
SELECT ?item ?itemLabel ?mass
WHERE {{
{{
    SELECT ?item ?mass WHERE {{
    ?item wdt:P31 wd:Q5;
            p:P2067/psn:P2067/wikibase:quantityAmount ?mass.
    }}
    ORDER BY DESC(?mass)
    LIMIT 10
}}
SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,es,fr,de" }}
}}
ORDER BY DESC(?mass)
```

- Question: Number of humans in Wikidata
Thoughts:
1. The question asks for the total number of humans recorded in Wikidata.
2. To find this, I need to identify items that are classified as humans. In Wikidata, the entity for "human" is represented by Q5.
3. I'll use the P31 property, which stands for "instance of," to find all items that are instances of humans.
4. Since the question asks for a count, I'll use the COUNT(*) function to calculate the total number of items that match this criterion.
SPARQL Query: ```sparql
SELECT (COUNT(*) AS ?count)
WHERE {{
?item wdt:P31 wd:Q5 .
}}
```

- Question: List of countries ordered by the number of their cities with a female mayor
Thoughts:
1. The goal is to find countries and list them based on the number of cities within each country that have a female mayor.
2. First, I need to identify instances of cities. In Wikidata, cities or their subclasses are represented by Q515.
3. To find cities with female mayors, I'll use the P6 property, which indicates the head of government. I need to ensure that the head of government is female, which is represented by Q6581072 in Wikidata.
4. I'll also filter out any entries where the mayor's term has ended by checking for the absence of the P582 property (end date).
5. Next, I'll retrieve the country associated with each city using the P17 property.
6. The results should be grouped by country and ordered by the count of cities with a female mayor in descending order.
7. The query will include labels for countries, prioritized by the "ru" (Russian) language, and falling back to "en" (English) if needed.
8. Finally, I'll limit the results to the top 100 countries.
SPARQL Query: ```sparql
SELECT ?country ?countryLabel (count(*) AS ?count)
WHERE
{{
    ?city wdt:P31/wdt:P279* wd:Q515 . # find instances of subclasses of city
    ?city p:P6 ?statement .           # with a P6 (head of goverment) statement
    ?statement ps:P6 ?mayor .         # ... that has the value ?mayor
    ?mayor wdt:P21 wd:Q6581072 .      # ... where the ?mayor has P21 (sex or gender) female
    FILTER NOT EXISTS {{ ?statement pq:P582 ?x }}  # ... but the statement has no P582 (end date) qualifier
    ?city wdt:P17 ?country .          # Also find the country of the city

    # If available, get the "ru" label of the country, use "en" as fallback:
    SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "ru,en" .
    }}
}}
GROUP BY ?country ?countryLabel
ORDER BY DESC(?count)
LIMIT 100
```

- Question: Average number of children per year
Thoughts:
1. The question asks for the average number of children that people have, grouped by their birth year.
2. I'll first identify individuals (humans) in Wikidata, which are represented by Q5.
3. The property P1971 is used to denote the number of children an individual has. I'll retrieve this information for each person.
4. I'll also retrieve each person's birth date using the P569 property and extract the year from the birth date.
5. The results will be filtered to include only those born after 1900 to ensure more recent and relevant data.
6. The query will then group the data by birth year and calculate the average number of children for each year using the AVG function.
7. Finally, I'll return the birth year (year) and the average number of children (count).
SPARQL Query: ```sparql
SELECT  (str(?year) AS ?year) (AVG( ?_number_of_children ) AS ?count) WHERE {{
  ?item wdt:P31 wd:Q5.
  ?item wdt:P1971 ?_number_of_children.
  ?item wdt:P569 ?_date_of_birth.
  BIND( year(?_date_of_birth) as ?year ).
  FILTER( ?year > 1900)
}}

GROUP BY ?year
```

## QUESTION
- Question: {question}
"""

        prompt = PromptTemplate(
            template=template, input_variables=["question", "entity_ids"]
        )

        model_kwargs = {"device": self.device, "max_new_tokens": 1000}
        if self.local:
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            llm = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        raw_response = llm_chain.run(
            question=question,
            entity_ids=entity_ids,
        ).split(
            "## QUESTION"
        )[-1]

        # postprocessing
        if (
            "order" not in question.lower() or "sort" not in question.lower()
        ) and "random" not in question.lower():
            raw_response = (
                raw_response.replace("ORDER BY RAND()", "")
                .replace("ORDER BY DESC(RAND())", "")
                .replace("ORDER BY ASC(RAND())", "")
            )

        if verbose:
            tmp = raw_response.replace("\n", "<br/>")
            print(tmp)

        query = self._extract_sparql_query(raw_response.split("SPARQL Query: ")[-1])
        return query

    def run(self, question: str, return_query: bool = False, verbose: int = 0):
        print("tes")
        extracted_entities = self.extract_entity(question, model_name=self.model_name)
        if verbose == 1:
            print(extracted_entities)
        entity_ids = self.get_entity_ids(
            question, extracted_entities, model_name=self.model_name
        )
        if verbose == 1:
            print(entity_ids)
        query = self.generate_sparql(
            question,
            entity_ids,
            model_name=self.model_name,
            verbose=verbose > 0,
        )
        if query == "":
            return "Sorry, we are not supported with this kind of query yet."
        try:
            result = self.execute_sparql_to_wikidata(query)
        except Exception as e:
            print(e)
            result = []
        if return_query:
            return query, result
        return result

    def chat(
        self,
        question: str,
        verbose: int = 0,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ) -> str:
        wikidata_context = self.run(question, verbose=verbose)
        if not wikidata_context:
            return "Sorry, we are not supported with this kind of question yet."
        if verbose == 1:
            print(wikidata_context)

        template = """## INSTRUCTIONS
- You are a master of Wikidata, you know everything about Wikidata because you are given the answer in the context.
- Generate the answer from the given question by utilizing the given context from Wikidata.
- Try your best to use the given context as the answer!
- DO NOT hallucinate and only provide answers from the given context.
- DO NOT make up an answer
- If you don't know the answer, just say that you don't know
- Answer the question in a natural way like you are the one who know the context, DO NOT mention like "according to the context", etc.
- Answer it using complete sentence!

## CONTEXT
```json{{
    "wikidata_response": {wikidata_context}
}}```

## QUESTION
{question}

## ANSWER
"""

        prompt = PromptTemplate(
            template=template, input_variables=["question", "entity_ids"]
        )
        model_kwargs = {"device": self.device, "max_new_tokens": 2000}
        if self.local:
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            llm = HuggingFaceHub(repo_id=model_name, model_kwargs=model_kwargs)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return llm_chain.run(
            question=question,
            wikidata_context=wikidata_context,
        ).split("## ANSWER\n")[-1]
