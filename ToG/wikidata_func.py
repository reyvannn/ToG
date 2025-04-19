import datetime
import email.utils

from SPARQLWrapper import SPARQLWrapper, JSON

# from ToG.main_wikidata import Log
import requests
import time

from utils import *

# SPARQLPATH = "http://192.168.80.12:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md
SPARQLPATH = "https://query.wikidata.org/sparql"

# pre-defined sparqls
# sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
# sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
# sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
# sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
# sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""

# Standard Wikidata Prefixes (good practice to include)
# WD_PREFIX = "PREFIX wd: <http://www.wikidata.org/entity/>\nPREFIX wdt: <http://www.wikidata.org/prop/direct/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX wikibase: <http://wikiba.se/ontology#>\nPREFIX bd: <http://www.bigdata.com/rdf#>\n"
WD_PREFIX=""

# Find outgoing properties (relations) from an entity
# Returns property URIs (e.g., http://www.wikidata.org/prop/direct/P31)
sparql_head_relations = WD_PREFIX + """
SELECT DISTINCT ?relation ?relationLabel WHERE {
  wd:%s ?relation ?x .
  FILTER(STRSTARTS(STR(?relation), STR(wdt:))) # Filter for direct properties only
  
  BIND(REPLACE(STR(?relation), STR(wdt:), "") AS ?propertyId) # Extract the property ID (Pxxx)
  BIND(IRI(CONCAT(STR(wd:), ?propertyId)) AS ?propertyEntity) # Construct the property URI (wd:Pxxx) needed by the label service
  
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
    ?propertyEntity rdfs:label ?relationLabel .
  }
}"""

# Find incoming properties (relations) to an entity
# Returns property URIs
sparql_tail_relations = WD_PREFIX + """
SELECT DISTINCT ?relation ?relationLabel WHERE {
  ?x ?relation wd:%s .
  FILTER(STRSTARTS(STR(?relation), STR(wdt:))) # Filter for direct properties only
  
  BIND(REPLACE(STR(?relation), STR(wdt:), "") AS ?propertyId) # Extract the property ID (Pxxx)
  BIND(IRI(CONCAT(STR(wd:), ?propertyId)) AS ?propertyEntity) # Construct the property URI (wd:Pxxx) needed by the label service
  
  # Get label
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
    ?propertyEntity rdfs:label ?relationLabel .
  }
}"""

# Find tail entities (objects) for a given head (subject) and relation (property)
sparql_tail_entities_extract = WD_PREFIX + """
SELECT DISTINCT ?tailEntity WHERE {
  wd:%s wdt:%s ?tailEntity .
  FILTER(isIRI(?tailEntity) && STRSTARTS(STR(?tailEntity), STR(wd:)))
}"""

# Find head entities (subjects) for a given tail (object) and relation (property)
# Kept ?tailEntity in SELECT for compatibility with existing parsing code instead of changing to ?headEntity
sparql_head_entities_extract = WD_PREFIX + """
SELECT DISTINCT ?tailEntity WHERE { # Variable name kept as ?tailEntity for compatibility
  ?tailEntity wdt:%s wd:%s .
  FILTER(isIRI(?tailEntity) && STRSTARTS(STR(?tailEntity), STR(wd:)))
}"""

# Get the English label for a given Wikidata entity ID
sparql_id = WD_PREFIX + """
SELECT DISTINCT ?tailEntity WHERE {
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
    wd:%s rdfs:label ?tailEntity .
  }
} LIMIT 1 # Often you just want one primary label
"""

sparql_property_label = WD_PREFIX + """
SELECT DISTINCT ?label WHERE {
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
    wd:%s rdfs:label ?label .
  }
}
LIMIT 1 # Return only one label
"""

def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


# def execurte_sparql(sparql_query):
#     results = None
#     while results is None:
#         sparql = SPARQLWrapper(SPARQLPATH)
#         sparql.setQuery(sparql_query)
#         sparql.setReturnFormat(JSON)
#         # results = sparql.query().convert()
#         try:
#             results = sparql.query().convert()
#         except Exception as e:
#             print(f"Error: {e}")
#             time.sleep(2)
#             continue
#     return results["results"]["bindings"]

def execurte_sparql(query_string, log):
    data = execute_sparql(query_string, log=log)
    if not data:
        return [] if isinstance(data, list) else None
    return data["results"]["bindings"]

import logging
from utils import Log
logging.basicConfig(level=logging.INFO)
def execute_sparql(query_string, max_retries=2**16, default_retry_after=1, log:Log=None):
    if log is None:log=Log(start_time=time.time())
    errors_and_warnings = []
    url = 'https://query.wikidata.org/sparql'
    headers = {
                # 'Authorization': f'Bearer {bearer_token}',
                'User-Agent': 'ThinkOnGraphBot/1.0 (muhammad.reyvan@ui.ac.id)'
               }
    params = {'query': query_string, 'format': 'json'}
    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)

            if response.status_code == 200:
                try:
                    data = response.json()
                    return data
                except requests.exceptions.JSONDecodeError as json_e:
                    logging.error(f"JSON Decode Error: {json_e}. Response text: {response.text[:]}")
                    errors_and_warnings.append(f"JSON Decode Error: {json_e}. Response text: {response.text[:]}")
                    log_errors_and_warnings = log.errors
                    log_errors_and_warnings.extend(errors_and_warnings)
                    log.errors = log_errors_and_warnings
                    return None # Or raise an exception

            # Handle Retry-After
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    logging.info(f"Retry-After header found: {retry_after}")

                    try:
                        # Try parsing as int
                        wait_time = int(retry_after)
                    except ValueError:
                        # If not, try parse as HTTP-date
                        retry_date_tuple = email.utils.parsedate_tz(retry_after)
                        if retry_date_tuple:
                            # Convert to timestamp
                            retry_timestamp = email.utils.mktime_tz(retry_date_tuple)
                            # Get current timestamp
                            now_utc_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
                            # Calculate the difference
                            wait_time = max(0, int(retry_timestamp-now_utc_time))
                            logging.info(f"Parsed Retry-After date. Need to wait approx {wait_time:.2f}")
                            errors_and_warnings.append(f"Parsed Retry-After date. Need to wait approx {wait_time:.2f}")
                            log_errors_and_warnings = log.errors
                            log_errors_and_warnings.extend(errors_and_warnings)
                            log.errors = log_errors_and_warnings
                        else:
                            logging.warning(f"Could not parse Retry-After value: {retry_after}. Use default backoff.")
                            errors_and_warnings.append(f"Could not parse Retry-After value: {retry_after}. Use default backoff.")
                            wait_time = default_retry_after * (2**retries)
                            log_errors_and_warnings = log.errors
                            log_errors_and_warnings.extend(errors_and_warnings)
                            log.errors = log_errors_and_warnings

                else:
                # No Retry-After
                    logging.warning("No Retry-After")
                    wait_time = default_retry_after * (2**retries)
                logging.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
                errors_and_warnings.append(f"Waiting for {wait_time:.2f} seconds before retrying...")
                log_errors_and_warnings = log.errors
                log_errors_and_warnings.extend(errors_and_warnings)
                log.errors = log_errors_and_warnings
                time.sleep(wait_time)
                retries += 1

            # Handle Server-Side Timeout
            elif response.status_code == 500 and "java.util.concurrent.TimeoutException" in response.text:
                logging.warning(f"Server-side timeout (500) detected (Attempt {retries + 1}/{max_retries}). Retrieved data: {response.json()}")
                # wait_time = default_retry_after * (2**retries)
                # logging.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
                # time.sleep(wait_time)
                # continue

                logging.warning("Skipping this query...")
                errors_and_warnings.append(f"Server-side timeout (500) detected (Attempt {retries + 1}/{max_retries}). Retrieved data: {response.json()}")
                log_errors_and_warnings = log.errors
                log_errors_and_warnings.extend(errors_and_warnings)
                log.errors = log_errors_and_warnings
                return []

            # Handle other non-successful status codes (4xx, 5xx)
            else:
                logging.error(f"Request failed with status {response.status_code}: {response.text[:]}")
                logging.info(f"Retrieved data: {response.json()}")
                # response.raise_for_status() # Or simply return None/break
                # raise_for_status() is good as it raises an HTTPError for bad responses
                # If we reach here, it means raise_for_status didn't raise (e.g., if disabled)
                # So we should explicitly stop.
                errors_and_warnings.append(f"Request failed with status {response.status_code}: {response.text[:]}\nRetrieved data: {response.json()}")
                log_errors_and_warnings = log.errors
                log_errors_and_warnings.extend(errors_and_warnings)
                log.errors = log_errors_and_warnings
                return [] # Failed


        # CLIENT-Side exception
        except requests.exceptions.ReadTimeout as e_read:
            logging.warning(
                f"Read Timeout occurred (attempt {retries+1}/{max_retries}): {e_read}. Query: {query_string[:100]}...")
            # Apply exponential backoff for timeouts
            # wait_time = default_retry_after * (2 ** (retries))  # Use retries-1 as it's already incremented
            # logging.info(f"Waiting {wait_time:.2f} seconds before retrying due to Read Timeout...")
            # time.sleep(wait_time)
            # retries += 1
            # continue

            logging.warning("Skipping this query...")
            errors_and_warnings.append(f"Read Timeout occurred (attempt {retries+1}/{max_retries}): {e_read}. Query: {query_string[:]}\nSkipping this query...")
            log_errors_and_warnings = log.errors
            log_errors_and_warnings.extend(errors_and_warnings)
            log.errors = log_errors_and_warnings
            return []
        except Exception as e:
            retries += 1
            logging.error(f"Request Exception (attempt {retries}/{max_retries}): {e}. Query: {query_string}. Exception: {e}")
            time.sleep(2 ** retries)  # Exponential backoff for connection errors
            errors_and_warnings.append(f"Request Exception (attempt {retries}/{max_retries}): {e}. Query: {query_string}. Exception: {e}")
            log_errors_and_warnings = log.errors
            log_errors_and_warnings.extend(errors_and_warnings)
            log.errors = log_errors_and_warnings
            continue
    return []

# ADDED
# CHANGED
def get_relation_labels(relations):
    if len(relations) == 0:
        return []
    labels = [relation['relationLabel']['value'] for relation in relations]
    return labels

def replace_relation_prefix(relations):
    # return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations] # CHANGED
    if len(relations) == 0:
        return []
    relations = [relation['relation']['value'] for relation in relations]
    new_relations = []
    for relation in relations:
        if 'wikidata.org' in relation:
            parts = relation.rsplit('/', 1)
            new_relations.append(parts[-1])
        else:
            new_relations.append(relation)
    return new_relations

def replace_entities_prefix(entities):
    # return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities] # CHANGED
    if len(entities) == 0:
        return []
    entities = [entity['tailEntity']['value'] for entity in entities]
    new_entities = []
    for entity in entities:
        if 'wikidata.org' in entity:
            parts = entity.rsplit('/', 1)
            new_entities.append(parts[-1])
        else:
            print(f"Unknown entity format: {entity}")
            new_entities.append(entity)
    return new_entities

def is_valid_wikidata_id(entity_id):
  """Checks if a string is a syntactically valid Wikidata ID (Q### or P###)."""
  if not isinstance(entity_id, str):
      return False
  # ^         - Start of the string
  # [PQ]      - Must start with P or Q
  # [1-9]     - Followed by a digit from 1 to 9 (ensures no leading zeros like Q0)
  # \d*       - Followed by zero or more digits (0-9)
  # $         - End of the string
  pattern = r"^[PQ][1-9]\d*$"
  return bool(re.fullmatch(pattern, entity_id))

# CHANGED
def id2entity_name_or_type(entity_id):
    if not is_valid_wikidata_id(entity_id):
        return "UnName_Entity"
    results = execute_sparql(sparql_id % (entity_id))
    if results is None:
        return "UnName Entity"
    elif results == []:
        return "UnName Entity"
    elif len(results["results"]["bindings"]) == 0 :
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']

# from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def clean_relations(string, entity_id, head_relations, relations_labels = None):
    # pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}" # CHANGED
    pattern = r"{\s*(?P<relation>P[0-9]+)[^()]*?\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True}) # CHANGED
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False}) # CHANGED
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt_wiki % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, log:Log):
    relation_to_label = {}
    if entity_id is None or entity_id == [] or entity_id == "UnName_Entity" or entity_id == "ERROR" or not is_valid_wikidata_id(entity_id):
        return [], {}
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head, log=log)
    log.wikidata_call_count+=1
    if head_relations is None: # CHANGED
        return [], {}
    head_relation_labels = get_relation_labels(head_relations)
    head_relations = replace_relation_prefix(head_relations)

    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail, log=log)
    log.wikidata_call_count+=1
    if tail_relations is None:
        return [], {}
    elif tail_relations == [] and head_relations == []: # CHANGED
        return [], {}
    tail_relation_labels = get_relation_labels(tail_relations)
    tail_relations = replace_relation_prefix(tail_relations)

    relation_to_label.update(dict(zip(head_relations, head_relation_labels)))
    relation_to_label.update(dict(zip(tail_relations, tail_relation_labels)))

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    total_relations_with_labels = [f"{relation}:{relation_to_label.get(relation, relation)}" for relation in total_relations]
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations_with_labels, args)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        log.llm_call_count+=1
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations, relation_to_label)

    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        return [rel for rel in retrieve_relations_with_scores if is_valid_wikidata_id(rel['entity'])], relation_to_label
    else:
        return [], {} # format error or too small max_length
    
    
def entity_search(entity, relation, head=True, log:Log=None):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract, log)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract, log)
    if log is not None:
        log.wikidata_call_count += 1

    entity_ids = replace_entities_prefix(entities)
    # new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    # return new_entity
    return entity_ids


def entity_score(question, entity_candidates_id, score, relation, args, relation_to_label = None, log:Log=None):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if log is not None:log.wikidata_call_count += len(entity_candidates_id)
    if relation_to_label is not None:
        relation = relation_to_label.get(relation, relation)
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        if log is not None:log.llm_call_count+=1
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id

    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, relation_to_label = None):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    if relation_to_label is not None:
        entity['relation'] = relation_to_label.get(entity['relation'], entity['relation'])
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args, log=None):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args, log=log)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset, log=log)


def generate_answer(question, cluster_chain_of_entities, args, log:Log):
    prompt = answer_prompt_wiki + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    if log is not None:log.llm_call_count+=1
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, log:Log):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list] # CHANGED
    if len(filtered_list) ==0:
        return False, [], [], [], [] # ERROR WHEN ALL SCORES IS 0
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    log.wikidata_call_count+= len(tops)
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args, log:Log=None):
    prompt = prompt_evaluate_wiki + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    if log is not None:log.llm_call_count+=1
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    
