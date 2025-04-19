import json
import time
import openai
import re

from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

class Log:
    def __init__(self, start_time, **kwargs):
        self.start_time = start_time
        self.llm_call_count = 0
        self.wikidata_call_count = 0
        self.depth = 0
        self.errors = []
        self.logs = {}
        if kwargs:
            self.logs = kwargs

    def update_log(self, **kwargs):
        self.logs.update(kwargs)

    def generate_logs(self):
        total_run_time = time.time() - self.start_time
        runtime_str = "%.2f" % total_run_time
        self.update_log(runtime=runtime_str)
        self.logs.update(llm_call_count=self.llm_call_count)
        self.logs.update(wikidata_call_count=self.wikidata_call_count)
        self.logs.update(depth=self.depth)
        self.logs.update(errors=self.errors)
        return self.logs

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
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
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
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

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    if "llama" in engine.lower():
        url = "https://8mgifevo4.localto.net/v1"
        response = None
        client = OpenAI(
            base_url = url,
            api_key="token-abc123",
            # base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            # api_key=os.getenv("GEMINI_API_KEY"),
        )
        messages = [
            {"role": "system", "content": "Answer according to the example format. Pay attention to the formatting of the answers, usually the answer is within curly brackets."}
                    ]
        message_prompt = {"role": "user", "content": prompt}
        messages.append(message_prompt)

        while response is None:
            try:
                completion_response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    # model="gemini-2.0-flash",
                    messages=messages,
                    temperature=temperature,
                    # max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    extra_headers={'localtonet-skip-warning': 'localtonet-skip-warning'},
                )
                response = completion_response
            except Exception as e:
                print(f"Error: {e}")
                print("LLM error, retry")
                time.sleep(2)
        result = response.choices[0].message.content
        return result
    else:
        openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information. Pay attention to the formatting of the answers, usually the answer is within curly brackets."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    f = 0
    client = OpenAI(api_key=opeani_api_keys)
    while(f == 0):
        try:
            # response = client.responses.create(
            #     model=engine,
            #     input=messages,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     frequency_penalty=0,
            #     presence_penalty=0,
            # )
            # result = response.output_text

            response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response.choices[0].message.content

            f = 1
        except Exception as e:
            print(f"Error: {e}")
            print("openai error, retry")
            time.sleep(2)
    return result

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, log:Log=None):
    logs = "No logging was done."
    if log is not None:
        logs = log.generate_logs()
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "logs": logs}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args, log:Log):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    log.llm_call_count+=1
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name == 'creak_test':
        with open('../data/creak_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name == 'qald_test':
        with open('../data/qald_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_test':
        with open('../data/cwq_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex_test':
        with open('../data/T-REX_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'
    elif dataset_name == 'webquestions_test':
        with open('../data/WebQuestions_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'zeroshotre_test':
        with open('../data/Zero_Shot_RE_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string