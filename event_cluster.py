import os
import json
from tqdm import tqdm
import argparse
import copy

import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb

from utils import completion_with_llm
from create_chroma_bert_large import create_db_word, create_db_sentence
from prompt_template import RELATION_STATEMENTS_SUMMARY_PROMPT, RELATION_STATEMENTS_SUMMARY_PROMPT_TMP, SAME_EVENT_CLUSTER_SPLIT_PROMPT_TMP, SAME_EVENT_CLUSTER_SPLIT_PROMPT, DAY_SUMMARIZE_PROMPT_TMP, DAY_SUMMARIZE_PROMPT, SAME_EVENT_CLUSTER_PROMPT, SAME_EVENT_CLUSTER_PROMPT_TMP


from openai import OpenAI
import openai
import requests


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    parser.add_argument("--model", type=str, default="bert-large-cased")
    parser.add_argument("--Qwen_model", type=str, default="/mnt/sdb1/Qwen/Qwen3-4B")
    args = parser.parse_args()
    return args


API_SECRET_KEY = "xxxxxx"
BASE_URL = "https://api.zhizengzeng.com/v1/"

# chat with other model
def chat_completions4(prompt_str, split_str):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_str}
        ]
    )
    
    if resp and hasattr(resp, 'choices') and resp.choices:
        compressed_text = resp.choices[0].message.content
    else:
        print("Error: Invalid API response", resp)
        compressed_text = ""
    compressed_content = compressed_text.split(split_str)[-1].strip()
    
    first_part = compressed_content.split("#################", 1)[0]
    first_part = first_part.split("###", 1)[0].strip()
    
    return first_part


def relation_events_summarization(dir_path, dataset, keyword, prompt):
    same_and_relation_events_file = os.path.join(dir_path, "same_and_relation_events.json")
    with open(same_and_relation_events_file, "r", encoding='utf-8') as f:
        same_and_relation_events = json.load(f)

    same_and_relation_events_summaries_list = []
    multievents_group = dict()
    for same_and_relation_event in tqdm(same_and_relation_events, desc=f"relational events summarization:{dataset}-{keyword}"):
        if len(same_and_relation_event[0][2]) == 0:
            if same_and_relation_event[0][1] not in multievents_group.keys():
                multievents_group[same_and_relation_event[0][1]] = same_and_relation_event[0][0]
            else:
                multievents_group[same_and_relation_event[0][1]] = same_and_relation_event[0][0] if len(same_and_relation_event[0][0]) > len(multievents_group[same_and_relation_event[0][1]]) else multievents_group[same_and_relation_event[0][1]]
            same_and_relation_events_summaries = [[same_and_relation_event[0][0], same_and_relation_event[0][1]], []]
        else:
            relation_set = set()
            relation_set.add(same_and_relation_event[0][1])
            content_list = []
            content_and_ids = same_and_relation_event[0][2]
            content_and_ids.append([same_and_relation_event[0][0], same_and_relation_event[0][1]])
            content_and_ids = sorted(content_and_ids, key=lambda x:x[1])
            for relation_event_info in content_and_ids:
                relation_set.add(relation_event_info[1])
                content_list.append(relation_event_info[0])
            relation_tuple = tuple(sorted(list(relation_set)))

            if relation_tuple not in multievents_group.keys():
                content = "\n".join(content_list)
                current_prompt = prompt.format(Input_Set=content)
                split_str = "### The Summarization of the above Input Set 3 is"
                res = chat_completions4(current_prompt, split_str)
                multievents_group[relation_tuple] = res
                same_and_relation_events_summaries = [[res, same_and_relation_event[0][1]], []]
            else:
                same_and_relation_events_summaries = [[multievents_group[relation_tuple], same_and_relation_event[0][1]], []]
        
        for same_events in same_and_relation_event[1]:
            if len(same_events[2]) == 0:
                if same_events[1] not in multievents_group.keys():
                    multievents_group[same_events[1]] = same_events[0]
                else:
                    multievents_group[same_events[1]] = same_events[0] if len(same_events[0]) > len(multievents_group[same_events[1]]) else multievents_group[same_events[1]]
                same_and_relation_events_summaries[1].append([same_events[0], same_events[1]])
            else:
                relation_set = set()
                relation_set.add(same_events[1])
                content_list = []
                content_and_ids = same_events[2]
                content_and_ids.append([same_events[0], same_events[1]])
                content_and_ids = sorted(content_and_ids, key=lambda x:x[1])
                for relation_event_info in content_and_ids:
                    relation_set.add(relation_event_info[1])
                    content_list.append(relation_event_info[0])
                relation_tuple = tuple(sorted(list(relation_set)))

                if relation_tuple not in multievents_group.keys():
                    content = "\n".join(content_list)
                    current_prompt = prompt.format(Input_Set=content)
                    split_str = "### The Summarization of the above Input Set 3 is"
                    res = chat_completions4(current_prompt, split_str)
                    multievents_group[relation_tuple] = res
                    same_and_relation_events_summaries[1].append([res, same_events[1]])
                else:
                    same_and_relation_events_summaries[1].append([multievents_group[relation_tuple], same_events[1]])

        same_and_relation_events_summaries_list.append(same_and_relation_events_summaries) 

        with open(os.path.join(dir_path, "same_events_summaries_1.json"), "a", encoding="utf-8") as f:
            json.dump(same_and_relation_events_summaries, f, ensure_ascii=False, indent=4)
            f.write("\n")
    
    same_and_relation_events_summaries_list = sorted(same_and_relation_events_summaries_list, key=lambda x:x[0][1])
    with open(os.path.join(dir_path, "same_events_summaries_1.json"), "w", encoding="utf-8") as f:
        json.dump(same_and_relation_events_summaries_list, f, ensure_ascii=False, indent=4)


def id2events_construction(dir_path, dataset, k):
    with open(os.path.join(dir_path, "same_events_summaries_1.json"), "r", encoding="utf-8") as f:
        same_events_summaries_list = json.load(f)
    with open(os.path.join(dir_path, "same_event.json"), "r", encoding="utf-8") as f:
        same_events_list = json.load(f)
    with open(os.path.join(dir_path, "same_and_relation_events.json"), "r", encoding="utf-8") as f:
        same_and_relation_events_list = json.load(f)
    with open(os.path.join(dir_path, "retrospect_events.json"), "r", encoding="utf-8") as f:
        retrospect_events = json.load(f)

    same_events_summaries_list = sorted(same_events_summaries_list, key=lambda x:x[0][1])
    same_events_list = sorted(same_events_list, key=lambda x:x[0][1])
    same_and_relation_events_list = sorted(same_and_relation_events_list, key=lambda x:x[0][1])
    
    retrospect_events = {int(key): value for key, value in retrospect_events.items()}

    assert len(same_events_summaries_list) == len(same_and_relation_events_list)
    id2events = dict()

    for same_events_summaries, same_and_relation_events in zip(tqdm(same_events_summaries_list, desc=f"id2event construction:{dataset}-{k}"), same_and_relation_events_list):
        assert same_events_summaries[0][1] == same_and_relation_events[0][1]
        event_id = same_and_relation_events[0][1]

        if len(same_and_relation_events[0][2]) == 0:
            if event_id in id2events.keys():
                id2events[event_id] = same_events_summaries[0][0] if (len(same_events_summaries[0][0]) < len(id2events[event_id]) and len(same_events_summaries[0][0].split(" ")) > 12) else id2events[event_id]
            else:
                id2events[event_id] = same_events_summaries[0][0]
        else:
            event_ids_set = set()
            event_ids_set.add(event_id)
            for relation_events_info in same_and_relation_events[0][2]:
                event_ids_set.add(relation_events_info[1])
                event_ids_tuple = tuple(sorted(list(event_ids_set)))
                if event_ids_tuple[0] in id2events.keys():
                    id2events[event_ids_tuple[0]] = same_events_summaries[0][0] if (len(same_events_summaries[0][0]) < len(id2events[event_ids_tuple[0]]) and len(same_events_summaries[0][0].split(" ")) > 12) else id2events[event_ids_tuple[0]]
                else:
                    id2events[event_ids_tuple[0]] = same_events_summaries[0][0]

        assert len(same_events_summaries[1]) == len(same_and_relation_events[1])
        for same_events_summary, same_events_info in zip(same_events_summaries[1], same_and_relation_events[1]):
            assert same_events_summary[1] == same_events_info[1]
            event_id = same_events_summary[1]
            if len(same_events_info[2]) == 0:
                if event_id in id2events.keys():
                    id2events[event_id] = same_events_summary[0] if (len(same_events_summary[0]) < len(id2events[event_id]) and len(same_events_summaries[0][0].split(" ")) > 12) else id2events[event_id]
                else:
                    id2events[event_id] = same_events_summary[0]

            else:
                event_ids_set = set()
                event_ids_set.add(event_id)
                for relation_events_info in same_events_info[2]:
                    event_ids_set.add(relation_events_info[1])
                    event_ids_tuple = tuple(sorted(list(event_ids_set)))
                    if event_ids_tuple[0] in id2events.keys():
                        id2events[event_ids_tuple[0]] = same_events_summary[0] if (len(same_events_summary[0]) < len(id2events[event_ids_tuple[0]]) and len(same_events_summaries[0][0].split(" ")) > 12) else id2events[event_ids_tuple[0]]
                    else:
                        id2events[event_ids_tuple[0]] = same_events_summary[0]

    for key, value in retrospect_events.items():
        if key not in id2events.keys():
            id2events[key] = value

    with open(os.path.join(dir_path, "id2event.json"), "w", encoding="utf-8") as f:
        json.dump(id2events, f, ensure_ascii=False, indent=4)


def same_events_clustering(dir_path, dataset, keyword, db_1):
    with open(os.path.join(dir_path, "id2event.json"), "r", encoding="utf-8") as f:
        id2event = json.load(f)
    
    id2event = {int(key): value for key, value in id2event.items()}

    event_pool = {}
    event2cluster = {}

    if dataset == "t17":
        if k != "h1n1":
            prompt = SAME_EVENT_CLUSTER_PROMPT
        else:
            prompt = SAME_EVENT_CLUSTER_PROMPT_TMP

    split_str = "### The determination of whether the above two statements are the same event is"

    retreive_event_id_set = set()
    for event_id, q_content in tqdm(id2event.items(), desc=f"same event clustering:{dataset}-{keyword}"):
        if len(q_content.split(" ")) < 10:
            continue
        if event_id not in event2cluster.keys():
            if len(event_pool) == 0:
                max_pool_len = 0
            else:
                max_pool_len = max(list(event_pool.keys())) + 1
            
            event_pool[max_pool_len] = [event_id]
            event2cluster[event_id] = max_pool_len
        
        cls_id = event2cluster[event_id]
        content_emb = gte_model.encode(q_content)
        results = db_1.query(
            query_embeddings=[content_emb.tolist()],
            n_results=9,
            include=["metadatas", "distances", "documents"]
        )
        no_count = 0
        for content, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if len(content.split(" ")) < 10:
                continue
            neibor_id = metadata["event_id"]
            if neibor_id == event_id:
                continue
            
            if neibor_id in retreive_event_id_set:
                continue
            
            current_prompt = prompt.format(input_1=q_content, input_2=content)
            res = chat_completions4(current_prompt, split_str)

            if "yes" in res.lower():        
                if neibor_id not in event2cluster.keys():
                    event_pool[cls_id].append(neibor_id)
                    event2cluster[neibor_id] = cls_id
                else:
                    neibo_cls_id = event2cluster[neibor_id]
                    if neibo_cls_id == cls_id:
                        continue
                    for n_id in event_pool[neibo_cls_id]:
                        if n_id not in event_pool[cls_id]:
                            event_pool[cls_id].append(n_id)
                        event2cluster[n_id] = cls_id
                    del event_pool[neibo_cls_id]
            
            else:
                no_count += 1
                if no_count == 2:
                    retreive_event_id_set.add(event_id)
                    break

    with open(os.path.join(dir_path, "event_pool_1.json"), "w", encoding="utf-8") as f:
        json.dump(event_pool, f, ensure_ascii=False, indent=4)
    with open(os.path.join(dir_path, "event2cluster_1.json"), "w", encoding="utf-8") as f:
        json.dump(event2cluster, f, ensure_ascii=False, indent=4)


def cluster_splitting(cls_id, event_id_group_dic, event_pool, event2cluster):
    for event_id_list in event_id_group_dic.values():
        max_key = max(list(event_pool.keys())) if event_pool else 0
        unused_keys = [c_id for c_id in range(max_key) if c_id not in event_pool]
        new_keys = list(range(max_key + 1, max_key * 10))
        new_cls_id = min(unused_keys + new_keys)
        event_pool[new_cls_id] = event_id_list
        for event_id in event_id_list:
            event2cluster[event_id] = new_cls_id
    
    del event_pool[cls_id]


def postprocess_clusters(id2event, same_event_pool, same_event2cluster, dataset, keyword):
    client = PersistentClient()
    collection = client.get_collection(name=f"gte_{dataset}_{keyword}")
    same_event_pool_tmp = copy.deepcopy(same_event_pool)
    for cls_id, event_id_list in tqdm(same_event_pool_tmp.items(), desc=f"postprocess clusters:{dataset}-{keyword}"):
        if len(event_id_list) > 1:
            article2event_id = dict()
            for event_id in event_id_list:
                results = collection.get(
                    where={"event_id": event_id}  # 元数据过滤条件
                )
                query_article_id = results["metadatas"][0]["article_id"]
                q_content = id2event[event_id]
                if query_article_id in article2event_id.keys():
                    flag = 0
                    for comp_event_id in article2event_id[query_article_id]:
                        comp_content = id2event[comp_event_id]
                        if q_content == comp_content:
                            flag = 1
                            same_event_pool[cls_id].remove(event_id)
                            del same_event2cluster[event_id]
                            break

                    if flag == 0:
                        article2event_id[query_article_id].append(event_id)
                
                else:
                    article2event_id[query_article_id] = []
                    article2event_id[query_article_id].append(event_id)
    
    return same_event_pool, same_event2cluster


def same_cluster_splitting(dir_path, dataset, keyword, gte_model):
    if dataset == "t17":
        if k != "h1n1":
            prompt = SAME_EVENT_CLUSTER_SPLIT_PROMPT
        else:
            prompt = SAME_EVENT_CLUSTER_SPLIT_PROMPT_TMP
    
    split_str = "### The determination of whether the above two statements are the same event is"

    with open(os.path.join(dir_path, "event_pool_1.json"), "r", encoding="utf-8") as f:
        event_pool = json.load(f)
    
    with open(os.path.join(dir_path, "event2cluster_1.json"), "r", encoding="utf-8") as f:
        event2cluster = json.load(f)

    with open(os.path.join(dir_path, "id2event.json"), "r", encoding="utf-8") as f:
        id2event = json.load(f)
    
    event_pool = {int(key): value for key, value in event_pool.items()}
    event2cluster = {int(key): value for key, value in event2cluster.items()}
    id2event = {int(key): value for key, value in id2event.items()}

    event_pool, event2cluster = postprocess_clusters(id2event, event_pool, event2cluster, dataset, keyword)

    event_pool_tmp = copy.deepcopy(event_pool)
    for cls_id, event_id_list in tqdm(event_pool_tmp.items(), desc=f"acquire same event clusters:{dataset}-{keyword}"):
        if len(event_id_list) > 1:
            event_id_group_dic = dict()
            flag_set = set()
            
            event_id_list_len = len(event_id_list)
            for e_idx_1 in range(15):
                if e_idx_1 < event_id_list_len:
                    event_id = event_id_list[e_idx_1]
                    content_1 = id2event[event_id]
                    if event_id in flag_set:
                        continue
                    flag_set.add(event_id)
                    event_id_group = [event_id]
                    event_id_group_dic[event_id] = event_id_group

                    for e_idx_2 in range(e_idx_1 + 1, 15):
                        if e_idx_2 >= event_id_list_len:
                            break
                        event_id_tmp = event_id_list[e_idx_2]
                        content_2 = id2event[event_id_tmp]
                        if event_id_tmp in flag_set:
                            continue
                        current_prompt = prompt.format(input_1=content_1, input_2=content_2)
                        res = chat_completions4(current_prompt, split_str)
                        # res = completion_with_llm(Qwen_tokenizer, Qwen_model, current_prompt, split_str, temperature=0.0, stop_tokens=["#################", "###"], max_len=3)
            
                        if "yes" in res.lower():
                            flag_set.add(event_id_tmp)
                            event_id_group.append(event_id_tmp)

                    event_id_group_dic[event_id] = event_id_group        

            if event_id_list_len > 15:
                client = chromadb.Client()
                collection = client.create_collection(name=f"{keyword}-my_collection", metadata={"hnsw:space": "cosine"})
                event_id_tmp_list = list(event_id_group_dic.keys())
                content_list = []
                embeddings = []
                for event_id_tmp in event_id_group_dic.keys():
                    content_list.append(id2event[event_id_tmp])
                    embedding = gte_model.encode(id2event[event_id_tmp])
                    embeddings.append(embedding.tolist())
                
                list_len = len(event_id_tmp_list)
                i = 0
                while i < list_len:
                    if i + 5000 >= list_len:
                        collection.add(
                            embeddings=embeddings[i: list_len],
                            documents=content_list[i: list_len],
                            metadatas=[{"event_id": e_id} for e_id in event_id_tmp_list[i: list_len]],
                            ids=[str(i_) for i_ in list(range(i, list_len))]
                        )
                        break
                    else:
                        collection.add(
                            embeddings=embeddings[i: i + 5000],
                            documents=content_list[i: i + 5000],
                            metadatas=[{"event_id": e_id} for e_id in event_id_tmp_list[i: i + 5000]],
                            ids = [str(i_) for i_ in list(range(i, i + 5000))]
                        )
                        i += 5000
                
                for e_idx_1 in tqdm(range(15, event_id_list_len), desc=f"{keyword}-long length"):
                    event_id = event_id_list[e_idx_1]
                    content_1 = id2event[event_id]
                    
                    embedding = gte_model.encode(content_1)
                    embeddings = [embedding.tolist()]
                    n_results = min(3, list_len)
                    results = collection.query(
                        query_embeddings=embeddings,  # 查询文本
                        n_results=n_results,               # 返回最相似的3个结果
                        include=["documents", "distances", "metadatas"]  # 返回的内容
                    )

                    similiar_events = results["documents"][0]
                    similiar_event_ids = [re["event_id"] for re in results["metadatas"][0]]


                    flag = 0
                    for event_neibor_id, event_content in zip(similiar_event_ids, similiar_events):

                        current_prompt = prompt.format(input_1=content_1, input_2=event_content)
                        res = chat_completions4(current_prompt, split_str)

                        if "yes" in res.lower():
                            flag = 1
                            event_id_group_dic[event_neibor_id].append(event_id)
                            break

                    if flag == 0:
                        event_id_group_dic[event_id] = [event_id]
                        event_id_tmp_list = list(set(list(event_id_group_dic.keys())))
                        i_ = len(event_id_tmp_list) - 1
                        
                        
                        content_tmp = id2event[event_id]
                        embedding = gte_model.encode(id2event[event_id])
                        embedding_tmp = embedding.tolist()
                        
                        collection.add(
                            embeddings=[embedding_tmp],
                            documents=[content_tmp],
                            metadatas=[{"event_id": event_id}],
                            ids=[str(i_)]
                        )

                client.delete_collection(name=f"{keyword}-my_collection")

            with open(os.path.join(dir_path, "splitting_same_events.json"), "a", encoding="utf-8") as f:
                json.dump(event_id_group_dic, f, ensure_ascii=False, indent=4)
                f.write("\n")

            cluster_splitting(cls_id, event_id_group_dic, event_pool, event2cluster)   

    with open(os.path.join(dir_path, "event_pool_same_events.json"), "w", encoding="utf-8") as f:
        json.dump(event_pool, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(dir_path, "event2cluster_same_events.json"), "w", encoding="utf-8") as f:
        json.dump(event2cluster, f, ensure_ascii=False, indent=4)


def create_db_id2event(dir_path, dataset, keyword, gte_model):
    client = PersistentClient()
    
    try:
        collection = client.get_collection(name=f"gte_id2event_{dataset}_{keyword}")
    except:
        # 创建或获取集合（Collection）
        collection = client.create_collection(
            name=f"gte_id2event_{dataset}_{keyword}",
            embedding_function=None,  # 禁用默认嵌入
            metadata={"hnsw:space": "cosine"}
        )
        
        with open(os.path.join(dir_path, "id2event.json"), "r", encoding='utf-8') as f:
            train_data = json.load(f)

        train_data = {int(key): value for key, value in train_data.items()}

        embeddings = []
        metadatas = []
        documents = []
        ids = []  # ChromaDB 1.0+ 需要显式指定 ID（可选）

        idx = 0
        for event_id, content in tqdm(train_data.items(), desc=f"create gte_id2event_{dataset}_{keyword}"):
            
            if len(content.split(" ")) < 10:
                continue
            # 获取句子嵌入（假设你的函数返回 numpy 数组）
            embedding = gte_model.encode(content)
                
            embeddings.append(embedding.tolist())  # 转为 list
            metadatas.append({"event_id": event_id})
            documents.append(content)
            ids.append(f"id_{idx}")  # 生成唯一 ID（可自定义）
            idx += 1

        batch_size = 5000  # 确保小于报错中的限制（5461）
        for i in range(0, len(documents), batch_size):
            collection.add(
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size] if ids else None
            )
        
    return collection  # 返回集合对象


if __name__ == "__main__":
    args = get_argparser()

    if args.keyword == "all":
        keyword = [name for name in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, name))]
    else:
        keyword = args.keyword.split(',')

    device = torch.device("cuda:3")
    
    gte_model = SentenceTransformer('thenlper/gte-large').to(device)

    for k in keyword:
        dir_path = f"./processing/{args.dataset}/{k}/"

        if args.dataset == "t17":
            if k != "h1n1":
                prompt = RELATION_STATEMENTS_SUMMARY_PROMPT
            else:
                prompt = RELATION_STATEMENTS_SUMMARY_PROMPT_TMP
        
        relation_events_summarization(dir_path, args.dataset, k, prompt)
        id2events_construction(dir_path, args.dataset, k)
        
        '''
        client = chromadb.PersistentClient()
        client.delete_collection(name=f"gte_id2event_{args.dataset}_{k}")
        '''
        
        db_id2event = create_db_id2event(dir_path, args.dataset, k, gte_model)
        same_events_clustering(dir_path, args.dataset, k, db_id2event)
        
        same_cluster_splitting(dir_path, args.dataset, k, gte_model) 
