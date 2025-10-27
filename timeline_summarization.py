import os
import json
from tqdm import tqdm
import argparse
import copy
import re

import torch

from datetime import datetime, date, timedelta
import arrow

from openai import OpenAI
import openai
import requests

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from prompt_template import SAME_CLUSTER_SUMMARIZE_PROMPT_TMP, SAME_CLUSTER_SUMMARIZE_PROMPT, DAY_TIMELINE_JUDGMENT_PROMPT_TMP, DAY_TIMELINE_JUDGMENT_PROMPT
from utils import parse_time

from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge

from pprint import pprint
from evaluation import get_scores, evaluate_dates, get_average_results

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    parser.add_argument("--src_dir", type=str, default="./processing")
    parser.add_argument("--Qwen_model", type=str, default="/mnt/sdb1/Qwen/Qwen3-4B")
    parser.add_argument("--des_dir", type=str, default="./timeline")
    args = parser.parse_args()
    return args


def get_avg_score(scores):
    return sum(scores) / len(scores)


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def date_to_summaries(timeline):
    timeline_dic = dict()
    for day_events_info in timeline:
        time = day_events_info[0]
        parse_gold_time = parse_time(time)
        date_time = parse_gold_time.date()
        timeline_dic[date_time] = day_events_info[1]

    return timeline_dic

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
    #first_part = first_part.split("**", 1)[0].strip()
    
    return first_part


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: list, tokenizer):
        # 将停止标记转换为它们对应的 token ID
        self.stop_token_ids = [tokenizer.encode(stop_token, return_tensors='pt').squeeze(0).tolist() for stop_token in stop_tokens]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查生成的 token 是否包含停止标记
        for stop_token_ids in self.stop_token_ids:
            # 检查生成序列是否以指定的 token 结尾
            if input_ids[0, -len(stop_token_ids):].tolist() == stop_token_ids:
                return True
        return False


def completion_with_llm(tokenizer, model, prompt_str, split_str, temperature=0.0, stop_tokens=[], max_len=800):
    model_input = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    stop_criteria = StopAtSpecificTokenCriteria(stop_tokens=stop_tokens, tokenizer=tokenizer)
    generation_config = GenerationConfig(
        temperature=max(temperature, 1e-5),  # 确保温度值严格为正
        do_sample=temperature > 0.0          # 仅当temperature>0时启用采样
    )
    model.eval()
    with torch.no_grad():
        completion = model.generate(
            **model_input,
            generation_config=generation_config,
            max_new_tokens=max_len,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
            #pad_token_id=model.config.eos_token_id
        )[0]
    compressed_text = tokenizer.decode(completion, skip_special_tokens=True)
    compressed_content = compressed_text.split(split_str)[-1].strip()
    
    first_part = compressed_content.split("#################", 1)[0]
    first_part = first_part.split("###", 1)[0].strip()
    first_part = first_part.split("**", 1)[0].strip()
    
    return first_part


def postprocess_clusters(id2event, same_event_pool, same_event2cluster, dataset, keyword):
    client = PersistentClient()
    collection = client.get_collection(name=f"gte_{dataset}_{keyword}")
    same_event_pool_tmp = copy.deepcopy(same_event_pool)
    id2event_c = dict()
    event_c2id = dict()
    for cls_id, event_id_list in tqdm(same_event_pool_tmp.items(), desc=f"postprocess clusters:{dataset}-{keyword}"):
        for event_id in event_id_list:
            event_content = id2event[event_id]
            if event_content in event_c2id.keys():
                neibor_e_id = event_c2id[event_content]
                neibor_cls_id = same_event2cluster[neibor_e_id]
                if neibor_cls_id != cls_id:
                    same_event_pool[cls_id] = list(set(same_event_pool[cls_id]) | set(same_event_pool[neibor_cls_id]))
                    for neibor_event_id in same_event_pool[neibor_cls_id]:
                        same_event2cluster[neibor_event_id] = cls_id
                    del same_event_pool[neibor_cls_id]
            else:
                event_c2id[event_content] = event_id
                id2event_c[event_id] = event_content
    
    return same_event_pool, same_event2cluster


def cluster_summarization(id2event, same_event_pool, same_event2cluster, dataset, keyword, des_dir):
    client = PersistentClient()
    collection = client.get_collection(name=f"gte_{dataset}_{keyword}")

    if os.path.exists(os.path.join(des_dir, "same_cls2event.json")):
        with open(os.path.join(des_dir, "same_cls2event.json"), "r", encoding="utf-8") as f:
            same_cls2event = json.load(f)
        same_cls2event = {int(key): value for key, value in same_cls2event.items()}
    else:
        same_cls2event = dict()

    if args.dataset == "t17":
        if k != "h1n1":
            prompt = SAME_CLUSTER_SUMMARIZE_PROMPT
        else:
            prompt = SAME_CLUSTER_SUMMARIZE_PROMPT_TMP

    split_str = "### Based on the statements 2 above, write a concise summary of the main event part:"
    
    summarize_cls = dict()
    for cls_id, event_id_list in same_event_pool.items():
        flag = 0
        for event_id in event_id_list:
            results = collection.get(
                where={"event_id": event_id}  # 元数据过滤条件
            )
            query_article_time = results["metadatas"][0]["pubtime"]
            query_event_time = results["metadatas"][0]["time"]
            if query_article_time != query_event_time:
                flag = 1
                break

        if flag == 0:
            summarize_cls[cls_id] = event_id_list
    
    for cls_id, event_id_list in tqdm(same_event_pool.items(), desc=f"same event summarization:{dataset}-{keyword}"):
        if len(event_id_list) > 1:
            pub_time = []
            event_time = []
            if cls_id in summarize_cls.keys():
                same_cls2event[cls_id]["refer_event"] = 0
                content_list = []
                content = ""
                for i in range(0, min(9, len(event_id_list))):
                    event_id = event_id_list[i]
                    if id2event[event_id] in content_list:
                        content = id2event[event_id]
                    content_list.append(id2event[event_id])
                    results = collection.get(
                        where={"event_id": event_id}  # 元数据过滤条件
                    )

                    query_event_time = results["metadatas"][0]["time"]
                    query_pubtime = results["metadatas"][0]["pubtime"]
                    parse_query_event_time = parse_time(query_event_time)
                    parse_query_pubtime = parse_time(query_pubtime)
                    if parse_query_event_time == parse_query_pubtime:
                        pub_time.append(parse_query_pubtime)
                    else:
                        event_time.append(parse_query_event_time)
                
                if content != "":
                    same_cls2event[cls_id]["content"] = content
                else:
                    if not os.path.exists(os.path.join(des_dir, "same_cls2event.json")):
                        statements = "\n".join(content_list)
                        current_prompt = prompt.format(statements=statements)
                        res = chat_completions4(current_prompt, split_str)
                        print(res)
                        same_cls2event[cls_id] = dict()
                        same_cls2event[cls_id]["content"] = res
                pub_time = sorted(pub_time)
                event_time = sorted(event_time) 

            else:
                same_cls2event[cls_id]["refer_event"] = 1
                content = ""
                for i in range(0, len(event_id_list)):
                    event_id = event_id_list[i]
                    results = collection.get(
                        where={"event_id": event_id}  # 元数据过滤条件
                    )

                    query_event_time = results["metadatas"][0]["time"]
                    query_pubtime = results["metadatas"][0]["pubtime"]
                    parse_query_event_time = parse_time(query_event_time)
                    parse_query_pubtime = parse_time(query_pubtime)
                    if parse_query_event_time == parse_query_pubtime:
                        pub_time.append(parse_query_pubtime)
                    else:
                        if content == "":
                            content = id2event[event_id]
                        else:
                            content = content if len(content.split(" ")) < len(id2event[event_id].split(" ")) else id2event[event_id]
                        event_time.append(parse_query_event_time)
                if not os.path.exists(os.path.join(des_dir, "same_cls2event.json")):
                    same_cls2event[cls_id] = dict()
                same_cls2event[cls_id]["content"] = content
                pub_time = sorted(pub_time)
                event_time = sorted(event_time) 
            
            if len(event_time) > 0 and len(pub_time) > 0:
                if pub_time[0] > event_time[0] and (event_time[0].month != 1 and event_time[0].day != 1):
                    same_cls2event[cls_id]["time"] = event_time[0].strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    same_cls2event[cls_id]["time"] = pub_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            elif len(event_time) > 0 and len(pub_time) == 0:
                same_cls2event[cls_id]["time"] = event_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            elif len(event_time) == 0 and len(pub_time) > 0:
                same_cls2event[cls_id]["time"] = pub_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            
            '''
            if len(event_time) > 0:
                same_cls2event[cls_id]["time"] = event_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            else:
                same_cls2event[cls_id]["time"] = pub_time[0].strftime("%Y-%m-%dT%H:%M:%S")
            '''
        else:
            if not os.path.exists(os.path.join(des_dir, "same_cls2event.json")):
                same_cls2event[cls_id] = dict()
            same_cls2event[cls_id]["content"] = id2event[event_id_list[0]]
            results = collection.get(
                where={"event_id": event_id_list[0]}  # 元数据过滤条件
            )

            query_event_time = results["metadatas"][0]["time"]
            query_pubtime = results["metadatas"][0]["pubtime"]
            parse_query_event_time = parse_time(query_event_time)
            parse_query_pubtime = parse_time(query_pubtime)
            if parse_query_event_time == parse_query_pubtime:
                same_cls2event[cls_id]["time"] = query_pubtime
            else:
                if parse_query_pubtime > parse_query_event_time and (parse_query_event_time.month != 1 and parse_query_event_time.day != 1):
                    same_cls2event[cls_id]["time"] = query_event_time
                else:
                    same_cls2event[cls_id]["time"] = query_pubtime
            if cls_id in summarize_cls.keys():
                same_cls2event[cls_id]["refer_event"] = 0
            else:
                same_cls2event[cls_id]["refer_event"] = 1
    
    with open(os.path.join(des_dir, "same_cls2event.json"), "w", encoding="utf-8") as f:
        json.dump(same_cls2event, f, ensure_ascii=False, indent=4)


def acquire_sorted_event_info(main_events_dic, id2event, same_event_pool, same_event2cluster, dataset, keyword, des_dir, alpha):
    with open(os.path.join(des_dir, "same_cls2event.json"), "r", encoding="utf-8") as f:
        same_cls2event = json.load(f)
    
    same_cls2event = {int(key): value for key, value in same_cls2event.items()}
    '''
    client = PersistentClient()
    collection = client.get_collection(name=f"gte_{dataset}_{keyword}")
    
    for cls_id, event_id_list in same_event_pool.items():
        flag = 0
        for event_id in event_id_list:
            results = collection.get(
                where={"event_id": event_id}  # 元数据过滤条件
            )
            query_article_time = results["metadatas"][0]["pubtime"]
            query_event_time = results["metadatas"][0]["time"]
            if query_article_time != query_event_time:
                same_cls2event[cls_id]["refer_event"] = 1
                flag = 1
                break

        if flag == 0:
            same_cls2event[cls_id]["refer_event"] = 0
    '''
    for cls_id, event_id_list in same_event_pool.items():
        same_count = len(event_id_list)
        same_cls2event[cls_id]["same_event_count"] = same_count

        event_article_set = set()
        for event_id in event_id_list:
            event_article_set.add(event_id)
            if event_id in main_events_dic.keys():
                event_article_set.update(main_events_dic[event_id])
            
        same_cls2event[cls_id]["event_article_count"] = len(event_article_set)
    
    same_cls2event_tmp = copy.deepcopy(same_cls2event)
    for key, value in same_cls2event_tmp.items():
        if key not in same_event_pool.keys():
            del same_cls2event[key]

    for key, value in same_cls2event.items():
        if "same_event_count" not in value:
            print(f"Missing 'same_event_count' in key: {key}")
    sorted_same_cls2event = sorted(same_cls2event.items(), key=lambda x: (alpha * x[1]["same_event_count"] + (1-alpha) * x[1]["event_article_count"] + 3 * x[1]["refer_event"]), reverse=True)
    sorted_same_cls2event = dict(sorted_same_cls2event)
    
    sorted_same_cls2event_tmp = copy.deepcopy(sorted_same_cls2event)
    for key, item in sorted_same_cls2event_tmp.items():
        if item["content"].startswith("The provided input set"):
            del sorted_same_cls2event[key]
        if item["content"] == "":
            del sorted_same_cls2event[key]
    
    with open(os.path.join(des_dir, "same_cls2event_info.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_same_cls2event, f, ensure_ascii=False, indent=4)


def get_max_summary_length(ref_tl):
    lens = [len(summary[1]) for summary in ref_tl]
    return max(lens)

def get_average_summary_length(ref_tl):
    lens = [len(summary[1]) for summary in ref_tl]
    return round(sum(lens) / len(lens))


def acquire_start_end_time(golden_timelines):
    g_start_time, g_end_time = [], []
    start_time, end_time = golden_timelines[0][0][0], golden_timelines[0][-1][0]
    
    for timeline in golden_timelines:
        g_start_time.append(timeline[0][0])
        g_end_time.append(timeline[-1][0])
        if timeline[0][0] < start_time:
            start_time = timeline[0][0]
        if timeline[-1][0] > end_time:
            end_time = timeline[-1][0] 
    
    return g_start_time, g_end_time, start_time, end_time


def filtered_timelines(golden_timelines, timeline_event_info, dataset, keyword, des_dir):
    g_start_time, g_end_time, start_time, end_time = acquire_start_end_time(golden_timelines)
    parse_start_time = parse_time(start_time)
    parse_end_time = parse_time(end_time)

    results = []
    pred_timelines = []
    timelines_split = []
    for tl_index, (start_time, end_time) in enumerate(zip(g_start_time, g_end_time)):
        tilse_timeline = dict()
        parse_start_time = parse_time(start_time)
        parse_end_time = parse_time(end_time)
        
        timeline_event_info_tmp = copy.deepcopy(timeline_event_info)
        keys_to_delete = []
        for cls_id, event_info in timeline_event_info.items():
            try:
                parser_event_time = parse_time(event_info["time"])
            except ValueError:
                continue

            if parser_event_time < parse_start_time or parser_event_time > parse_end_time:
                keys_to_delete.append(cls_id)
        
        for cls_id in keys_to_delete:
            del timeline_event_info_tmp[cls_id]
        
        count = len(timeline_event_info_tmp)
        timeline_count = int(0.3 * count)
        timeline_event_info_tmp = dict(list(timeline_event_info_tmp.items())[:timeline_count])
        timelines_split.append(timeline_event_info_tmp)

        time_to_event = dict()
        for cls_id, event_info in timeline_event_info_tmp.items():
            event_info_time = event_info["time"]
            if event_info_time not in time_to_event.keys():
                time_to_event[event_info_time] = []
            time_to_event[event_info_time].append(event_info["content"])

        golden_timeline = golden_timelines[tl_index]
        max_len = get_max_summary_length(golden_timeline)
        print(keyword, str(max_len))
        event_info_len = len(timeline_event_info_tmp)
        for index, (cls_id, event_info) in enumerate(timeline_event_info_tmp.items()):
            try:
                parser_event_time = parse_time(event_info["time"])
            except:
                continue
            '''
            if parser_event_time < parse_start_time and (parse_start_time - parser_event_time).days >= day_diff:
                continue
            if parser_event_time > parse_end_time and (parser_event_time - parse_end_time).days >= day_diff:
                continue
            '''
            if parser_event_time < parse_start_time:
                continue
            if parser_event_time > parse_end_time:
                continue
            event_time_date = parser_event_time.date()
            if event_time_date not in tilse_timeline.keys():
                tilse_timeline[event_time_date] = []
            
            if len(time_to_event[event_info["time"]]) == 1:
                if index > len(timeline_event_info_tmp) / 2:
                    continue

            if len(tilse_timeline[event_time_date]) >= max_len + 1:
                continue
            tilse_timeline[event_time_date].append(event_info["content"])

        tilse_timeline_tmp = copy.deepcopy(tilse_timeline)
        '''
        for key, value in tilse_timeline_tmp.items():
            if len(value) == 0:
                del tilse_timeline[key]
        '''
        tilse_timeline_tmp = {key.strftime("%Y-%m-%d"): value for key, value in tilse_timeline.items()}

        pred_timelines.append(tilse_timeline_tmp)
    
    with open(os.path.join(des_dir, "pred_timelines.json"), "w", encoding="utf-8") as f:
        json.dump(pred_timelines, f, ensure_ascii=False, indent=4)

    with open(os.path.join(des_dir, "split_timelines.json"), "w", encoding="utf-8") as f:
        json.dump(timelines_split, f, ensure_ascii=False, indent=4)

def extract_text_between_stars(text):
    pattern = r'\*\*(.*?)\*\*'
    matches = re.findall(pattern, text)
    return matches


def clean_llm_output(timelines):
    new_timelines = []
    for timeline in timelines:
        timeline_dic = dict()
        for time, event_content in timeline.items():
            if "\n\n" in event_content:
                continue
            
            if event_content ==  "":
                continue

            matches = extract_text_between_stars(event_content)
            if len(matches) == 0:
                str_list = event_content.split("\n")
                event_list = [s for s in str_list if s != "---" and s != ""]
                timeline_dic[time] = event_list
            else:
                if "none" in matches[0].lower():
                    continue
                
                if len(matches[0].split(" ")) < 10:
                    continue

                else:
                    
                    timeline_dic[time] = []
                    for match in matches:
                        if len(match.split(" ")) >= 10:
                            timeline_dic[time].append(match)
        new_timelines.append(timeline_dic)
    
    return new_timelines


def furthur_filtering(pred_timelines, time_list, dataset, keyword, des_dir, average_timeline_length):
    '''
    date_list = []
    for timestamp in time_list:
        parse_gold_time = parse_time(timestamp)
        date_time = parse_gold_time.date()
        date_list = date_list + [
            (date_time - timedelta(days=i)).strftime("%Y-%m-%d")  # 前 3 天
            for i in range(3, 0, -1)
        ] + [
            date_time.strftime("%Y-%m-%d")  # 当天
        ] + [
            (date_time + timedelta(days=i)).strftime("%Y-%m-%d")  # 后 3 天
            for i in range(1, 3)
        ]
    date_list = sorted(list(set(date_list)))
    print(date_list)
    '''
    with open(os.path.join(des_dir, "split_timelines.json"), "r", encoding="utf-8") as f:
        split_timelines = json.load(f)

    assert len(split_timelines) == len(pred_timelines)
    
    number = 0
    for timeline in pred_timelines:
        for day_time, eevnt_list in timeline.items():
            for event_content in eevnt_list:
                number += 1
    
    if number > 300:
        new_timelines = []
        for timeline, split_timeline in zip(pred_timelines, split_timelines):
            split_len = len(split_timeline)
            timeline_dic = dict()
            new_split_timeline = dict(sorted(split_timeline.items(), key=lambda x: (0.5 * x[1]["same_event_count"] + 0.5 * x[1]["event_article_count"]), reverse=True))
            for day_time, event_list in timeline.items():
                new_event_list = []
                for event_content in event_list:
                    index, event_key = [(ind, key) for (ind, key) in enumerate(new_split_timeline.keys()) if new_split_timeline[key]["content"] == event_content][0]
                    event_dict = new_split_timeline[event_key]
                    
                    if index < int(0.03 * split_len):
                        new_event_list.append(event_content)
                        continue

                    if len(new_event_list) == 0:
                        if index < int(0.15 * split_len):
                            new_event_list.append(event_content)
                            continue
                    else:
                        if len(new_event_list) > average_timeline_length:
                            continue
                        if index >= int(0.3 * split_len):
                            continue
                        new_event_list.append(event_content)

                if len(new_event_list) > 0:
                    timeline_dic[day_time] = new_event_list
            new_timelines.append(timeline_dic)
    
        with open(os.path.join(des_dir, "final_pred_timelines_1.json"), "w", encoding="utf-8") as f:
            json.dump(new_timelines, f, ensure_ascii=False, indent=4)
    
    else:
        with open(os.path.join(des_dir, "final_pred_timelines_1.json"), "w", encoding="utf-8") as f:
            json.dump(pred_timelines, f, ensure_ascii=False, indent=4)


def final_timeline_formation(dataset, keyword, des_dir, src_dir, keywords_str, average_timeline_length):
    with open(os.path.join(src_dir, "candidate_filtering.json"), "r", encoding="utf-8") as f:
        cls2event_list = json.load(f)
    time_count = dict()
    for cls2event in cls2event_list:
        timestamp = cls2event["pubtime"]
        
        if timestamp not in time_count.keys():
            time_count[timestamp] = 0
        time_count[timestamp] += 1

    time_count = sorted(time_count.items(), key=lambda x:x[1], reverse=True)
    rank_length = int(len(time_count) * 0.1)
    time_count = time_count[:rank_length]
    time_list = list(set([timestamp[0] for timestamp in time_count]))
    

    '''
    with open(os.path.join(des_dir, "pred_timelines.json"), "r", encoding="utf-8") as f:
        pred_timelines = json.load(f)

    number = 0
    for timeline in pred_timelines:
        for day_time, eevnt_list in timeline.items():
            for event_content in eevnt_list:
                number += 1
    
    if number > 100:
        split_str = f"### The determination of whether the above event statement has potential to become a timeline event is:"
        pred_timelines_tmp = list()
        for timeline in tqdm(pred_timelines, desc=f"day remove:{dataset}-{keyword}"):
            new_timeline_dic = dict()
            for day_time, eevnt_list in timeline.items():
                new_event_list = []
                for event_content in eevnt_list:
                    if keyword != "syria" and keyword != "h1n1" and keyword != "egypt":
                        prompt = DAY_TIMELINE_JUDGMENT_PROMPT
                    else:
                        prompt = DAY_TIMELINE_JUDGMENT_PROMPT_TMP
                    c_prompt = prompt.format(keywords=keywords_str, statements=event_content)
                    
                    res = chat_completions4(c_prompt, split_str)
                    #res = completion_with_llm(Qwen_tokenizer, Qwen_model, c_prompt, split_str, temperature=0.35, stop_tokens=[], max_len=3000)
                    print(res)
                    print("\n")
                    if "yes" in res.lower():
                        new_event_list.append(event_content)
                if len(new_event_list) > 0:
                    new_timeline_dic[day_time] = new_event_list
            pred_timelines_tmp.append(new_timeline_dic)

        pred_timelines = pred_timelines_tmp

    #new_timlines = clean_llm_output(pred_timelines_tmp)

    with open(os.path.join(des_dir, "final_pred_timelines.json"), "w", encoding="utf-8") as f:
        json.dump(pred_timelines, f, ensure_ascii=False, indent=4)
    '''
    with open(os.path.join(des_dir, "final_pred_timelines.json"), "r", encoding="utf-8") as f:
        pred_timelines = json.load(f)

    furthur_filtering(pred_timelines, time_list, dataset, keyword, des_dir, average_timeline_length)
                

def evaluate_timelines(golden_timelines, dataset, keyword, des_dir):
    '''
    if os.path.exists(os.path.join(des_dir, "pred_timelines_info.json")):
        with open(os.path.join(des_dir, "pred_timelines_info.json"), "r", encoding="utf-8") as f:
            pred_timelines = json.load(f)
        pred_timelines_tmp = copy.deepcopy(pred_timelines)
        for pred_timeline_tmp, pred_timeline in zip(pred_timelines_tmp, pred_timelines):
            for event_time_str, events_list in pred_timeline_tmp.items():
                events_list = [event_c for event_c in events_list if event_c != ""]
                pred_timeline[event_time_str] = events_list
    ''' 
    with open(os.path.join(des_dir, "final_pred_timelines_1.json"), "r", encoding="utf-8") as f:
        pred_timelines = json.load(f)
    '''
    if dataset == "t17":
        if k == "haiti" or k == "iraq":
            prompt = DAY_SUMMARIZE_PROMPT
        else:
            prompt = DAY_SUMMARIZE_PROMPT_TMP
    '''
    assert len(pred_timelines) == len(golden_timelines)

    results = []
    
    pred_timelines_info = []
    # split_str = "### Based on Statements 3, all simplified event statements placed on separate lines are:"
    
    for golden_timeline, pred_timeline in zip(tqdm(golden_timelines, desc=f"acquire timeline info:{dataset}-{keyword}"), pred_timelines):    

        tilse_timeline = {datetime.strptime(key, "%Y-%m-%d").date(): value for key, value in pred_timeline.items()}
        '''
        if not os.path.exists(os.path.join(des_dir, "pred_timelines_info.json")):
            tilse_timeline_tmp = copy.deepcopy(tilse_timeline)
            for event_time, event_content_list in tilse_timeline_tmp.items():
                day_events = "\n".join(event_content_list)
                current_prompt = prompt.format(statements=day_events)
                res = chat_completions4(current_prompt, split_str)
                print(res)
                res_list = res.split("\n")
                tilse_timeline[event_time] = res_list
        '''

        pred_timeline = TilseTimeline(tilse_timeline)
        ground_truth = TilseGroundTruth([TilseTimeline(date_to_summaries(golden_timeline))])    

        evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
        rouge_scores = get_scores(metric, pred_timeline, ground_truth, evaluator)
        date_scores = evaluate_dates(pred_timeline, ground_truth)
        timeline_res = (rouge_scores, date_scores, pred_timeline)
        results.append(timeline_res)
        tilse_timeline_tmp = {key.strftime("%Y-%m-%d"): value for key, value in tilse_timeline.items()}

        pred_timelines_info.append(tilse_timeline_tmp)
    
    trial_res = get_average_results(results)
    rouge_1 = trial_res[0]['f_score']
    trial_res[0]['f_score'] = rouge_1
    rouge_2 = trial_res[1]['f_score']
    trial_res[1]['f_score'] = rouge_2
    date_f1 = trial_res[2]['f_score']
    trial_res[2]['f_score'] = date_f1

    trial_save_path = des_dir
    save_json(trial_res, os.path.join(trial_save_path, 'avg_score_llm_fur.json'))
    
    print(rouge_1, rouge_2, date_f1)
    
    return rouge_1


if __name__ == "__main__":
    args = get_argparser()

    root_des_dir = os.path.join(args.des_dir, args.dataset)
    root_src_dir = os.path.join(args.src_dir, args.dataset)
    root_dataset_path = os.path.join(args.dataset_path, args.dataset)

    if args.keyword == "all":
        keyword = [name for name in os.listdir(root_src_dir) 
                if os.path.isdir(os.path.join(root_src_dir, name))]
    else:
        keyword = args.keyword.split(',')

    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    '''
    Qwen_tokenizer = AutoTokenizer.from_pretrained(args.Qwen_model)
    Qwen_model = AutoModelForCausalLM.from_pretrained(args.Qwen_model, device_map="cuda:1")
    '''
    r1_list = []
    r2_list = []
    date_list = []
    for k in tqdm(keyword, desc=f"acquire timeline and evaluate:{args.dataset}"):
        src_dir = os.path.join(root_src_dir, k)
        des_dir = os.path.join(root_des_dir, k)
        os.makedirs(des_dir, exist_ok=True)
        
        dataset_path = os.path.join(dataset_dir, k)
        
        keywords_path = os.path.join(dataset_path, "keywords.json")
        with open(keywords_path, "r", encoding="utf-8") as f:
            keywords_list = json.load(f)
        keywords_str = ",".join(keywords_list)

        id2event_path = os.path.join(src_dir, "id2event.json")
        with open(id2event_path, "r", encoding="utf-8") as f:
            id2event = json.load(f)
        same_event_pool_path = os.path.join(src_dir, "event_pool_same_events.json")
        with open(same_event_pool_path, "r", encoding="utf-8") as f:
            same_event_pool = json.load(f)
        same_event2cluster_path = os.path.join(src_dir, "event2cluster_same_events.json")
        with open(same_event2cluster_path, "r", encoding="utf-8") as f:
            same_event2cluster = json.load(f)
        
        event_diff_article_file = os.path.join(src_dir, "same_events_summaries_1.json")
        with open(event_diff_article_file, "r", encoding="utf-8") as f:
            event_diff_articles = json.load(f)
        
        main_events_dic = dict()
        for item in event_diff_articles:
            main_events_dic[item[0][1]] = []
            e_list = item[1]
            for e in e_list:
                main_events_dic[item[0][1]].append(e[1])
        
        same_event_pool = {int(key): value for key, value in same_event_pool.items()}
        same_event2cluster = {int(key): value for key, value in same_event2cluster.items()}
        id2event = {int(key): value for key, value in id2event.items()}

        '''
        same_event_pool, same_event2cluster = postprocess_clusters(id2event, same_event_pool, same_event2cluster, args.dataset, k)
        same_event_pool_path = os.path.join(src_dir, "event_pool_same_events.json")
        same_event2cluster_path = os.path.join(src_dir, "event2cluster_same_events.json")
        with open(same_event_pool_path, "w", encoding="utf-8") as f:
            json.dump(same_event_pool, f, ensure_ascii=False, indent=4)
        with open(same_event2cluster_path, "w", encoding="utf-8") as f:
            json.dump(same_event2cluster, f, ensure_ascii=False, indent=4)
        '''

        #cluster_summarization(id2event, same_event_pool, same_event2cluster, args.dataset, k, des_dir)
        
        #acquire_sorted_event_info(main_events_dic, id2event, same_event_pool, same_event2cluster, args.dataset, k, des_dir, 0.3)
        
        with open(os.path.join(des_dir, "same_cls2event_info.json"), "r", encoding="utf-8") as f:
            sorted_timeline_event_info = json.load(f)

        evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
        metric = 'align_date_content_costs_many_to_one'
        
        golden_timelines_path = os.path.join(dataset_path, "timelines.jsonl")
        golden_timelines = []
        with open(golden_timelines_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                golden_timelines.append(data)

        average_timeline_length = get_average_summary_length(golden_timelines)
        #filtered_timelines(golden_timelines, sorted_timeline_event_info, args.dataset, k, des_dir)
        
        final_timeline_formation(args.dataset, k, des_dir, src_dir, keywords_str, average_timeline_length)
        
        evaluate_timelines(golden_timelines, args.dataset, k, des_dir)
        
        with open(os.path.join(des_dir, "avg_score_llm_fur.json"), "r", encoding="utf-8") as f:
            score_list = json.load(f)
        
        r1_list.append(score_list[0]["f_score"])
        r2_list.append(score_list[1]["f_score"])
        date_list.append(score_list[2]["f_score"])
        
    
    r1_f1 = get_avg_score(r1_list)
    r2_f1 = get_avg_score(r2_list)
    date_f1 = get_avg_score(date_list)

    results_dic = dict()
    results_dic["r1_f1"] = r1_f1
    results_dic["r2_f1"] = r2_f1
    results_dic["date_f1"] = date_f1

    with open(os.path.join(root_des_dir, "results_1_llm_fur.json"), "w", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False, indent=4)

    print(r1_f1, r2_f1, date_f1)
    
