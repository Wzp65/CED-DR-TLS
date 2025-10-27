# RC-TLS
Repo for paper "Optimized Comprehensive Event Discovery Based on Dual-Retrieval Combined with An Accelerated Clustering Algorithm for Timeline Summarization". 


## Timeline Summarization 

### Download T17 Dataset
To download datasets(T17, Crisis, Entities), please refer to [complementizer/news-tls ](https://github.com/complementizer/news-tls).

### Download MAVEN Dataset
To download MAVEN datasets, please refer to [complementizer/news-tls ](https://github.com/THU-KEG/MAVEN-dataset).

### Workflow
To preprocess dataset articles to sentences, please refer to ```data_preprocess.py```

```
python data_preprocess.py --dataset "t17" --keyword "all"
```

To preprocess MAVEN dataset, please execute command ```cd ./maven_dataset``` and execute the following instruction steps in sequence:

```
python event_type_count.py
python syntactic_decomposite_maven.py --datatype "train"
python syntactic_decomposite_maven.py --datatype "valid"
python syntactic_statistics.py
python event_type_class.py
cd ../
```

Perform MAVEN chroma construction by ```create_chroma_bert_large.py```.

```
python create_chroma_bert_large.py
```

Parse T17 sentences to POS taggers by ```syntactic_decomposite.py```.

```
python syntactic_decomposite.py --dataset "t17" --keyword "all"
```

Perform event detection by ```event_detection.py```.

```
python event_detection.py --dataset "t17" --keyword "all"
```

Perform clustering process by ```event_cluster.py```

```
python generate_clusters.py \
    --dataset "t17" \
    --keyword "all"
```

Perform timeline summarization and evaluation by ```timeline_summarization.py```

```
python timeline_summarization.py \
    --dataset "t17" \
    --keyword "all"
```
