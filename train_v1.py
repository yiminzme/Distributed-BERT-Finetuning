import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, lit
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import uuid
from datasets import load_dataset
import pandas as pd
import os


# Initialize Spark Session with MongoDB Connector
def init_spark():
    spark = SparkSession.builder \
        .appName("Distributed BERT Fine-Tuning") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()
    return spark

# Load datasets into MongoDB
def load_data_to_mongodb(spark):
    # Load IMDB dataset from Hugging Face
    imdb_dataset = load_dataset("imdb")
    imdb_train = imdb_dataset["train"].to_pandas()[["text", "label"]]
    imdb_test = imdb_dataset["test"].to_pandas()[["text", "label"]]
    imdb_df = pd.concat([imdb_train, imdb_test], ignore_index=True)
    imdb_df["source"] = "IMDB"
    imdb_spark_df = spark.createDataFrame(imdb_df).select(col("text"), col("label"), col("source"))
    
    # Load SST-2 dataset from Hugging Face
    sst2_dataset = load_dataset("glue", "sst2")
    sst2_data = sst2_dataset["train"].to_pandas()[["sentence", "label"]]
    sst2_data = sst2_data.rename(columns={"sentence": "text"})
    sst2_data["source"] = "SST-2"
    sst2_spark_df = spark.createDataFrame(sst2_data).select(col("text"), col("label"), col("source"))
    
    # Write to MongoDB
    imdb_spark_df.write.format("mongo").mode("append").save()
    sst2_spark_df.write.format("mongo").mode("append").save()

# Preprocess data using Spark DataFrame
def preprocess_data(spark):
    # Load raw data from MongoDB
    raw_df = spark.read.format("mongo").load()
    
    # Filter out short reviews (< 10 words)
    processed_df = raw_df.filter(length(col("text")) >= 10)
    
    # Convert labels to binary (assuming IMDB/SST-2 labels need standardization)
    processed_df = processed_df.withColumn(
        "label",
        when(col("label").isin(["positive", 1, "1"]), 1).otherwise(0)
    )
    
    # Split IMDB data into train/test (85%/15%)
    imdb_df = processed_df.filter(col("source") == "IMDB")
    train_df, test_df = imdb_df.randomSplit([0.85, 0.15], seed=42)
    
    # SST-2 as additional test set
    sst2_test_df = processed_df.filter(col("source") == "SST-2")
    
    return train_df, test_df, sst2_test_df

# Tokenization using Spark RDD with BERT Tokenizer
def tokenize_data(df, tokenizer):
    def tokenize_review(text):
        # Tokenize with BERT tokenizer
        encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"].squeeze().tolist(),
            "attention_mask": encodings["attention_mask"].squeeze().tolist()
        }
    
    # Convert DataFrame to RDD for tokenization
    rdd = df.rdd.map(lambda row: (row["text"], row["label"], tokenize_review(row["text"])))
    return rdd

# Store processed data back to MongoDB
def store_processed_data(spark, train_df, test_df, sst2_test_df):
    # Define DataFrames and their corresponding collection names
    datasets = [
        (train_df, f"processed_train_{uuid.uuid4().hex}"),
        (test_df, f"processed_test_{uuid.uuid4().hex}"),
        (sst2_test_df, f"processed_sst2_{uuid.uuid4().hex}")
    ]
    
    # Write each DataFrame to its respective MongoDB collection
    for df, collection_name in datasets:
        df.write.format("mongo").option("collection", collection_name).mode("overwrite").save()
    
    # Return collection names
    return datasets[0][1], datasets[1][1], datasets[2][1]

# Initialize BERT model and tokenizer
def init_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return tokenizer, model

# Distributed training setup
def setup_distributed_training(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Training loop with gradient accumulation
def train_bert(model, train_rdd, rank, world_size, batch_size=8, grad_accum_steps=4):
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Convert RDD to list for simplicity (in practice, use a distributed DataLoader)
    train_data = train_rdd.collect()
    
    model.train()
    for epoch in range(3):  # Example: 3 epochs
        accumulated_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            input_ids = torch.tensor([item[2]["input_ids"] for item in batch]).to(rank)
            attention_mask = torch.tensor([item[2]["attention_mask"] for item in batch]).to(rank)
            labels = torch.tensor([item[1] for item in batch]).to(rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            if (i // batch_size + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    
    dist.destroy_process_group()

# Evaluation
def evaluate_bert(model, test_rdd, dataset_name, rank):
    model.eval()
    correct = 0
    total = 0
    
    # Convert RDD to list for simplicity
    test_data = test_rdd.collect()
    
    with torch.no_grad():
        for item in test_data:
            input_ids = torch.tensor([item[2]["input_ids"]]).to(rank)
            attention_mask = torch.tensor([item[2]["attention_mask"]]).to(rank)
            labels = torch.tensor([item[1]]).to(rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    if rank == 0:
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    return accuracy

# Distributed training and evaluation wrapper
def run_distributed_training(rank, world_size, model, train_rdd, test_rdd, sst2_rdd):
    setup_distributed_training(rank, world_size)
    train_bert(model, train_rdd, rank, world_size)
    evaluate_bert(model, test_rdd, "IMDB Test", rank)
    evaluate_bert(model, sst2_rdd, "SST-2 Test", rank)

mp.set_start_method("spawn", force=True)

# Initialize Spark
spark = init_spark()

# Load datasets
load_data_to_mongodb(spark)

# Preprocess data
train_df, test_df, sst2_test_df = preprocess_data(spark)
train_df, test_df, sst2_test_df = train_df.limit(6), test_df.limit(6), sst2_test_df.limit(6)

# Initialize BERT tokenizer and model
tokenizer, model = init_bert()

# Tokenize data with BERT tokenizer
train_rdd = tokenize_data(train_df, tokenizer)
test_rdd = tokenize_data(test_df, tokenizer)
sst2_rdd = tokenize_data(sst2_test_df, tokenizer)

# Store processed data
train_collection, test_collection, sst2_collection = store_processed_data(
    spark, train_df, test_df, sst2_test_df
)

# Distributed training
# world_size = torch.cuda.device_count()
# mp.spawn(
#     run_distributed_training,
#     args=(world_size, model, train_rdd, test_rdd, sst2_rdd),
#     nprocs=world_size,
#     join=True
# )
world_size = torch.cuda.device_count()
run_distributed_training(
    rank=0,
    world_size=world_size,
    model=model,
    train_rdd=train_rdd,
    test_rdd=test_rdd,
    sst2_rdd=sst2_rdd
)