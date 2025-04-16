import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, lit
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import uuid
import pandas as pd

# Initialize Spark with MongoDB connector
def init_spark():
    return SparkSession.builder \
        .appName("Distributed BERT Fine-Tuning") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()

# Load IMDB and SST-2 data to MongoDB
def load_data_to_mongodb(spark):
    # IMDB dataset
    imdb_dataset = load_dataset("imdb")
    imdb_df = pd.concat([
        imdb_dataset["train"].to_pandas()[["text", "label"]].head(50), # use 50 for debug
        imdb_dataset["test"].to_pandas()[["text", "label"]].head(50)
    ])
    imdb_df["source"] = "IMDB"
    imdb_spark_df = spark.createDataFrame(imdb_df).select(col("text"), col("label").cast("integer"), col("source"))
    
    # SST-2 dataset
    sst2_dataset = load_dataset("glue", "sst2")
    sst2_df = sst2_dataset["train"].to_pandas()[["sentence", "label"]].head(50) # use 50 for debug
    sst2_df = sst2_df.rename(columns={"sentence": "text"})
    sst2_df["source"] = "SST-2"
    sst2_spark_df = spark.createDataFrame(sst2_df).select(col("text"), col("label").cast("integer"), col("source"))
    
    # Write to MongoDB
    imdb_spark_df.write.format("mongo").mode("append").save()
    sst2_spark_df.write.format("mongo").mode("append").save()

# Preprocess data
def preprocess_data(spark):
    raw_df = spark.read.format("mongo").load()
    processed_df = raw_df.filter(length(col("text")) >= 10)
    
    # Split IMDB into train/test
    imdb_df = processed_df.filter(col("source") == "IMDB")
    train_df, test_df = imdb_df.randomSplit([0.8, 0.2], seed=42)
    
    # SST-2 as test set
    sst2_test_df = processed_df.filter(col("source") == "SST-2")
    
    # Store processed data in MongoDB
    train_collection = f"train_{uuid.uuid4().hex}"
    test_collection = f"test_{uuid.uuid4().hex}"
    sst2_collection = f"sst2_{uuid.uuid4().hex}"
    train_df.write.format("mongo").option("collection", train_collection).mode("overwrite").save()
    test_df.write.format("mongo").option("collection", test_collection).mode("overwrite").save()
    sst2_test_df.write.format("mongo").option("collection", sst2_collection).mode("overwrite").save()
    
    return spark.read.format("mongo").option("collection", train_collection).load(), \
           spark.read.format("mongo").option("collection", test_collection).load(), \
           spark.read.format("mongo").option("collection", sst2_collection).load()

# PyTorch Dataset for Spark DataFrame
class SparkDataset(Dataset):
    def __init__(self, spark_df, tokenizer, max_length=128):
        self.data = spark_df.toPandas()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Training and evaluation
def train_and_evaluate(rank, world_size, train_dataset, test_dataset, sst2_test_dataset, batch_size=8, epochs=3):
    # Setup distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = DDP(model.to(rank), device_ids=[rank])
    
    # DataLoaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    sst2_test_loader = DataLoader(sst2_test_dataset, batch_size=batch_size)
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # if rank==0:
        print(f"GPU[{rank}]: Epoch {epoch+1}, Avg Loss: {total_loss / len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    for dataset_name, loader in [("IMDB Test", test_loader), ("SST-2 Test", sst2_test_loader)]:
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(rank)
                attention_mask = batch["attention_mask"].to(rank)
                labels = batch["labels"].to(rank)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        # if rank == 0:
        print(f"GPU[{rank}]: {dataset_name} Accuracy: {correct / total:.4f}")
    
    dist.destroy_process_group()

# Main
if __name__ == "__main__":
    spark = init_spark()
    
    # Load and preprocess data
    load_data_to_mongodb(spark)
    train_df, test_df, sst2_test_df = preprocess_data(spark)
    train_df, test_df, sst2_test_df = train_df.limit(6), test_df.limit(6), sst2_test_df.limit(6)
    
    # Create datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SparkDataset(train_df, tokenizer)
    test_dataset = SparkDataset(test_df, tokenizer)
    sst2_test_dataset = SparkDataset(sst2_test_df, tokenizer)
    
    # Run distributed training
    world_size = max(1, torch.cuda.device_count())
    print(f"Using {world_size} GPU(s)")
    
    import torch.multiprocessing as mp
    mp.spawn(
        train_and_evaluate,
        args=(world_size, train_dataset, test_dataset, sst2_test_dataset),
        nprocs=world_size,
        join=True
    )
    
    spark.stop()