import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
import os
import uuid
import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf
import pyarrow.parquet as pq
import glob
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark with MongoDB connector
def init_spark(num_cpus = None):
    # if num_spark_executor_core: logger.info(f"{num_spark_executor_core} cores for executor")
    # else: logger.info(f"number of cores for executor UNDEFINED")
    if num_cpus: logger.info(f"{num_cpus} cores for spark")
    else: logger.info(f"num_cpus UNDEFINED")
    spark = SparkSession.builder \
        .appName("Distributed BERT Fine-Tuning with Preprocessing") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", 10) \
        .config("spark.executor.instances", 9) \
        .config("spark.default.parallelism", 96) \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()
        # .config("spark.cores.max", num_cpus) \
        # .config("spark.driver.cores", 2) \
        # .config("spark.executor.cores", 4) \
        # .config("spark.executor.instances", 3) \
        # .config("spark.default.parallelism", 10) \
    return spark

# Load IMDB and SST-2 data to MongoDB
def load_data_to_mongodb(spark):
    # IMDB dataset
    start_time = time.time()
    logger.info("Loading IMDB dataset...")
    imdb_dataset = load_dataset("imdb")
    imdb_df = pd.concat([
        imdb_dataset["train"].to_pandas()[["text", "label"]].head(2000),  # use 50 for debug
        imdb_dataset["test"].to_pandas()[["text", "label"]].head(2000)
    ])
    imdb_df["source"] = "IMDB"
    imdb_spark_df = spark.createDataFrame(imdb_df).select(col("text"), col("label").cast("integer"), col("source"))
    
    # SST-2 dataset
    logger.info("Loading SST-2 dataset...")
    sst2_dataset = load_dataset("glue", "sst2")
    sst2_df = sst2_dataset["train"].to_pandas()[["sentence", "label"]].head(2000)  # use 50 for debug
    sst2_df = sst2_df.rename(columns={"sentence": "text"})
    sst2_df["source"] = "SST-2"
    sst2_spark_df = spark.createDataFrame(sst2_df).select(col("text"), col("label").cast("integer"), col("source"))
    
    logger.info("Writing datasets to MongoDB...")
    imdb_spark_df.write.format("mongo").mode("append").save()
    sst2_spark_df.write.format("mongo").mode("append").save()
    return time.time() - start_time

# Batch tokenizer UDF
def create_batch_tokenizer_udf(max_length=128):
    def tokenize_batch(texts: pd.Series) -> pd.DataFrame:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encodings = tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        return pd.DataFrame({
            "input_ids": [ids.tolist() for ids in encodings["input_ids"]],
            "attention_mask": [mask.tolist() for mask in encodings["attention_mask"]]
        })
    
    schema = StructType([
        StructField("input_ids", ArrayType(IntegerType())),
        StructField("attention_mask", ArrayType(IntegerType()))
    ])
    
    return pandas_udf(tokenize_batch, schema)

# Preprocess data and save to Parquet
def preprocess_data(spark, output_dir, max_length=128):
    start_time = time.time()

    # Load and preprocess data
    logger.info("Reading data from MongoDB...")
    raw_df = spark.read.format("mongo").load()
    processed_df = raw_df.filter(length(col("text")) >= 10)
    
    # Apply distributed batch tokenization
    logger.info("Tokenizing data...")
    tokenize_udf = create_batch_tokenizer_udf(max_length)
    tokenized_df = processed_df.withColumn("encoding", tokenize_udf(col("text")))
    
    # Extract input_ids and attention_mask
    tokenized_df = tokenized_df.select(
        col("label").cast("integer").alias("label"),
        col("source"),
        col("encoding.input_ids").alias("input_ids"),
        col("encoding.attention_mask").alias("attention_mask")
    )
    
    # Split IMDB into train/test
    imdb_df = tokenized_df.filter(col("source") == "IMDB")
    train_df, test_df = imdb_df.randomSplit([0.8, 0.2], seed=42)
    sst2_test_df = tokenized_df.filter(col("source") == "SST-2")
    
    # Save to Parquet with dynamic partitioning
    num_partitions = max(16, spark.sparkContext.defaultParallelism * 2)  # Adjust based on cluster size
    train_path = os.path.join(output_dir, f"train_{uuid.uuid4().hex}")
    test_path = os.path.join(output_dir, f"test_{uuid.uuid4().hex}")
    sst2_test_path = os.path.join(output_dir, f"sst2_{uuid.uuid4().hex}")
    
    logger.info(f"Writing Parquet files: train={train_path}, test={test_path}, sst2={sst2_test_path}")
    train_df.select("input_ids", "attention_mask", "label").repartition(num_partitions).write.mode("overwrite").parquet(train_path)
    test_df.select("input_ids", "attention_mask", "label").repartition(num_partitions).write.mode("overwrite").parquet(test_path)
    sst2_test_df.select("input_ids", "attention_mask", "label").repartition(num_partitions).write.mode("overwrite").parquet(sst2_test_path)
    
    # Store processed data in MongoDB for reference
    train_collection = f"train_{uuid.uuid4().hex}"
    test_collection = f"test_{uuid.uuid4().hex}"
    sst2_collection = f"sst2_{uuid.uuid4().hex}"
    train_df.write.format("mongo").option("collection", train_collection).mode("overwrite").save()
    test_df.write.format("mongo").option("collection", test_collection).mode("overwrite").save()
    sst2_test_df.write.format("mongo").option("collection", sst2_collection).mode("overwrite").save()
    
    preprocess_time = time.time() - start_time
    return train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, preprocess_time

# Check for cached Parquet files
def check_cached_parquet(output_dir):
    train_path = test_path = sst2_test_path = None
    train_collection = test_collection = sst2_collection = None
    
    for dir_name in os.listdir(output_dir):
        if dir_name.startswith("train_"):
            train_path = os.path.join(output_dir, dir_name)
            train_collection = dir_name
        elif dir_name.startswith("test_"):
            test_path = os.path.join(output_dir, dir_name)
            test_collection = dir_name
        elif dir_name.startswith("sst2_"):
            sst2_test_path = os.path.join(output_dir, dir_name)
            sst2_collection = dir_name
    
    if train_path and test_path and sst2_test_path:
        logger.info(f"Found cached Parquet files: train={train_path}, test={test_path}, sst2={sst2_test_path}")
        return train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection
    return None

# Lazy-loading Parquet dataset
class LazyParquetDataset(IterableDataset):
    def __init__(self, parquet_path, rank, world_size, batch_size=1000):
        self.parquet_files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        
        # Shard files across ranks
        files_per_rank = len(self.parquet_files) // world_size
        start_idx = rank * files_per_rank
        end_idx = (rank + 1) * files_per_rank if rank < world_size - 1 else len(self.parquet_files)
        self.parquet_files = self.parquet_files[start_idx:end_idx]
    
    def __iter__(self):
        for file in self.parquet_files:
            logger.debug(f"Rank {self.rank} reading Parquet file: {file}")
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    yield {
                        "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
                        "labels": torch.tensor(row["label"], dtype=torch.long)
                    }

# Training and evaluation
def train_and_evaluate(rank, world_size, train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, finetune_time, batch_size=8, epochs=3):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = DDP(model.to(rank), device_ids=[rank])
    
    # Create datasets
    train_dataset = LazyParquetDataset(train_path, rank, world_size)
    test_dataset = LazyParquetDataset(test_path, rank, world_size)
    sst2_test_dataset = LazyParquetDataset(sst2_test_path, rank, world_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    sst2_test_loader = DataLoader(sst2_test_dataset, batch_size=batch_size, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # scaler = torch.cuda.amp.GradScaler()  # For mixed-precision training
    scaler = torch.amp.GradScaler('cuda')  # For mixed-precision training
    
    # Measure training wall time
    dist.barrier()  # Synchronize all ranks before timing
    train_start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)
            
            # with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        logger.info(f"GPU[{rank}], Epoch {epoch+1}, Avg Loss: {total_loss / num_batches:.4f}")
    
    dist.barrier()  # Synchronize all ranks after training
    train_end_time = time.time()
    train_wall_time = train_end_time - train_start_time
    
    # Aggregate max training time across ranks
    train_wall_time_tensor = torch.tensor(train_wall_time, dtype=torch.float64).cuda(rank)
    dist.all_reduce(train_wall_time_tensor, op=dist.ReduceOp.MAX)
    train_wall_time_max = train_wall_time_tensor.item()
    
    # Log training time only from rank 0
    if rank == 0:
        finetune_time[0] = train_wall_time_max
        logger.info(f"Training wall time (max across ranks): {train_wall_time_max:.2f} seconds")
    
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
        logger.info(f"GPU[{rank}]: {dataset_name} Accuracy: {correct / total:.4f}")
    
    dist.destroy_process_group()

# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    NUM_CPUs = args.num_cpus
    NUM_GPUs = args.num_gpus
    logger.info("Initializing Spark...")
    # os.environ['PYSPARK_PYTHON'] = '/home/goodh/miniconda3/envs/5003/bin/python'
    # os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/goodh/miniconda3/envs/5003/bin/python'
    spark = init_spark(NUM_CPUs)
    
    # Output directory for Parquet files
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for cached Parquet files
    # cached_data = check_cached_parquet(output_dir)
    cached_data = None
    train_path = test_path = sst2_test_path = train_collection = test_collection = sst2_collection = None
    preprocess_time = 0
    
    if cached_data:
        logger.info("Cached Parquet files found. Skipping data loading and preprocessing...")
        train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection = cached_data
    else:
        # Load and preprocess data
        logger.info("No cached Parquet files found. Running full pipeline...")
        logger.info("Loading data to MongoDB...")
        load_data_time = load_data_to_mongodb(spark)
        logger.info(f"Data loading to MongoDB took {load_data_time:.2f} seconds")
        
        logger.info("Distributed preprocessing and saving to Parquet...")
        train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, preprocess_time = preprocess_data(spark, output_dir)
        logger.info(f"Distributed preprocessing took {preprocess_time:.2f} seconds")
    
    # Run distributed training
    world_size = NUM_GPUs if NUM_GPUs else max(1, torch.cuda.device_count())
    logger.info(f"Using {world_size} GPU(s)")
    
    logger.info("Distributed fine-tuning...")
    import torch.multiprocessing as mp
    finetune_time = torch.zeros(world_size, dtype=torch.float32).share_memory_()
    mp.spawn(
        train_and_evaluate,
        args=(world_size, train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, finetune_time),
        nprocs=world_size,
        join=True
    )
    
    # append results
    result = f"{time.strftime('%Y/%m/%d-%H:%M:%S')}\t{NUM_CPUs}\t\t{NUM_GPUs}\t\t{preprocess_time:.2f}\t\t{finetune_time[0]:.2f}\n"
    logger.info(result)
    with open("out/results.out", "a") as f:
        f.write(result)
    
    spark.stop()