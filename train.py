import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, udf
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField, StringType
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
import os
import uuid
import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf
import pyarrow.parquet as pq
import glob
import logging
import time
import re
import matplotlib.pyplot as plt

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.bert.modeling_bert import BertLayer
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_loss_curves(checkpoint_path, output_dir="plots"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    train_losses = checkpoint.get('train_losses', [])
    imdb_eval_losses = checkpoint.get('imdb_eval_losses', [])
    sst2_eval_losses = checkpoint.get('sst2_eval_losses', [])
    
    if not train_losses:
        logger.warning("No loss data found in checkpoint.")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, imdb_eval_losses, label='IMDB Evaluation Loss', marker='s')
    plt.plot(epochs, sst2_eval_losses, label='SST-2 Evaluation Loss', marker='^')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"loss_curves_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Loss curves saved at {plot_path}")

# Text preprocessing UDF
def create_preprocess_text_udf():
    def preprocess_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # # Remove non-alphanumeric characters (keep spaces)
        # text = re.sub(r'[^a-z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove continuous whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespaces
        text = text.strip()
        return text
    
    return udf(preprocess_text, StringType())

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
        .config("spark.master", f"local[{num_cpus}]") \
        .config("spark.sql.shuffle.partitions", num_cpus) \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/sentiment_db.reviews") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.mongodb.input.partitionerOptions.partitionSizeMB", "256") \
        .getOrCreate() \
        # .config("spark.cores.max", num_cpus) \
        # .config("spark.driver.cores", "2") \
        # .config("spark.executor.cores", str(num_spark_executor_core) if num_spark_executor_core else 4) \
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
        imdb_dataset["train"].to_pandas()[["text", "label"]],  # use 50 for debug
        imdb_dataset["test"].to_pandas()[["text", "label"]]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    imdb_df["source"] = "IMDB"
    imdb_spark_df = spark.createDataFrame(imdb_df).select(col("text"), col("label").cast("integer"), col("source"))
    
    # SST-2 dataset
    logger.info("Loading SST-2 dataset...")
    sst2_dataset = load_dataset("glue", "sst2")
    sst2_df = sst2_dataset["train"].to_pandas()[["sentence", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)  # use 50 for debug
    sst2_df = sst2_df.rename(columns={"sentence": "text"})
    sst2_df["source"] = "SST-2"
    sst2_spark_df = spark.createDataFrame(sst2_df).select(col("text"), col("label").cast("integer"), col("source"))
    
    logger.info("Writing datasets to MongoDB...")
    imdb_spark_df.write.format("mongo").mode("append").save()
    sst2_spark_df.write.format("mongo").mode("append").save()
    return time.time() - start_time, imdb_spark_df, sst2_spark_df

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
def preprocess_data(spark, imdb_spark_df, sst2_spark_df, output_dir, max_length=256, num_samples=None):
    start_time = time.time()

    # Load and preprocess data
    logger.info("Reading data from MongoDB...")
    # raw_df = spark.read.format("mongo").load()
    raw_df = imdb_spark_df.union(sst2_spark_df)
    
    # Apply text preprocessing
    logger.info("Applying text preprocessing (lowercase, remove punctuation, non-alnum, continuous whitespace, strip)...")
    preprocess_udf = create_preprocess_text_udf()
    processed_df = raw_df.withColumn("text_cleaned", preprocess_udf(col("text")))
    processed_df = processed_df.filter(length(col("text_cleaned")) >= 10).drop("text").withColumnRenamed("text_cleaned", "text")
    
    # def save_df_to_text(df, output_path):
    #     logger.info(f"Saving dataset to text file: {output_path}")
    #     pandas_df = df.select("text", "label", "source").toPandas()
    #     with open(output_path, "w") as f:
    #         for _, row in pandas_df.iterrows():
    #             # Write each row as a JSON-like line for readability
    #             record = {
    #                 "text": row["text"],
    #                 "label": int(row["label"]),
    #                 "source": row["source"]
    #             }
    #             f.write(json.dumps(record) + "\n")
    # # Save df as txt
    # save_df_to_text(raw_df.limit(200), os.path.join(output_dir, "raw_df.txt"))
    # save_df_to_text(processed_df.limit(200), os.path.join(output_dir, "processed_df.txt"))
    
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
    if num_samples and num_samples>0: 
        train_df, test_df, sst2_test_df = train_df.limit(num_samples), test_df.limit(int(num_samples*0.2)), sst2_test_df.limit(int(num_samples*0.2)) # for debug, use 50 samples only
        logger.info(f"num_samples[{num_samples}]")
    
    # Save to Parquet with dynamic partitioning
    num_partitions = int(spark.conf.get("spark.sql.shuffle.partitions"))
    logger.info(f"num_partitions {num_partitions}")
    train_path = os.path.join(output_dir, f"train_{uuid.uuid4().hex}")
    test_path = os.path.join(output_dir, f"test_{uuid.uuid4().hex}")
    sst2_test_path = os.path.join(output_dir, f"sst2_{uuid.uuid4().hex}")
    
    _start_time = time.time()
    logger.info(f"Writing Parquet files: train={train_path}, test={test_path}, sst2={sst2_test_path}")
    train_df.select("input_ids", "attention_mask", "label").repartition(num_partitions).write.mode("overwrite").parquet(train_path)
    logger.info(f"{time.time()-_start_time:.4f}s for train_df partition")
    _start_time = time.time()
    test_df.select("input_ids", "attention_mask", "label").repartition(num_partitions).write.mode("overwrite").parquet(test_path)
    logger.info(f"{time.time()-_start_time:.4f}s for test_df partition")
    _start_time = time.time()
    sst2_test_df.select("input_ids", "attention_mask", "label").repartition(num_partitions).write.mode("overwrite").parquet(sst2_test_path)
    logger.info(f"{time.time()-_start_time:.4f}s for sst2_df partition")
    
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
def train_and_evaluate(rank, world_size, train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, finetune_time, batch_size=8, epochs=3, checkpoint_path=None):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12347"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Create datasets
    train_dataset = LazyParquetDataset(train_path, rank, world_size)
    test_dataset = LazyParquetDataset(test_path, rank, world_size)
    sst2_test_dataset = LazyParquetDataset(sst2_test_path, rank, world_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    sst2_test_loader = DataLoader(sst2_test_dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    
    # 计算本地批次数量
    local_train_batCnt, local_test_batCnt, local_sst2_test_batCnt = sum(1 for _ in train_loader), sum(1 for _ in test_loader), sum(1 for _ in sst2_test_loader)  # 统计本地数据加载器的批次数量
    train_batCnt_tensor, test_batCnt_tensor, sst2_test_batCnt_tensor = torch.tensor(local_train_batCnt, dtype=torch.long).cuda(rank), torch.tensor(local_test_batCnt, dtype=torch.long).cuda(rank), torch.tensor(local_sst2_test_batCnt, dtype=torch.long).cuda(rank)
    
    # 确定全局最小批次数量
    dist.all_reduce(train_batCnt_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(test_batCnt_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(sst2_test_batCnt_tensor, op=dist.ReduceOp.MIN)
    global_train_batCnt, global_test_batCnt, global_sst2_test_batCnt = train_batCnt_tensor.item(), test_batCnt_tensor.item(), sst2_test_batCnt_tensor.item()
    logger.info(f"Rank {rank}: Local train batch count = {local_train_batCnt}, Global min train batch count = {global_train_batCnt}")
    logger.info(f"Rank {rank}: Local test batch count = {local_test_batCnt}, Global min test batch count = {global_test_batCnt}")
    logger.info(f"Rank {rank}: Local sst2_test batch count = {local_sst2_test_batCnt}, Global min sst2_test batch count = {global_sst2_test_batCnt}")
    
    
    model = BertForSequenceClassification.from_pretrained(
    	"bert-base-uncased", 
        num_labels=2, 
        hidden_dropout_prob=0.3, 
        attention_probs_dropout_prob=0.3
    ).to(rank)
    # ############## classifier head 微调
    # # 冻结 BERT 编码器
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    # # 仅训练分类头
    # optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=2e-5)
    # ############## 全量微调
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) 
    ############## 最后两层➕classifier head 微调
    # 冻结所有层
    for param in model.bert.parameters():
        param.requires_grad = False
    # 解冻最后的神经网络和分类头
    for param in model.bert.encoder.layer[-1:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Fixed: Pass transformer_auto_wrap_policy as a callable
    auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
        module=module,
        recurse=recurse,
        nonwrapped_numel=nonwrapped_numel,
        transformer_layer_cls={BertLayer}
    )
    
    model = FSDP(
        model,
        device_id=rank,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        ),
        use_orig_params=True
    )
    
    # Optimizer only for trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5
    )
    scaler = torch.amp.GradScaler('cuda')  # For mixed-precision training
    
    train_losses, imdb_eval_losses, sst2_eval_losses, imdb_eval_acc, sst2_eval_acc = [], [], [], [], []
    
    # Measure training wall time
    dist.barrier()  # Synchronize all ranks before timing
    train_start_time = time.time()
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        # Load loss lists if available
        train_losses = checkpoint.get('train_losses', [])
        imdb_eval_losses = checkpoint.get('imdb_eval_losses', [])
        sst2_eval_losses = checkpoint.get('sst2_eval_losses', [])
        if rank==0:
            logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
            logger.info(f"Loaded losses: train={len(train_losses)}, imdb_eval={len(imdb_eval_losses)}, sst2_eval={len(sst2_eval_losses)}")

    # model = DDP(model.to(rank), device_ids=[rank])
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss, num_batches, local_total_loss = 0, 0, 0
        train_loader_iter = iter(train_loader)
        # for batch in train_loader:
        for i in range(global_train_batCnt):
            try:
                batch = next(train_loader_iter)  # 获取下一批数据
            except StopIteration:
                logger.warning(f"Rank {rank}: Reached end of data at batch {i+1}/{global_train_batCnt}")
                break  # 如果数据不足，提前退出
            input_ids = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels = batch["labels"].to(rank)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            # Aggregate loss across GPUs
            loss_tensor = torch.tensor(loss.item()).cuda(rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size
            local_total_loss += loss_tensor.item()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += avg_loss
            num_batches += 1

        avg_epoch_loss = total_loss / num_batches
        local_avg_epoch_loss = local_total_loss / num_batches
        if rank == 0:
            train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1}, Avg Training Loss: {avg_epoch_loss:.4f}")
        logger.info(f"GPU[{rank}], Epoch {epoch+1}, Avg Loss: {local_avg_epoch_loss:.4f}")
        
        # Evaluation phase
        model.eval()
        for dataset_name, loader, eval_losses, eval_acc, global_test_batCnt in [
            ("IMDB Test", test_loader, imdb_eval_losses, imdb_eval_acc, global_test_batCnt),
            ("SST-2 Test", sst2_test_loader, sst2_eval_losses, sst2_eval_acc, global_sst2_test_batCnt)
        ]:
            total_eval_loss = num_eval_batches = correct = total = 0
            test_loader_iter = iter(loader)
            with torch.no_grad():
                # for batch in loader:
                for i in range(global_test_batCnt):
                    try:
                        batch = next(test_loader_iter)  # 获取下一批数据
                    except StopIteration:
                        logger.warning(f"Rank {rank}: Reached end of data at batch {i+1}/{global_test_batCnt}")
                        break  # 如果数据不足，提前退出
                    input_ids = batch["input_ids"].to(rank)
                    attention_mask = batch["attention_mask"].to(rank)
                    labels = batch["labels"].to(rank)
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    # Aggregate evaluation loss
                    loss_tensor = torch.tensor(loss.item()).cuda(rank)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_eval_loss = loss_tensor.item() / world_size
                    total_eval_loss += avg_eval_loss
                    num_eval_batches += 1
                    # Accuracy calculation
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            avg_eval_loss = total_eval_loss / num_eval_batches
            if rank == 0:
                eval_losses.append(avg_eval_loss)
                eval_acc.append(correct / total)
                logger.info(f"Epoch {epoch+1}, {dataset_name} Eval Loss: {avg_eval_loss:.4f}, Accuracy: {correct / total:.4f}")
        model.train()

    dist.barrier()  # Synchronize all ranks after training
    train_end_time = time.time()
    train_wall_time = train_end_time - train_start_time
    
    # Aggregate max training time across GPUs
    train_wall_time_tensor = torch.tensor(train_wall_time, dtype=torch.float64).cuda(rank)
    dist.all_reduce(train_wall_time_tensor, op=dist.ReduceOp.MAX)
    train_wall_time_max = train_wall_time_tensor.item()
    
    # Log training time and save checkpoint (rank 0 only)
    if rank == 0:
        finetune_time[0] = train_wall_time_max
        logger.info(f"Training wall time (max across ranks): {train_wall_time_max:.2f} seconds")
        
        # Save checkpoint
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_bert_finetuned_epoch_{epochs}.pt")
        
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            'imdb_eval_losses': imdb_eval_losses,
            'sst2_eval_losses': sst2_eval_losses,
            'imdb_eval_acc': imdb_eval_acc,
            'sst2_eval_acc': sst2_eval_acc,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    
    model.eval()
    for dataset_name, loader, global_batCnt in [("IMDB Test", test_loader, global_test_batCnt), ("SST-2 Test", sst2_test_loader, global_sst2_test_batCnt)]:
        correct = total = 0
        loader_iter = iter(loader)
        with torch.no_grad():
            # for batch in loader:
            for i in range(global_batCnt):
                try: batch = next(loader_iter)
                except:
                    logger.warning(f"Rank {rank}: Reached end of data at batch {i+1}/{global_batCnt} for {dataset_name}")
                    break
                input_ids = batch["input_ids"].to(rank)
                attention_mask = batch["attention_mask"].to(rank)
                labels = batch["labels"].to(rank)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        correct_tensor, total_tensor = torch.tensor(correct).cuda(rank), torch.tensor(total).cuda(rank)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        if rank==0:
            logger.info(f"{dataset_name} Accuracy: {correct_tensor.item() / total_tensor.item()}")
        logger.info(f"GPU[{rank}]: {dataset_name} Accuracy: {correct / total:.4f}")
    
    dist.destroy_process_group()

# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_train_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoches', type=int, default=10)
    args = parser.parse_args()
    NUM_CPUs = args.num_cpus
    NUM_GPUs = args.num_gpus
    NUM_TRAIN_SAMPLES = args.num_train_samples
    BATCH_SIZE, EPOCH_SIZE = args.batch_size, args.epoches
    CHECKPOINT_PATH = None
    logger.info("Initializing Spark...")
    # os.environ['PYSPARK_PYTHON'] = '/home/goodh/miniconda3/envs/5003/bin/python'
    # os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/goodh/miniconda3/envs/5003/bin/python'
    spark = init_spark(NUM_CPUs)
    
    # Output directory for Parquet files
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for cached Parquet files
    cached_data = None
    # cached_data = check_cached_parquet(output_dir)
    train_path = test_path = sst2_test_path = train_collection = test_collection = sst2_collection = None
    preprocess_time = 0
    
    if cached_data:
        logger.info("Cached Parquet files found. Skipping data loading and preprocessing...")
        train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection = cached_data
    else:
        # Load and preprocess data
        logger.info("No cached Parquet files found. Running full pipeline...")
        logger.info("Loading data to MongoDB...")
        load_data_time, imdb_spark_df, sst2_spark_df = load_data_to_mongodb(spark)
        logger.info(f"Data loading to MongoDB took {load_data_time:.2f} seconds")
        
        logger.info("Distributed preprocessing and saving to Parquet...")
        train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, preprocess_time = preprocess_data(spark, imdb_spark_df, sst2_spark_df, output_dir, num_samples=NUM_TRAIN_SAMPLES)
        logger.info(f"Distributed preprocessing took {preprocess_time:.2f} seconds")
    
    # Run distributed training
    world_size = NUM_GPUs if NUM_GPUs else max(1, torch.cuda.device_count())
    logger.info(f"Using {world_size} GPU(s)")
    
    logger.info("Distributed fine-tuning...")
    import torch.multiprocessing as mp
    finetune_time = torch.zeros(world_size, dtype=torch.float32).share_memory_()
    mp.spawn(
        train_and_evaluate,
        args=(world_size, train_path, test_path, sst2_test_path, train_collection, test_collection, sst2_collection, finetune_time, BATCH_SIZE, EPOCH_SIZE, CHECKPOINT_PATH),
        nprocs=world_size,
        join=True
    )
    
    # Plot loss curves (using the latest checkpoint)
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        latest_checkpoint = max(
            glob.glob(os.path.join(checkpoint_dir, "*.pt")),
            key=os.path.getctime,
            default=None
        )
        if latest_checkpoint:
            plot_loss_curves(latest_checkpoint)
        else:
            logger.warning("No checkpoint found for plotting.")
    
    # append results
    result = f"{time.strftime('%Y/%m/%d-%H:%M:%S')}\tNUM_CPUs[{NUM_CPUs}]\t\tNUM_GPUs[{NUM_GPUs}]\t\tNUM_TRAIN_SAMPLES[{NUM_TRAIN_SAMPLES}]\t\tpreprocess_time[{preprocess_time:.2f} sec]\t\tfinetune_time[{finetune_time[0]:.2f} sec]" + \
        "\n"
    logger.info(result)
    with open("out/results.out", "a") as f:
        f.write(result)
    
    spark.stop()