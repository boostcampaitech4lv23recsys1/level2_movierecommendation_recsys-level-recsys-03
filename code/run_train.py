import argparse
import os

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from datasets import SASRecDataset,BERT4RecDataset
from models import S3RecModel, BERT4RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # attribute json
    parser.add_argument("--attjson", default="Ml_item2attributes.json", type=str)

    

    # model args
    parser.add_argument("--model", default="SASRec", type=str) ##### 수정
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mask_prob", type=float, default=0.15, help="mask prob") #### 수정

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--using_pretrain", action="store_true")
    # pretrained .pt file name
    parser.add_argument("--pretrain_name", default="Pretrain.pt", type=str)


    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.attjson

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    
    # model 선택
    if args.model == "SASRec":
        model = S3RecModel(args=args)
        
        train_dataset = SASRecDataset(args, user_seq, data_type="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )
        
    elif args.model == "BERT4Rec":
        model = BERT4RecModel(args=args)
        
        df = pd.read_csv(args.data_file)
        item_ids = df['item'].unique()
        user_ids = df['user'].unique()
        num_item, num_user = len(item_ids), len(user_ids)
        # re_users = user_seq
        mask_users = []
        print("마스킹 중...")
        for user in tqdm(np.arange(num_user)):
            seq = user_seq[user]
            tokens = []

            for s in seq[:-3]:
                prob = np.random.random() # TODO1: numpy를 사용해서 0~1 사이의 임의의 값을 샘플링하세요.
                if prob < args.mask_prob:
                    prob /= args.mask_prob

                    # BERT 학습
                    if prob < 0.8:
                        # masking
                        tokens.append(args.mask_id)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index

                    else:
                        tokens.append(s)
                else:
                    tokens.append(s)

                    
            for s in seq[-3:]:
                tokens.append(s)
            mask_users.append(tokens)


        train_dataset = SASRecDataset(args, mask_users, data_type="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )
        
        
    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )
        

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )


    print(args.using_pretrain)
    if args.using_pretrain:
        pretrained_path = os.path.join(args.output_dir, args.pretrain_name)
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as {args.model}")
    else:
        print(f"Not using pretrained model. The Model is same as {args.model}")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)


if __name__ == "__main__":
    main()
