import os
import json
import torch
import argparse
import torch.optim as optim
import transformers

from tqdm import tqdm
from tokenizers import AddedToken

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls

import preprocessing
import schema_item_classifier
import text2sql_data_generator

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 128,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type = str, default = "tensorboard_log/text2sql",
                        help = 'save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type = str, default = "t5-3b",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "data/preprocessed_data/resdsql_train_spider.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument('--tables_for_natsql', type = str, default = "NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help = 'file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--output", type = str, default = "predicted_sql.txt",
                help = "save file of the predicted sqls.")
    
    opt = parser.parse_args()

    return opt

def _test(opt):
    
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    
    if opt.target_type == "natsql":
        tables = json.load(open(opt.tables_for_natsql,'r'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    # initialize model
    model = T5ForConditionalGeneration.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []

    quit = False
    question = ""
    live_id = "perpetrator"

    while(not quit):

        question = input("NL query:")

        batch_inputs = [question]
        batch_db_ids = [live_id]

        tokenized_inputs = tokenizer(
                batch_inputs, 
                return_tensors="pt",
                padding = "max_length",
                max_length = 512,
                truncation = True
            )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]

        if torch.cuda.is_available():
                encoder_input_ids = encoder_input_ids.cuda()
                encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
                model_outputs = model.generate(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    max_length = 256,
                    decoder_start_token_id = model.config.decoder_start_token_id,
                    num_beams = opt.num_beams,
                    num_return_sequences = opt.num_return_sequences
                )

                model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
                if opt.target_type == "sql":
                    predict_sqls += decode_sqls(
                        opt.db_path, 
                        model_outputs, 
                        batch_db_ids, #schema name in list
                        batch_inputs, #input sequence in list
                        tokenizer, 
                        batch_tc_original #table.column (list)
                    )
                elif opt.target_type == "natsql":
                    predict_sqls += decode_natsqls(
                        opt.db_path, 
                        model_outputs, 
                        batch_db_ids, 
                        batch_inputs, 
                        tokenizer, 
                        batch_tc_original, 
                        table_dict
                    )
                else:
                    raise ValueError()
                
                print(predict_sqls)

    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    
    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls)
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
    
        return spider_metric_result["exact_match"], spider_metric_result["exec"]
    
if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        print("Live mode is inference only!")
    elif opt.mode in ["eval", "test"]:
        _test(opt)