"""
torchrun --nproc_per_node=7  ./training/codegen_rust.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import fire
# import torch.distributed as dist
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
from transformers import TrainerCallback


#define the gpu backend to launch training
# dist.init_process_group(backend="nccl")

def encode(examples):
    prompt = "generate test code:"
    combined_input = [prompt + src_code for src_code in examples['code']]
    encoded_input = tokenizer(combined_input, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    encoded_labels = tokenizer(examples['test'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    return {
        'input_ids': encoded_input['input_ids'].clone(),
        'attention_mask': encoded_input['attention_mask'].clone(),
        'labels': encoded_labels['input_ids'].clone(),
    }
def encode_src_test(examples):
    combined_input = [src_code + " [SEP] " + test_code for src_code, test_code in zip(examples['code'], examples['test'])]
    encoded_input = tokenizer(combined_input, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    # encoded_input = tokenizer(examples['code']+examples['test'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    encoded_labels = tokenizer(examples['test'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    return {
        'input_ids': encoded_input['input_ids'].clone(),
        'attention_mask': encoded_input['attention_mask'].clone(),
        'labels': encoded_labels['input_ids'].clone(),
    }

#define loading the data
def prepare_data(whether_fuzz_repo,whether_with_test):
    if whether_fuzz_repo == 0:
        train_dataset = load_dataset('json', data_files='./training/data/train_dataset.jsonl')['train']
        valid_dataset = load_dataset('json', data_files='./training/data/valid_dataset.jsonl')['train']
        test_dataset = load_dataset('json', data_files='./training/data/test_dataset.jsonl')['train']
    else:
        train_dataset = load_dataset('json', data_files='./training/data/train_dataset_fuzz.jsonl')['train']
        valid_dataset = load_dataset('json', data_files='./training/data/valid_dataset_fuzz.jsonl')['train']
        test_dataset = load_dataset('json', data_files='./training/data/test_dataset_fuzz.jsonl')['train']
        
    if whether_with_test == 0:
        train_dataset = train_dataset.map(encode, batched=True)
        valid_dataset = valid_dataset.map(encode, batched=True)
        test_dataset = test_dataset.map(encode, batched=True)
    else:
        train_dataset = train_dataset.map(encode_src_test, batched=True)
        valid_dataset = valid_dataset.map(encode_src_test, batched=True)
        test_dataset = test_dataset.map(encode_src_test, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, valid_dataset, test_dataset


def generate_test_predictions(model, test_dataset, tokenizer):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    test_ids = []
    num = 0 #using this to calculate the 
    for meta in test_dataset['test_id']:
        test_ids.append(meta)

    for item in test_dataset:
        input_ids = item['input_ids'].unsqueeze(0).to('cuda:0')
        attention_mask = item['attention_mask'].unsqueeze(0).to('cuda:0')
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": 512,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": False
        }
        with torch.no_grad():
            outputs = model.generate(**generate_args)
        
        decoded_predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        
        test_id = test_ids[num]
        if num < len(test_ids):
            num = num + 1
        #adding the data
        predictions.append({'test_id': test_id, 'test': decoded_predictions})

    return predictions

def save_predictions_to_jsonl(predictions, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + '\n')

#define loss recorder function
class LossRecorder(TrainerCallback):
    def __init__(self, output_dir):
        self.train_losses = []
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.loss_file_path = os.path.join(self.output_dir, "epoch_losses.txt")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.train_losses.append(state.log_history[-1]["loss"])
        # write into file
        with open(self.loss_file_path, "a") as file:
            file.write(f"{state.epoch}: {state.log_history[-1]['loss']}\n")




def train_model_and_get_loss(model, train_dataset, valid_dataset, training_args):
    torch.cuda.empty_cache()
    loss_recorder = LossRecorder(training_args.output_dir)  
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[loss_recorder,EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.evaluate(valid_dataset)
    # trainer.save_model("./train/saved_model")
    trainer.save_model("./train/saved_model_withunittest")
    torch.cuda.empty_cache()
    return loss_recorder.train_losses

#loading the model
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi")
#apply tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()


config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

def main():

    torch.cuda.empty_cache()
    train_dataset_src, valid_dataset_src, test_dataset_src = prepare_data(0,0)
    train_dataset_unittest, valid_dataset_unittest, test_dataset_unittest = prepare_data(0,1)
    train_dataset_fuzz, valid_dataset_fuzz, test_dataset_fuzz = prepare_data(1,1)
    #loss is Cross-Entropy Loss
    training_args_src = TrainingArguments(
        output_dir='./results',  
        # output_dir='./results_with_test',  
        # output_dir='./results_with_fuzz',         
        num_train_epochs=10,    
        learning_rate=5e-5,                   
        per_device_train_batch_size=4,              
        per_device_eval_batch_size=4,               
        no_cuda=False,  
        warmup_steps=500,                           
        weight_decay=0.01,                          
        logging_dir='./logs',                       
        logging_steps=30,
        local_rank=-1,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    training_args_unittest = TrainingArguments( 
        output_dir='./results_with_test',  
        # output_dir='./results_with_fuzz',         
        num_train_epochs=5,                         
        per_device_train_batch_size=8,              
        per_device_eval_batch_size=8,               
        no_cuda=False,  
        warmup_steps=500,                           
        weight_decay=0.01,                          
        logging_dir='./logs',                       
        logging_steps=30,
        local_rank=-1,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    training_args_fuzz = TrainingArguments(
        output_dir='./results_with_fuzz',         
        num_train_epochs=1,                         
        per_device_train_batch_size=8,              
        per_device_eval_batch_size=8,               
        no_cuda=False,  
        warmup_steps=500,                           
        weight_decay=0.01,                          
        logging_dir='./logs',                       
        logging_steps=30,
        local_rank=-1,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    src_losses = train_model_and_get_loss(model, train_dataset_src, valid_dataset_src, training_args_src)
    # src_with_unittest_losses = train_model_and_get_loss(model, train_dataset_unittest, valid_dataset_unittest, training_args_unittest)
    # src_with_fuzz_losses = train_model_and_get_loss(model, train_dataset_fuzz, valid_dataset_fuzz, training_args_fuzz)

    # plt.plot(src_losses,label = "source code")
    # plt.plot(src_with_unittest_losses,label = "source code with UnitTests")
    # plt.plot(src_with_fuzz_losses,label = "source code with Fuzz")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Over Epochs for Different Models")
    # plt.savefig("./training/data/loss_with_3_experienments")
    # plt.show()    
    


    # loss_recorder = LossRecorder()
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     callbacks=[loss_recorder],#EarlyStoppingCallback(early_stopping_patience=3),
    # )

    # trainer.train()
    # trainer.evaluate(test_dataset)

    # trainer.save_model("./train/saved_model")
    # trainer.save_model("./training/saved_model_in_pair")
    # # trainer.save_model("./training/saved_model_with_fuzz")

    # #generate predictions from test datasets and save them in files
    predictions = generate_test_predictions(model, test_dataset_src, tokenizer)
    save_predictions_to_jsonl(predictions, './training/data/test_predictions.jsonl')
    # save_predictions_to_jsonl(predictions, './training/data/test_predictions_in_pair.jsonl')
    # # save_predictions_to_jsonl(predictions, './training/data/test_predictions_in_fuzz.jsonl')
    # torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)
    
#draw a picture of train loss and test loss
#reinforcement learning? maybe another way to work better



#1.how to prompt a function that generate a test code model based on the model
#2.