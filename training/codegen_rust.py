"""
torchrun --nproc_per_node=7  ./training/codegen_rust.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import fire
# import torch.distributed as dist
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


#define the gpu backend to launch training
# dist.init_process_group(backend="nccl")

def encode(examples):
    encoded_input = tokenizer(examples['code'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    encoded_labels = tokenizer(examples['test'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    return {
        'input_ids': encoded_input['input_ids'].clone(),
        'attention_mask': encoded_input['attention_mask'].clone(),
        'labels': encoded_labels['input_ids'].clone(),
    }


def prepare_data():
    train_dataset = load_dataset('json', data_files='./training/data/train_dataset.jsonl')['train']
    valid_dataset = load_dataset('json', data_files='./training/data/valid_dataset.jsonl')['train']
    test_dataset = load_dataset('json', data_files='./training/data/test_dataset.jsonl')['train']

    train_dataset = train_dataset.map(encode, batched=True)
    valid_dataset = valid_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(encode, batched=True)

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
    train_dataset, valid_dataset, test_dataset = prepare_data()

    training_args = TrainingArguments(
        output_dir='./results',            
        num_train_epochs=5,                         
        per_device_train_batch_size=8,              
        per_device_eval_batch_size=8,               
        no_cuda=False,  
        warmup_steps=500,                           
        weight_decay=0.01,                          
        logging_dir='./logs',                       
        logging_steps=10,
        local_rank=-1,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    trainer.train()
    trainer.evaluate(test_dataset)

    trainer.save_model("./training/saved_model")

    #generate predictions from test datasets and save them in files
    predictions = generate_test_predictions(model, test_dataset, tokenizer)
    save_predictions_to_jsonl(predictions, './training/data/test_predictions.jsonl')
    torch.cuda.empty_cache()

if __name__ == "__main__":
    fire.Fire(main)