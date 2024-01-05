from transformers import CodeGenForCausalLM, AutoTokenizer
import fire

model_name = "./train/saved_model"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi", padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = CodeGenForCausalLM.from_pretrained(model_name)


def generate_test_code(rust_function):
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode the Rust code into a format understandable by the model
    input_ids = tokenizer.encode(rust_function, return_tensors="pt", padding="max_length", max_length=512)

    # Create the attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Use the model to generate test code
    
    generated_ids = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        pad_token_id=tokenizer.pad_token_id, 
        max_length=1024
    )
    
    # Decode the generated code into a readable format
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

def main():
    # example function
    example_rust_function = """
    pub fn stdout_is<T: AsRef<str>>(&self, msg: T) -> &Self {\n        assert_eq!(self.stdout_str(), String::from(msg.as_ref()));\n        self\n    }",
    """

    test_code = generate_test_code(example_rust_function)

    # output the generated test code
    print("generated test code:")
    print(test_code)

# 程序入口
if __name__ == "__main__":
    fire.Fire(main)
