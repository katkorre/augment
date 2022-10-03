tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
# define the number of synthetic samples to generate
n = 10
new_texts = []
new_labels = []

iter = 0
while iter < n:
    # select two random samples from training set
    text1, label1, text2, label2 = get_two_random_samples()
    # create the prompt
    prompt = get_prompt(text1, label1, text2, label2)

    # generate text using GPT-J model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    # the generated output will be in the form "<text> (Sentiment: <label>)"
    data = gen_text.split('\n')[3].strip('Post: ').split('(Label:')
    if len(data) < 2:
        # the format of the response is invalid
        continue

    text = data[0]
    label = data[1].split(')')[0].strip()
    if label not in ['sexist', 'not sexist']:
        # the format of the response is invalid
        continue

    new_texts.append(text)
    new_labels.append(mapping.str2int(label))
    iter += 1


# define the synthetic dataset and save it to disk 
synthetic_ds = Dataset.from_dict({'text': new_texts, 'label': new_labels})
synthetic_ds.save_to_disk('./data/gpt-neo/' + str(n))
