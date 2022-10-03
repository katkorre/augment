sexist_train_samples = sexist['train'].filter(lambda e: e['label_sexist'] == 1).shuffle(seed=42).select(range(10))
not_train_samples = sexist['train'].filter(lambda e: e['label_sexist'] == 0).shuffle(seed=42).select(range(10))

# map emotions to integers for labeling
def map_label(example):
    if example['label_sexist'] == 0:
        example['label_sexist'] = 0
    elif example['label_sexist'] == 1:
       example['label_sexist'] = 1
    
   

# create a train set that consists of 10 samples per class and filter the test 
# set to contain only the valid labels
sexist_train_ds = concatenate_datasets([sexist_train_samples, not_train_samples]).map(lambda e: map_label(e)).shuffle(seed=42)
sexist_test_ds = sexist["test"].filter(lambda e: e['label_sexist'] in [0,1]).map(lambda e: map_label(e))

# define the maping between emotions and labels
mapping = ClassLabel(names=['sexist', 'not sexist'])
