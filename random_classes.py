import csv
import random

seed = 1
random.seed(seed)

data = [
    ('Hello!', 1), ('Yes!', 2), ('No.', 3), ('Water', 4), ('Tea', 5), ('Coffee', 6),
    ('Food', 7), ('Up', 8), ('Down', 9), ('Left', 10), ('Right', 11), ('Light', 12),
    ('Dark', 13), ('Head', 14), ('Arm', 15), ('Arms', 16), ('Leg', 17), ('Legs', 18),
    ('Heart', 19), ('Hands', 20), ('Foot', 21), ('Feet', 22), ('Zero', 23), ('One', 24),
    ('Two', 25), ('Three', 26), ('Four', 27), ('Five', 28), ('Six', 29), ('Seven', 30),
    ('Eight', 31), ('Nine', 32), ('Times', 33), ('Stop!', 34), ('Go on.', 35), ('Before', 36),
    ('Now.', 37), ('After', 38), ('Yesterday', 39), ('Today', 40), ('Tomorrow', 41), ('Shower', 42),
    ('Wash', 43), ('Bath', 44), ('Love', 45), ('Daughter', 46), ('Son', 47), ('Partner', 48),
    ('Assistant', 49), ('Nurse', 50), ('Music', 51), ('Drink', 52), ('Jam', 53), ('Yellow', 54),
    ('What?', 55), ('Where?', 56), ('When?', 57), ('Why?', 58), ('Who?', 59), ('How?', 60),
    ('Thank you', 61), ('I need', 62), ('I want', 63), ('I can', 64), ('I can t', 65), ('Sure!', 66),
    ('Joy', 67), ('Help!', 68), ('Quit', 69), ('Hot', 70), ('Cold', 71), ('Egg', 72),
    ('Ribbon', 73), ('Palm', 74)
]

def rndm_nonov_splits(items, chunk_size=13):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

sorted_items = sorted(data, key=lambda x: len(x[0]), reverse=True)

print(sorted_items)

rnd_items = sorted_items.copy()
random.shuffle(rnd_items)

print(rnd_items)

sets = rndm_nonov_splits(rnd_items, chunk_size=13)

last_set = sets[-1]
needed = 13 - len(last_set)
    
previous_items = [item for s in sets[:-1] for item in s]
overlap_samples = random.sample(previous_items, needed)
sets[-1] = last_set + overlap_samples

filename="random_sets.csv"
with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Set_ID', 'Word', 'Word_Length', 'Class_ID'])
        for set_idx, current_set in enumerate(sets, 1):
            for word, class_id in current_set:
                writer.writerow([f"Set_{set_idx}", word, len(word), class_id])

for idx, s in enumerate(sets, 1):
    words = [item[0] for item in s]
    ids = [item[1] for item in s]
    print(f"Set {idx} ({len(s)} elements):")
    print(f"  Parole: {words}")
    print(f"  ID:     {ids}\n")