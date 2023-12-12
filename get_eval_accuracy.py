eval_mycomp_sents = []
# Replace with your file path
with open('eval-data-mycomp.txt', 'r') as file:
    for line in file:
        word_list = eval(line)
        joined_line = ' '.join(word_list)
        eval_mycomp_sents.append(joined_line)

eval_gold_sents = []
with open('eval-data-gold.txt', 'r') as file:
    for line in file:
        eval_gold_sents.append(line)

exact_count = 0
total_sents = 0
for i in range(len(eval_gold_sents)):
    if eval_mycomp_sents[i] == eval_mycomp_sents:
        print(f"My Comp: {eval_mycomp_sents[i]}")
        print(f'Eval: {eval_gold_sents[i]}')
        exact_count += 1
        print(i)
    total_sents += 1

print(f'Exact Compression Count: {exact_count}')
print(f'Accuracy: {exact_count / total_sents}')
