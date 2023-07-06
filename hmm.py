import string
import numpy as np
import pandas as pd
from hmmlearn import hmm
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# run program with command:
# python hmm.py BBM411_Assignment2_Q3_TrainingDataset.txt tp53.txt
header_list = []
sequence_list_initial = []
SS_list_initial = []

file_train = sys.argv[1]
file_seq = sys.argv[2]
def readFile(filename):
    with open(filename) as infile:
        counter = -1
        for line in infile:
            counter += 1
            if line.startswith(">"):
                header_list.append(line[1:])
            if counter % 3 == 1:
                sequence_list_initial.append(line)
            if counter % 3 == 2:
                SS_list_initial.append(line)
def readSeq(filename):
    with open(filename) as infile:
        counter = -1
        for line in infile:
            counter += 1
            if counter % 3 == 1:
                return line

readFile(file_train)

# ------- DATA PREPROCESSING -------

# helix: G, H and I; sheet/strand: B and E; turn/coil: T, S, L
# I will use H as "helix", S as "sheet/strand" and T as "turn/coil" states.
SS_list = []
sequence_list = []
def remove_underscore(ss, seq):
    df = pd.DataFrame()
    df['SS_list'] = list(ss)
    df['Sequences'] = list(seq)
    df = df[df.SS_list != "_"]
    df.reset_index(inplace=True)
    ss = ''.join(map(str, df['SS_list']))
    seq = ''.join(map(str, df['Sequences']))
    SS_list.append(ss)
    sequence_list.append(seq)


for j in range(0, len(SS_list_initial)):
    remove_underscore(SS_list_initial[j], sequence_list_initial[j])

for i in range(0, len(SS_list)):
    SS_list[i] = SS_list[i].replace("G", "H")
    SS_list[i] = SS_list[i].replace("I", "H")
    SS_list[i] = SS_list[i].replace("S", "T")
    SS_list[i] = SS_list[i].replace("L", "T")
    SS_list[i] = SS_list[i].replace("B", "S")
    SS_list[i] = SS_list[i].replace("E", "S")


def clear_empty_characters(string):
    return "".join([c for c in string if c.strip()])


def capitalize_string(string):
    return string.upper()


for i in range(0, len(SS_list)):
    sequence_list[i] = clear_empty_characters(sequence_list[i])
    sequence_list[i] = capitalize_string(sequence_list[i])
    SS_list[i] = clear_empty_characters(SS_list[i])

to_be_deleted_index = []

for i in range(0, len(SS_list)):
    if SS_list[i] == "":
        to_be_deleted_index.append(i)
    if sequence_list[i] == "":
        to_be_deleted_index.append(i)

for t in range(0, len(to_be_deleted_index)):
    del SS_list[to_be_deleted_index[t]]
    del sequence_list[to_be_deleted_index[t]]


# ------- MATRIX COLLECTION BEFORE MODEL -------

def get_letters(lst):
    new_lst = list({i for letter in lst for i in set(letter)})
    return sorted(new_lst)


sequence_letters = get_letters(sequence_list)
ss_letters = get_letters(SS_list)

# print(sequence_letters)
# print(ss_letters)

start_state_matrix = np.zeros(3)
transition_matrix = np.zeros((3, 3))
emission_matrix = np.zeros((3, len(sequence_letters)))

for i in range(0, len(SS_list)):
    for j in range(0, len(SS_list[i]) - 1):
        str_x = SS_list[i][j]
        if str_x == "H":
            index_x = 0
        elif str_x == "S":
            index_x = 1
        elif str_x == "T":
            index_x = 2
        else:
            continue
        str_y = SS_list[i][j + 1]
        if str_y == "H":
            index_y = 0
        elif str_y == "S":
            index_y = 1
        elif str_y == "T":
            index_y = 2
        else:
            continue

        transition_matrix[index_x][index_y] += 1
transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:,None]
transition_matrix = np.log2(transition_matrix)
print("Transition Matrix:")
print(transition_matrix , "\n")

for i in range(0, len(SS_list)):
    seq = sequence_list[i]
    ss = SS_list[i]
    for j in range(0, len(SS_list[i]) - 1):
        str_x = SS_list[i][j]
        if str_x == "H":
            index_x = 0
        elif str_x == "S":
            index_x = 1
        elif str_x == "T":
            index_x = 2
        else:
            continue
        index_y = sequence_letters.index(seq[j])
        emission_matrix[index_x][index_y] += 1
emission_matrix = (emission_matrix + 1) / (np.sum(emission_matrix, axis=1) + len(sequence_letters))[:, None]
emission_matrix = np.log2(emission_matrix)
print("Emission Matrix:")
print(emission_matrix, "\n")
start_letters = []
for i in range(0, len(SS_list)):
    if SS_list[i] != '':
        start_letters.append(SS_list[i][0])

count_H = 0
count_S = 0
count_T = 0
for i in range(0, len(start_letters)):
    if start_letters[i] == "H":
        count_H += 1
    elif start_letters[i] == "S":
        count_S += 1
    elif start_letters[i] == "T":
        count_T += 1
    else:
        continue

counts = [count_H, count_S, count_T]
prob_H, prob_S, prob_T = counts[0] / sum(counts), counts[1] / sum(counts), counts[2] / sum(counts)
probabilities = [prob_H, prob_S, prob_T]
for i in range(0, len(probabilities)):
    start_state_matrix[i] = probabilities[i]

start_state_matrix = np.log2(start_state_matrix)
print("Start State Matrix")
print(start_state_matrix, "\n")

# ------- HMM MODEL -------

model = hmm.CategoricalHMM(3, algorithm="viterbi")
model.startprob_ = start_state_matrix
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

letter_dict = dict(enumerate(string.ascii_uppercase))
letter_dict = {y: x for x, y in letter_dict.items()}

target_seq = str(readSeq(file_seq))
target_ss =  "__HHHH_TTT________HHHHH___SSSHHHSSSHHH__HHHH__HHHHHHHHH_________________________________________________TTTT_SSS_____SSSTTTSSSSTTTTSSSS_____SSSSSS_SSS_____SSSSSSSSSSHHH_______HHHHSSS___SSS____SSSSSS____SSSS_TTT__SSSSSS_____TTTSSSSSSSSS___HHHTTTTTT__SSSSSSSS_SSS_SSSSSSSSSSS___HHHHHHHHHHHHH_______________________________HHH__SSSSSSSSHHHHHHHHHHHHHHHHHHHHHH____________________HHHHHH____________"
df_target = pd.DataFrame()
df_target['seq'] = list(target_seq)
df_target['ss'] = list(target_ss)
df_target = df_target[df_target.ss != "_"]
df_target.reset_index(inplace=True)
ss_target = ''.join(map(str, df_target['ss']))
seq_target = ''.join(map(str, df_target['seq']))

SS_list_numeric = []
sequence_list_numeric = []
target_seq_numeric = None
target_ss_numeric = None
for i in range(0, len(SS_list)):
    SS_list[i] = [letter_dict[x] for x in SS_list[i]]
    sequence_list[i] = [letter_dict[x] for x in sequence_list[i]]
for i in range(0, len(target_seq)):
    target_line_numeric = [letter_dict[x] for x in seq_target]
    target_ss_line_numeric = [letter_dict[x] for x in ss_target]
    target_seq_numeric = target_line_numeric
    target_ss_numeric = target_ss_line_numeric

data = []
for i in range(0, len(sequence_list)):
    sequence_list[i] = np.asarray(sequence_list[i],dtype=int).reshape(-1, 1)
    SS_list[i] = np.asarray(SS_list[i],dtype=int).reshape(-1, 1)
    data.append(sequence_list[i])
    data.append(SS_list[i])

X_test = np.asarray(target_seq_numeric).reshape(-1, 1)
y_test = target_ss_numeric

train_data = np.concatenate(data, dtype=int)
model.fit(train_data)
prediction = model.predict(X_test,len(X_test))
prediction = list(prediction)

for i in range(0, len(prediction)):
    if prediction[i] == 0:
        prediction[i] = "H"
    if prediction[i] == 1:
        prediction[i] = "S"
    if prediction[i] == 2:
        prediction[i] = "T"

for i in range(0, len(y_test)):
    if y_test[i] == 7:
        y_test[i] = "H"
    if y_test[i] == 18:
        y_test[i] = "S"
    if y_test[i] == 19:
        y_test[i] = "T"

#print("Predictions = ", "\n", prediction)
#print(y_test)
print("Accuracy Score: ", accuracy_score(y_test,prediction), "\n")
report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
print("Report:")
print(report_df, "\n")
conf_matrix = confusion_matrix(y_test, prediction)
print("Confusion Matrix:")
print(conf_matrix, "\n")