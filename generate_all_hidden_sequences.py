import pandas as pd
from tabulate import tabulate

def generate_sequence(states,sequence_length):    
    all_sequences = []  
    depth = sequence_length    
    def gen_seq_recur(states,nodes,depth):
        if depth == 0:
            #print nodes
            all_sequences.append(nodes)
        else:
            for state in states:
                temp_nodes = list(nodes)
                temp_nodes.append(state)
                gen_seq_recur(states,temp_nodes,depth-1)
    
    gen_seq_recur(states,[],depth)
                
    return all_sequences

def score_sequences(sequences,initial_probs,transition_probs,emission_probs,obs):
    
    best_score = -1
    best_sequence = None
    
    sequence_scores = []
    for seq in sequences:
        total_score = 1
        total_score_breakdown = []
        first = True
        for i in range(len(seq)):
            state_score = 1
            # compute transitition probability score
            if first == True:
                state_score *= initial_probs[seq[i]]
                # reset first flag
                first = False
            else:  
                state_score *= transition_probs[seq[i] + "|" + seq[i-1]]
            # add to emission probability score
            state_score *= emission_probs[obs[i] + "|" + seq[i]]
            # update the total score
            #print state_score
            total_score *= state_score
            total_score_breakdown.append(total_score)

        if(total_score > best_score):
            best_score = total_score
            best_sequence = total_score_breakdown

        sequence_scores.append(total_score)
        
    return best_sequence,sequence_scores

def pretty_print_probs(distribs):
    
    rows = set()
    cols = set()
    for val in distribs.keys():
        temp = val.split("|")
        rows.add(temp[0])
        cols.add(temp[1])
        
    rows = list(rows)
    cols = list(cols)
    df = []
    for i in range(len(rows)):
        temp = []
        for j in range(len(cols)):
            temp.append(distribs[rows[i]+"|"+cols[j]])
            
        df.append(temp)
        
    I = pd.Index(rows, name="rows")
    C = pd.Index(cols, name="cols")
    df = pd.DataFrame(data=df,index=I, columns=C)
    
    print(tabulate(df, headers='keys', tablefmt='psql'))

def initializeSequences(states,prob_dict,initial_probs,_obs):
    # Generate list of sequences
    transition_probs = prob_dict[0]
    emission_probs = prob_dict[1]
    seqLen = len(_obs)
    seqs = generate_sequence(states,seqLen)
    # Score sequences
    best_seq,seq_scores = score_sequences(seqs,initial_probs,transition_probs,emission_probs,_obs)
    
    return (seqLen,seqs,best_seq,seq_scores)