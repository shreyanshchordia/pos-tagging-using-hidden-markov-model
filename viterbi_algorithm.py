import numpy as np
def viterbi(states,prob_dict,initial_probs,_obs):
  '''
  Parameters:
  1. states - list of hidden states ['Hot','Cold','Mild']
  2. prob_dict - tuple that contains the dictionaries transition_probs and emission probs
     example of transition_probs:
     transition_probs = {'Hot|Hot':0.6,'Hot|Mild':0.3,'Hot|Cold':0.1,
                         'Mild|Hot':0.4,'Mild|Mild':0.3,'Mild|Cold':0.2,
                         'Cold|Hot':0.1,'Cold|Mild':0.4,'Cold|Cold':0.5}

     example of emission_probs:
     emission_probs =   {'CasualWear|Hot':0.8,'CasualWear|Mild':0.19,'CasualWear|Cold':0.01,
                         'SemiCasualWear|Hot':0.5,'SemiCasualWear|Mild':0.4,'SemiCasualWear|Cold':0.1,
                         'ApparelWear|Hot':0.01,'ApparelWear|Mild':0.2,'ApparelWear|Cold':0.79}

  3. _obs - list of a sequence of observable states
     example of _obs = ['ApparelWear','CasualWear', 'CasualWear', 'SemiCasualWear']

  Returns:
  1. cache - required for backtracking to achieve the best hidden state sequence
  2. viterbi_list - the index of the max value of this list is required to initiate backward pass of the Viterbi Algorithm
  '''
  # Generate list of sequences
  n = len(states)
  transition_probs = prob_dict[0]
  emission_probs = prob_dict[1]
  viterbi_list = []
  cache = {}

  for state in states:
    viterbi_list.append(initial_probs[state]*emission_probs[_obs[0]+"|"+state])

  for i,ob in enumerate(_obs):
    if i==0: continue
    temp_list = [None]*n
    for j,state in enumerate(states):
      x = -1
      for k,prob in enumerate(viterbi_list):
        val = prob*transition_probs[state+"|"+states[k]]*emission_probs[ob+"|"+state]
        if (x<val):
          x = val
          cache[str(i)+"-"+state] = [states[k],val]
      temp_list[j]= x
    viterbi_list = [x for x in temp_list]
    

  return cache,viterbi_list

def viterbi_backward(states,cache,viterbi_list):
  '''
  Parameters:

  To be used by passing (states , return values of Viterbi Algorithm) as parameters

  1. cache - dictionary that stores state information of algorithm
     example of cache:
     {'1-Hot': ['Cold', 0.015800000000000005], '1-Mild': ['Cold', 0.02528000000000001], '1-Cold': ['Cold', 0.015800000000000005], '2-Hot': ['Hot', 0.007584000000000002], '2-Mild': ['Mild', 0.0014409600000000007], '2-Cold': ['Mild', 0.00010112000000000005]}

  2. viterbi_list - list of numeric values (one corresponding to each state)

  Returns:
  1. best_sequence - list of predicted hidden states 
     example of best_sequence:
     best_sequence = ['Hot','Cold','Cold']...
  
  2. best_sequence_breakdown - list of probabilities at each stage (used for debugging)
     example of best_sequence_breakdown:
     best_sequence_breakdown = [0.5832000000000002, 0.4199040000000001, 0.3023308800000001]
  '''
  num_states = len(states)
  n = len(cache)//num_states
  best_sequence = []
  best_sequence_breakdown=[]
  x = states[np.argmax(np.asarray(viterbi_list))]
  best_sequence.append(x)

  for i in range(n,0,-1):
    val = cache[str(i)+'-'+x][1]
    x = cache[str(i)+'-'+x][0]
    best_sequence = [x] + best_sequence
    best_sequence_breakdown = [val]+best_sequence_breakdown
  
  return best_sequence,best_sequence_breakdown
