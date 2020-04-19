import numpy as np
def get_transition_probs(states,tr_matrix):
  state_dict = {}
  for i,state in enumerate(states):
    state_dict[i]=state
  transition_probs = {}
  for i in range(tr_matrix.shape[0]):
    for j in range(tr_matrix.shape[1]):
      transition_probs[state_dict[j]+'|'+state_dict[i]] = tr_matrix[i][j]
  return transition_probs

def get_tr_matrix(states,tagged_sentences):
	tr_matrix = np.zeros((len(states),len(states)))
	for sentence in tagged_sentences:
		for i in range(len(sentence)):
			if i==0: continue
			tr_matrix[states.index(sentence[i-1][1])][states.index(sentence[i][1])]+=1
	tr_matrix /= np.sum(tr_matrix,keepdims=True,axis=1)
	return tr_matrix

def get_emission_probs(states,vocab,em_matrix):
  state_dict = {}
  for i,state in enumerate(states):
    state_dict[i]=state
  emission_probs = {}
  for i in range(em_matrix.shape[0]):
    for j in range(em_matrix.shape[1]):
      emission_probs[vocab[i]+'|'+state_dict[j]] = em_matrix[i][j]
  return emission_probs

def get_em_matrix(states,vocab,tagged_sentences):
  vocab = sorted(list(vocab))
  em_matrix = np.zeros((len(vocab),len(states)))
  for sentence in tagged_sentences:
    for word,tag in sentence:
      em_matrix[vocab.index(word)][states.index(tag)]+=1
  sum_matrix = np.sum(em_matrix,keepdims=True,axis=0)
  em_matrix = np.divide(em_matrix, sum_matrix, out=np.zeros_like(em_matrix), where=sum_matrix!=0)
  return em_matrix

def get_probs(states,vocab,tagged_sentences):
  tr_matrix = get_tr_matrix(states,tagged_sentences)
  em_matrix = get_em_matrix(states,vocab,tagged_sentences)
  transition_probs = get_transition_probs(states,tr_matrix)
  emission_probs = get_emission_probs(states,vocab,em_matrix)
  return transition_probs,emission_probs

def get_initial_probs(states,tagged_sentences):
  initial_list = [0]*len(states)
  for sent in tagged_sentences:
    tag = sent[0][1]
    initial_list[states.index(tag)]+=1
  initial_list = np.asarray(initial_list)
  initial_list = initial_list/np.sum(initial_list,keepdims=True)
  initial_probs = {}
  for i,state in enumerate(states):
    initial_probs[state] = initial_list[i]
  return initial_probs

def get_observation(sent):
  ob = []
  ob_tags = []
  for word,tag in sent:
    ob.append(word)
    ob_tags.append(tag)
  return ob,ob_tags