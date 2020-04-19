import viterbi_algorithm  as v
import prob_functions as pf
import numpy as np
import nltk
#nltk.download('treebank')
#nltk.download('universal_tagset')
tagged_sentences = nltk.corpus.treebank.tagged_sents(tagset='universal')


print("Number of Tagged Sentences ",len(tagged_sentences))
tagged_words=[tup for sent in tagged_sentences for tup in sent]
print("\nTotal Number of Tagged words", len(tagged_words))
vocab=set([word for word,tag in tagged_words])
print("\nVocabulary of the Corpus",len(vocab))
tags=set([tag for word,tag in tagged_words])
print("\nNumber of Tags in the Corpus ",len(tags))

vocab = sorted(list(vocab))
states = list(tags)

print("\nTags:",states)

initial_probs = pf.get_initial_probs(states,tagged_sentences)
probs = pf.get_probs(states,vocab,tagged_sentences)

import random
ob = random.choice(tagged_sentences)
obs,ob_tags = pf.get_observation(ob)

cache,l = v.viterbi(states,probs,initial_probs,obs)
print("\nACTUAL TAGS-",ob_tags,sep="\n")
best_seq,_ = v.viterbi_backward(states,cache,l)
print("\nTAGS ACHIEVED USING HIDDEN MARKOV MODEL-",best_seq,sep="\n")
print()
