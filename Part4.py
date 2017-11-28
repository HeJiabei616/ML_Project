import codecs
import Part2,Part3
import numpy as np
import operator
import copy

def refined_trans_dict(tr_d,klabels):
    r_trans_dict={}
    mylabels = copy.deepcopy(klabels)
    mylabels.insert(0,'START')
    mylabels.insert(len(mylabels),'STOP')
    for o_label in mylabels:
        r_trans_dict[o_label]={}
        for i_label in mylabels:
            key = (o_label,i_label)
            if key in tr_d.keys():
                r_trans_dict[o_label][i_label]=tr_d[key]
            else:
                r_trans_dict[o_label][i_label]=0
    return r_trans_dict



def refined_emi_dict(o_em_d,labels):
    r_emi_dict={}
    lb_index=0
    for o_label in labels:
        r_emi_dict[o_label]={}
        for i_word in o_em_d.keys():
            r_emi_dict[o_label][i_word]=o_em_d[i_word][lb_index]
        lb_index+=1

    return r_emi_dict



def forward_backward(emi_dict,trans_dict,obs,labels):

    #Defined required var#
    obs_data = obs.split()
    obs_len = len(obs_data)
    lb_len = len(labels)

    #ini alpha
    alpha = np.zeros((lb_len, obs_len))

    for j in range (obs_len):
        if j==0:
            for u in range(lb_len):
                label_u = labels[u]
                alpha[u][j]=trans_dict['START'][label_u]
        else:
            for u in range(lb_len):
                label_u=labels[u]
                for v in range(lb_len):
                    label_v=labels[v]
                    if obs_data[j] in emi_dict[label_v].keys():
                        alpha[u][j] += alpha[v][j-1]*trans_dict[label_v][label_u]*emi_dict[label_v][obs_data[j]]
                    else:
                        alpha[u][j] += alpha[v][j-1]*trans_dict[label_v][label_u]*emi_dict[label_v]['#UNK#']


    #ini alpha
    beta = np.zeros((lb_len, obs_len))

    for j in range (obs_len-1,-1,-1):
        if j==obs_len-1:
            for u in range(lb_len):
                label_u = labels[u]
                if obs_data[j] in emi_dict[label_u].keys():
                    beta[u][j]=trans_dict[label_u]['STOP']*emi_dict[label_u][obs_data[j]]
                else:
                    beta[u][j]=trans_dict[label_u]['STOP']*emi_dict[label_u]['#UNK#']
        else:
            for u in range(lb_len):
                label_u=labels[u]
                for v in range(lb_len):
                    label_v=labels[v]
                    if obs_data[j] in emi_dict[label_u].keys():
                        beta[u][j] += beta[v][j+1]*trans_dict[label_u][label_v]*emi_dict[label_u][obs_data[j]]
                    else:
                        beta[u][j] += beta[v][j+1]*trans_dict[label_u][label_v]*emi_dict[label_u]['#UNK#']



    rs_path = []

    for j in range(obs_len):
        score = []
        for u in range(len(labels)):
            score.append(alpha[u][j]*beta[u][j])
        slcted_index = np.argmax(score)
        rs_path.append(labels[slcted_index])

    return rs_path

def rslt_out(file_path,trans_dict,emi_dict,labels):
    fs = codecs.open(file_path, 'r', 'UTF-8')
    lines = fs.readlines()
    fo = codecs.open(file_path[:-2] + 'p4_out', 'w', 'UTF-8')
    sentence = ""
    for line in lines:
        if line != '\n':
            data = line.rstrip('\n')
            sentence+=data+" "
        if line == '\n':
            # a sentence end, remove last blank
            sentence=sentence[:-1]
            sentence_words=sentence.split(' ')
            rslt_for_sentence = forward_backward(emi_dict,trans_dict,sentence,labels)
            for i in range(len(sentence_words)):
                # formulate word and label
                current_word=sentence_words[i]
                current_label=rslt_for_sentence[i]
                word_with_label=current_word+' '+current_label+'\n'
                fo.write(word_with_label)
            # Reset sentence to null and write a '\n' to slice sentences
            fo.write('\n')
            sentence=""
    print("Part4 output done!")

