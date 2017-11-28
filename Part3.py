import codecs
import Part2
import numpy as np
import operator
import copy

def gen_transition(file_path):
    fp = codecs.open(file_path,'r',"UTF-8")
    trans_count_dict = {}
    lines = fp.readlines()
    ps_state = 'START'
    c_state = ''
    for data in lines:
        split_data_lines = data.split(' ')
        if data == '\n':
            c_state = 'STOP'
        else:
            # Parsed current state
            if 'O' in split_data_lines[1]:
                c_state = 'O'
            elif 'B-neutral' in split_data_lines[1]:
                c_state = 'B-neutral'
            elif 'I-neutral' in split_data_lines[1]:
                c_state = 'I-neutral'
            elif 'B-positive' in split_data_lines[1]:
                c_state = 'B-positive'
            elif 'I-positive' in split_data_lines[1]:
                c_state = 'I-positive'
            elif 'B-negative' in split_data_lines[1]:
                c_state = 'B-negative'
            elif 'I-negative' in split_data_lines[1]:
                c_state = 'I-negative'

        tuple_state = (ps_state, c_state)

        if tuple_state not in trans_count_dict.keys():
            trans_count_dict[tuple_state] = 1  # Create Key in trans Dict, Format--{[previous state, current state]:count}
            if data == '\n':
                c_state = 'START'
            ps_state = c_state
        else:
            trans_count_dict[tuple_state] += 1
            if data == '\n':
                c_state = 'START'
            ps_state = c_state
    fp.close()
    tot_count_list = [0] * 9 #[START-,O-,B-n-,I-n-,B-p-,I-p-,B-ne-,I-ne-,STOP-]
    for ckey in trans_count_dict.keys():
        if ckey[0]=='START':
            tot_count_list[0]+=trans_count_dict[ckey]
        if ckey[0]=='O':
            tot_count_list[1]+=trans_count_dict[ckey]
        if ckey[0]=='B-neutral':
            tot_count_list[2]+=trans_count_dict[ckey]
        if ckey[0]=='I-neutral':
            tot_count_list[3]+=trans_count_dict[ckey]
        if ckey[0]=='B-positive':
            tot_count_list[4]+=trans_count_dict[ckey]
        if ckey[0]=='I-positive':
            tot_count_list[5]+=trans_count_dict[ckey]
        if ckey[0]=='B-negative':
            tot_count_list[6]+=trans_count_dict[ckey]
        if ckey[0]=='I-negative':
            tot_count_list[7]+=trans_count_dict[ckey]
        if ckey[0]=='STOP':
            tot_count_list[8]+=trans_count_dict[ckey]
    trans_dict={}
    for mkey in trans_count_dict.keys():
        if mkey[0] == 'START':
            value = trans_count_dict[mkey]/tot_count_list[0]
            trans_dict[mkey] = value
        if mkey[0] == 'O':
            value = trans_count_dict[mkey]/tot_count_list[1]
            trans_dict[mkey] = value
        if mkey[0] == 'B-neutral':
            value = trans_count_dict[mkey]/tot_count_list[2]
            trans_dict[mkey] = value
        if mkey[0] == 'I-neutral':
            value = trans_count_dict[mkey]/tot_count_list[3]
            trans_dict[mkey] = value
        if mkey[0] == 'B-positive':
            value = trans_count_dict[mkey]/tot_count_list[4]
            trans_dict[mkey] = value
        if mkey[0] == 'I-positive':
            value = trans_count_dict[mkey]/tot_count_list[5]
            trans_dict[mkey] = value
        if mkey[0] == 'B-negative':
            value = trans_count_dict[mkey]/tot_count_list[6]
            trans_dict[mkey] = value
        if mkey[0] == 'I-negative':
            value = trans_count_dict[mkey]/tot_count_list[7]
            trans_dict[mkey] = value
        if mkey[0] == 'STOP':
            value = trans_count_dict[mkey]/tot_count_list[8]
            trans_dict[mkey] = value
    return trans_dict

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


def viterbi(trans_dict,emi_dict,obs,labels):
    #Defined required var#
    obs_data = obs.split()
    N = len(obs_data)
    T = len(labels)
    prev = np.zeros((T,N))
    V = np.zeros((T,N))


    for i in range(N):
        if i == 0:
            for j in range(T):
                if obs_data[i] in emi_dict[labels[j]]:
                    V[j][i] = trans_dict['START'][labels[j]]*emi_dict[labels[j]][obs_data[i]]
                elif obs_data[i] not in emi_dict[labels[j]]:
                    V[j][i] = trans_dict['START'][labels[j]]*emi_dict[labels[j]]['#UNK#']
        else:
            for j in range(T):
                score = []
                for k in range(T):
                    score.append(np.multiply(trans_dict[labels[k]][labels[j]], V[k][i-1]))
                if obs_data[i] in emi_dict[labels[j]]:
                    score = np.multiply(score, emi_dict[labels[j]][obs_data[i]])
                elif obs_data[i] not in emi_dict[labels[j]]:
                    score = np.multiply(score, emi_dict[labels[j]]['#UNK#'])
                V[j][i] = max(score)
                prev[j][i]=np.argmax(score)

    judge_case=[]
    reverse_out_index_list=[]
    for j in range(T):
        judge_case.append(np.multiply(V[j][-1],trans_dict[labels[j]]['STOP']))
    fi_last = int(np.argmax(judge_case))
    reverse_out_index_list.append(fi_last)
    for i in range(N-1,0,-1):
        fi_last=prev[fi_last][i]
        fi_last = int(fi_last)
        reverse_out_index_list.append(fi_last)

    out_index_list = reverse_out_index_list[::-1]
    labels_out=[]
    for index in out_index_list:
        labels_out.append(labels[index])

    return labels_out


def rslt_out(file_path,trans_dict,emi_dict,labels):
    fs = codecs.open(file_path, 'r', 'UTF-8')
    lines = fs.readlines()
    fo = codecs.open(file_path[:-2] + 'p3_out', 'w', 'UTF-8')
    sentence = ""
    for line in lines:
        if line != '\n':
            data = line.rstrip('\n')
            sentence+=data+" "
        if line == '\n':
            # a sentence end, remove last blank
            sentence=sentence[:-1]
            sentence_words=sentence.split(' ')
            rslt_for_sentence = viterbi(trans_dict,emi_dict,sentence,labels)
            for i in range(len(sentence_words)):
                # formulate word and label
                current_word=sentence_words[i]
                current_label=rslt_for_sentence[i]
                word_with_label=current_word+' '+current_label+'\n'
                fo.write(word_with_label)
            # Reset sentence to null and write a '\n' to slice sentences
            fo.write('\n')
            sentence=""
    print("Part3 output done!")






