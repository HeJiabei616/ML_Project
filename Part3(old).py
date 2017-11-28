import codecs
import Part2
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


def get_emi(emi_dict,word):
    if word in emi_dict.keys():
        return emi_dict[word]
    else:
        return emi_dict['#UNK#']


def get_dis(label,emi_dis):
    if label == 'START':
        return 0
    if label == 'O':
        return emi_dis[0]
    if label == 'B-neutral':
        return emi_dis[1]
    if label == 'I-neutral':
        return emi_dis[2]
    if label == 'B-positive':
        return emi_dis[3]
    if label == 'I-positive':
        return emi_dis[4]
    if label == 'B-negative':
        return emi_dis[5]
    if label == 'I-negative':
        return emi_dis[6]
    if label == 'STOP':
        return 0

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


def viterbi(trans_dict,emi_dict,obs_sequence,labels):
    seq_list = obs_sequence.split()
    total_count = len(seq_list)
    labels_size = len(labels)
    unk_dis=emi_dict['#UNK#']
    c_label = 'START'
    full_state=[]
    recur_case = []
    #[Previous label, Target label, Score]

    for i in range(total_count):
        # Predict with a label
        if i==0:
            for j in range(labels_size):
                current_target_label = labels[j]
                trans_key = (c_label, current_target_label)
                if trans_key in trans_dict.keys():
                    trans_prob = trans_dict[trans_key]
                    input_obs_word = seq_list[i]
                    emi_prob = get_dis(current_target_label, get_emi(emi_dict,input_obs_word))
                    score = trans_prob * emi_prob
                    recur_case.append(("START",current_target_label, score))
                else:
                    score = 0
                    recur_case.append(("START",current_target_label, score))

            full_state.append(recur_case)
            print(recur_case)
        elif i>=0:
            recur_case_mirror = []
            for j in range(labels_size):
                target_label = labels[j]
                target_max_list=[]
                for k in recur_case:
                    previous_label=k[1]
                    previous_score=k[2]
                    trans_key = (previous_label,target_label)
                    if trans_key in trans_dict.keys():
                        trans_prob = trans_dict[trans_key]
                        score = previous_score*trans_prob
                        target_max_list.append(score)
                    elif trans_key not in trans_dict.keys():
                        score=0
                        target_max_list.append(score)

                #print(target_max_list)
                max_index, max_value = max(enumerate(target_max_list), key=operator.itemgetter(1))
                del target_max_list[:]
                input_obs_word = seq_list[i]
                emi_prob = get_dis(target_label, get_emi(emi_dict,input_obs_word))
                score=max_value*emi_prob
                recur_case_mirror.append((labels[max_index],target_label,score))
            recur_case = recur_case_mirror
            full_state.append(recur_case)

    e_label = 'STOP'
    recur_case_mirror = []
    target_label = e_label
    target_max_list = []
    for k in recur_case:
        previous_label = k[1]
        previous_score = k[2]
        trans_key = (previous_label, target_label)
        if trans_key in trans_dict.keys():
            trans_prob = trans_dict[trans_key]
            score = previous_score * trans_prob
            target_max_list.append(score)
        elif trans_key not in trans_dict.keys():
            score = 0
            target_max_list.append(score)

    max_index, max_value = max(enumerate(target_max_list), key=operator.itemgetter(1))
    del target_max_list[:]
    score = max_value
    recur_case_mirror.append((labels[max_index],target_label,score))
    recur_case = recur_case_mirror
    full_state.append(recur_case)

    #Integrated track_backward here also can see below
    k = len(full_state)
    #STOP case
    c_label = full_state[k-1][0][1]
    o_p_label = full_state[k-1][0][0]
    result_lst = []
    result_lst.append(c_label)
    for i in range(k-2, -1, -1):
        label_list = full_state[i]
        #print(label_list)
        for label in label_list:
            iclb=label[1]
            nxtpb=label[0]
            if iclb == o_p_label:
                result_lst.insert(0,iclb)
                o_p_label=nxtpb
    result_lst.insert(0,'START')
    return result_lst

def track_backward(full_state):
    k = len(full_state)
    #STOP case
    c_label = full_state[k-1][0][1]
    o_p_label = full_state[k-1][0][0]
    result_lst = []
    result_lst.append(c_label)
    for i in range(k-2, -1, -1):
        label_list = full_state[i]
        #print(label_list)
        for label in label_list:
            iclb=label[1]
            nxtpb=label[0]
            if iclb == o_p_label:
                result_lst.insert(0,iclb)
                o_p_label=nxtpb
    result_lst.insert(0,'START')
    return result_lst

def viterbi_out(file_path,trans_dict,emi_dict,labels):
    fs = codecs.open(file_path, 'r', 'UTF-8')
    lines = fs.readlines()
    fo = codecs.open(file_path + 'vb_out', 'w', 'UTF-8')
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
            # delete 'START' and 'STOP' label
            del rslt_for_sentence[0]
            del rslt_for_sentence[-1]
            for i in range(len(sentence_words)):
                # formulate word and label
                current_word=sentence_words[i]
                current_label=rslt_for_sentence[i]
                word_with_label=current_word+' '+current_label+'\n'
                fo.write(word_with_label)
            # Reset sentence to null and write a '\n' to slice sentences
            fo.write('\n')
            sentence=""
    print("Label done!")

def compute_score(gold_file1,predicted_file2):
    fs1 = codecs.open(gold_file1, 'r', "UTF-8")
    fs2 = codecs.open(predicted_file2, 'r', "UTF-8")
    lines1 = fs1.readlines()
    lines2 = fs2.readlines()
    match_count=0
    gold_count=len(lines1)
    predicted_count=len(lines2)
    index=0
    for data1 in lines1:
        if data1 == lines2[index]:
            match_count+=1
        index+=1

    Precision = match_count/predicted_count
    Recall = match_count/gold_count
    score = 2/((1/Precision)+(1/Recall))
    print("Precision: " + str(Precision))
    print("Recall: " + str(Recall))
    return score




if __name__ == "__main__":
    train_in = '/Users/zhouxuexuan/PycharmProjects/ML_Project/CN/CN/trainb'
    f_in = '/Users/zhouxuexuan/PycharmProjects/ML_Project/CN/CN/dev.in'
    trans_dict = gen_transition(train_in)
    emi_dict = Part2.gen_emission(train_in)
    obs_seq1 = "[ 哈哈 ] [ 哈哈 ] [ 哈哈 ] 收获 颇丰 还有 明年 的 希腊 游 开心 的 不得了"
    obs_seq2 = "# 彭于 晏 # # 方新武 # # 湄公河 行动 # @ 彭于 晏 @ 彭于 晏 又 被 秒 到 [ 摊手 ] [ 摊手 ] [ 摊手 ] [ 摊手 ]"
    obs_seq3 = "我 吃 小丸子"
    labels = ['O','B-neutral','I-neutral','B-positive','I-positive','B-negative','I-negative']
    tr_d=refined_trans_dict(trans_dict,labels)
    em_d=refined_emi_dict(emi_dict,labels)
    print(viterbi(tr_d,em_d,obs_seq3,labels))
    viterbi_out(f_in,trans_dict,emi_dict,labels)

