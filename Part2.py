#coding:utf-8
import codecs
import operator


def modify_input_set(k, file_path, dict):
    #Create UNK Probe List
    unk_list=[]
    for key in dict.keys():
        tot_appr = sum(dict[key])
        if tot_appr < k:
            unk_list.append(key)

    #Modify Based on Probe list
    fs = codecs.open(file_path,'r','UTF-8')
    lines = fs.readlines()
    fo = codecs.open(file_path+'b','w','UTF-8')
    for data in lines:
        if data != '\n':
            split_data_lines = data.split(' ')
            if split_data_lines[0] in unk_list:
                fo.write('#UNK# '+split_data_lines[1])
            else:
                fo.write(data)
        else:
            fo.write(data)
    fs.close()
    fo.close()

def gen_word_label(file_path):
    fp = codecs.open(file_path,'r',"UTF-8")
    word_dict = {}
    lines = fp.readlines()
    for data in lines:
        if data != '\n':
            split_data_lines = data.split(' ')
            if split_data_lines[0] not in word_dict.keys():
                word_dict[split_data_lines[0]] = [0,0,0,0,0,0,0] #Create Key in word_label Dict   Format--{word:[0_count,B_N_count,I_N_count,B_P_count,I_P_count,B_Ne_count,I_Ne_count]}

            if 'O' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[0] += 1
                word_dict[split_data_lines[0]] = c_list
            elif 'B-neutral' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[1] += 1#/B_N_count
                word_dict[split_data_lines[0]] = c_list
            elif 'I-neutral' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[2] += 1#/I_N_count
                word_dict[split_data_lines[0]] = c_list
            elif 'B-positive' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[3] += 1#/B_P_count
                word_dict[split_data_lines[0]] = c_list
            elif 'I-positive' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[4] += 1#/I_P_count
                word_dict[split_data_lines[0]] = c_list
            elif 'B-negative' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[5] += 1#/B_Ne_count
                word_dict[split_data_lines[0]] = c_list
            elif 'I-negative' in split_data_lines[1]:
                c_list = word_dict[split_data_lines[0]]
                c_list[6] += 1#/I_Ne_count
                word_dict[split_data_lines[0]] = c_list
    fp.close()
    return word_dict


def gen_emission(file_path):
    fp = codecs.open(file_path,'r',"UTF-8")
    emi_dict = {}
    lines = fp.readlines()
    o_count = 0
    B_N_count = 0
    I_N_count = 0
    B_P_count = 0
    I_P_count = 0
    B_Ne_count = 0
    I_Ne_count = 0
    for data in lines:
        if data != '\n':
            split_data_lines = data.split(' ')
            if 'O' in split_data_lines[1]:
                o_count+=1
            elif 'B-neutral' in split_data_lines[1]:
                B_N_count+=1
            elif 'I-neutral' in split_data_lines[1]:
                I_N_count+=1
            elif 'B-positive' in split_data_lines[1]:
                B_P_count+=1
            elif 'I-positive' in split_data_lines[1]:
                I_P_count+=1
            elif 'B-negative' in split_data_lines[1]:
                B_Ne_count+=1
            elif 'I-negative' in split_data_lines[1]:
                I_Ne_count+=1
    tot_count = [o_count,B_N_count,I_N_count,B_P_count,I_P_count,B_Ne_count,I_Ne_count]
    #print(tot_count)
    for data in lines:
        if data != '\n':
            split_data_lines = data.split(' ')
            if split_data_lines[0] not in emi_dict.keys():
                emi_dict[split_data_lines[0]] = [0,0,0,0,0,0,0] #Create Key in word_label Dict   Format--{word:[0_count,B_N_count,I_N_count,B_P_count,I_P_count,B_Ne_count,I_Ne_count]}

            if 'O' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[0] += 1/o_count
                emi_dict[split_data_lines[0]] = c_list
            elif 'B-neutral' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[1] += 1/B_N_count
                emi_dict[split_data_lines[0]] = c_list
            elif 'I-neutral' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[2] += 1/I_N_count
                emi_dict[split_data_lines[0]] = c_list
            elif 'B-positive' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[3] += 1/B_P_count
                emi_dict[split_data_lines[0]] = c_list
            elif 'I-positive' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[4] += 1/I_P_count
                emi_dict[split_data_lines[0]] = c_list
            elif 'B-negative' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[5] += 1/B_Ne_count
                emi_dict[split_data_lines[0]] = c_list
            elif 'I-negative' in split_data_lines[1]:
                c_list = emi_dict[split_data_lines[0]]
                c_list[6] += 1/I_Ne_count
                emi_dict[split_data_lines[0]] = c_list
    fp.close()
    #print(emi_dict['#UNK#'])
    return emi_dict

def gen_dev_out(emi_dict,input_file_path):
    #File w,r
    fp = codecs.open(input_file_path, 'r', "UTF-8")
    lines = fp.readlines()
    fo = codecs.open(input_file_path[:-2]+"p2_out", 'w', 'UTF-8')

    #Unk operation
    unk_label=''
    unk_dis_list = emi_dict['#UNK#']
    max_index, max_value = max(enumerate(unk_dis_list), key=operator.itemgetter(1))
    if max_index == 0:
        unk_label = 'O'
    elif max_index == 1:
        unk_label = 'B-neutral'
    elif max_index == 2:
        unk_label = 'I-neutral'
    elif max_index == 3:
        unk_label = 'B-positive'
    elif max_index == 4:
        unk_label = 'I-positive'
    elif max_index == 5:
        unk_label = 'B-negative'
    elif max_index == 6:
        unk_label = 'I-negative'

    #Loop over to add tag
    for data in lines:
        if data != '\n':
            label = ''
            split_data_lines = data.split('\n')
            if split_data_lines[0] in emi_dict.keys():
                label_distribution_list = emi_dict[split_data_lines[0]]
                max_index, max_value = max(enumerate(label_distribution_list), key=operator.itemgetter(1))
                if max_index == 0:
                    label='O'
                elif max_index == 1:
                    label='B-neutral'
                elif max_index == 2:
                    label='I-neutral'
                elif max_index == 3:
                    label='B-positive'
                elif max_index == 4:
                    label='I-positive'
                elif max_index == 5:
                    label='B-negative'
                elif max_index == 6:
                    label='I-negative'
            #UNK case
            elif split_data_lines[0] not in emi_dict.keys():
                label = unk_label #'O'
            fo.write(split_data_lines[0]+' '+label+'\n')
        elif data=='\n':
            fo.write('\n')
    fp.close()
    fo.close()
    print('Part2 output done!')

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
