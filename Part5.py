import codecs
import Part2,Part3
import numpy as np
import operator
import copy
import Part3,Part4

def mix_method(emi_dict,trans_dict,sentence,labels):
    #Defined required var#
    obs_data = sentence.split()
    N = len(obs_data)

    #Main#
    viterbi_label = Part3.viterbi(trans_dict,emi_dict,sentence,labels)
    fb_label = Part4.forward_backward(emi_dict,trans_dict,sentence,labels)

    final_labels = viterbi_label # Us
    for index in range (N):
        c_vb_lb = viterbi_label[index] # Current viterbi label
        c_fb_lb = fb_label[index] # Current forward-backward label
        if c_vb_lb == 'O' and c_fb_lb != 'O': # Check if current viterbi is O and fb is some other
            # If so
            fb_likely = False  # ini flag
            # Loop all previous index for viterbi
            for p_index in range(index):
                # Check if all previous viterbi output are all 'O'
                if viterbi_label[p_index] != 'O':
                    # If got some non-'O' entity which indicate viterbi would probably working, set flag to false, break
                    fb_likely = 'False'
                    break
                # If all viterbi out are 'O' then set flag to true
                fb_likely = True
            if fb_likely == True:
                # Start to adapt fb output if fb flag set to true
                if c_fb_lb[0] == 'I':
                    # Correct fb entity since FB are more sentiment driven (since all 'O' before then should be 'B' here)
                    c_fb_lb = 'B-'+c_fb_lb[2:]
                # Write to final output
                final_labels[index] = c_fb_lb

    return final_labels

def rslt_out(file_path,trans_dict,emi_dict,labels):
    fs = codecs.open(file_path, 'r', 'UTF-8')
    lines = fs.readlines()
    fo = codecs.open(file_path[:-2] + 'p5_out', 'w', 'UTF-8')
    sentence = ""
    for line in lines:
        if line != '\n':
            data = line.rstrip('\n')
            sentence+=data+" "
        if line == '\n':
            # a sentence end, remove last blank
            sentence=sentence[:-1]
            sentence_words=sentence.split(' ')
            rslt_for_sentence = mix_method(emi_dict,trans_dict,sentence,labels)
            for i in range(len(sentence_words)):
                # formulate word and label
                current_word=sentence_words[i]
                current_label=rslt_for_sentence[i]
                word_with_label=current_word+' '+current_label+'\n'
                fo.write(word_with_label)
            # Reset sentence to null and write a '\n' to slice sentences
            fo.write('\n')
            sentence=""
    print("Part5 output done!")

