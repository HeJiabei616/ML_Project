import Part2,Part3,Part4,Part5
import codecs
import os
import numpy as np
import time
import threading as Thread

def input_justify(language_list):
    nb = input('Choose a language: ')
    while nb not in language_list:
        nb = input('Please input a valid language: ')
    if nb in language_list:
        print('Please wait for process to finish (Depending on size of data, up to few seconds to minutes)')
    return nb


def main_func(f_in,rvs_emi_dict,tr_d,em_d,labels):

    ######## Main Answer Here #######


    ### Part2 out is dev.p2_out #####
    ### Part3 out is dev.p2_out #####
    ### Part4 out is dev.p2_out #####


    ## Part2 Generate output by emission ##
    Part2.gen_dev_out(rvs_emi_dict, f_in)

    ## Part3 Generate output by viterbi ##
    Part3.rslt_out(f_in,tr_d,em_d,labels)

    ## Part4 Generate output by forward-backward ##
    Part4.rslt_out(f_in,tr_d,em_d,labels)

    ## Part5 Generate output by forward-backward ##
    Part5.rslt_out(f_in,tr_d,em_d,labels)



if __name__ == "__main__":
    ## Define data set contry ##
    cwd = os.getcwd() #Directory to ML_Project
    language_set = ['CN','EN','FR','SG'] #Modify learning language
    #######################
    in_lan = input_justify(language_set)
    ## Listing require parameters ##
    origin_train = cwd+'/'+in_lan+'/train'
    ## Modify train file ##
    Part2.modify_input_set(3, origin_train, Part2.gen_word_label(origin_train))
    ######################
    modified_train = cwd+'/'+in_lan+'/trainb'
    f_in = cwd+'/'+in_lan+'/dev.in'
    labels = ['O', 'B-neutral', 'I-neutral', 'B-positive', 'I-positive', 'B-negative', 'I-negative']
    trans_dict = Part3.gen_transition(modified_train)
    rvs_emi_dict = Part2.gen_emission(modified_train)
    tr_d=Part3.refined_trans_dict(trans_dict,labels)
    em_d=Part3.refined_emi_dict(rvs_emi_dict,labels)

    main_func(f_in,rvs_emi_dict,tr_d,em_d,labels)

