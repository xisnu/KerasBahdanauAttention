from __future__ import print_function
import numpy as np

#Data prepared in word level

root="/media/parthosarothi/OHWR/Dataset/English-French/"
eng_fr=root+"fra.txt"

def pick_random_sentences_frommain(eng_fr,outfile,nbsentences):
    all_lines=[]
    f=open(eng_fr)
    line=f.readline()
    while line:
        all_lines.append(line)
        line=f.readline()
    print("All line gathered")
    f.close()
    rand_indices=np.random.randint(0,len(all_lines),nbsentences)
    outfile=outfile+"_"+str(nbsentences)+".txt"
    f=open(outfile,'w')
    for i in range(nbsentences):
        f.write(all_lines[rand_indices[i]])
    f.close()
    print("%d Random lines are written in %s"%(nbsentences,outfile))

def pick_sequential_sentences_from_main(eng_fr,outfile,nbsentences):
    f = open(eng_fr)
    fout = open(outfile, 'w')
    line = f.readline()
    count=0
    while line:
        fout.write(line)
        line = f.readline()
        count+=1
        if(count>=nbsentences):
            break
    f.close()
    fout.close()
    print("All line gathered")

def read_reduced_corpus(eng_fr_reduced,outfile):
    """
    Reads a reduced version of main corpus file and writes parameters in outputfile
    also creates lexicon file (word level) for source and target language
    :param eng_fr_reduced:
    :return: source_maxtime,target_maxtime, source_tokens, target_tokens
    """
    source_tokens=[]
    target_tokens=[]
    source_times=[]
    target_times=[]
    f=open(eng_fr_reduced)
    line=f.readline()
    while line:
        info=line.strip("\n").split("\t")
        eng=info[0].strip()
        fr=info[1].strip()
        eng_tokens=eng.split()
        fr_tokens=fr.split()
        source_tokens.extend(eng_tokens)
        target_tokens.extend(fr_tokens)
        source_times.append(len(eng_tokens))
        target_times.append(len(fr_tokens))
        line=f.readline()
    f.close()
    print("All lines processed")
    source_tokens=list(set(source_tokens))
    target_tokens=list(set(target_tokens))
    f=open(outfile,'w')
    f.write("source_tokens,"+str(len(source_tokens))+"\n")
    f.write("target_tokens," + str(len(target_tokens)) + "\n")
    f.write("source_time," + str(max(source_times)) + "\n")
    f.write("target_time," + str(max(target_times)) + "\n")
    f.close()
    f=open("Source_Words.txt","w")
    for s in source_tokens:
        f.write(s+"\n")
    f.close()
    f = open("Target_Words.txt", "w")
    for t in target_tokens:
        f.write(t + "\n")
    f.close()
    print('Configuration and Lexicon files ready')

def read_config_file(config):
    """
    Reads the config file and returns a dictionary
    :param config:
    :return:dictionary
    """
    dict={}
    f=open(config)
    line=f.readline()
    while line:
        info=line.strip("\n").split(",")
        dict[info[0]]=int(info[1])
        line=f.readline()
    f.close()
    return dict

def load_lexicon(lexfile):
    """
    Read a lexfile containing tokens (words)
    :param lexfile:
    :return: a list of tokens
    """
    char_int=[]
    char_int.append("<s>")#start character
    f=open(lexfile)
    line=f.readline()
    while line:
        token=line.strip("\n")
        char_int.append(token)
        line=f.readline()
    f.close()
    char_int.append("<e>")#end character
    char_int.append("<p>")#pad character
    return char_int

def string_to_padded_one_hot(string,char_int,padlength,put_marker=False,align='c'):
    if(put_marker):
        if(align=='r'):
            string=string+" <e>"
            padlength=padlength-1
        elif(align=='l'):
            string = "<s> " + string
            padlength=padlength-1
        else:
            string = "<s> " + string + " <e>"
    print("converting %s to one hot" % string)
    lexsize=len(char_int)
    tokens=string.split()
    nb_tokens=len(tokens)
    one_hot=np.zeros([padlength,lexsize])
    for n in range(nb_tokens):
        t=tokens[n]
        t_index=char_int.index(t)
        one_hot[n][t_index]=1
    pad_index=char_int.index('<p>')
    for r in range(padlength-nb_tokens):
        one_hot[r][pad_index]=1
    return one_hot

def prepare_data(corpus,lexfile,config):
    attrs=read_config_file(config)
    source_time=attrs['source_time']
    target_time=attrs['target_time']+2
    print('Source time %d , target time %d'%(source_time,target_time))
    eng_char_int=load_lexicon(lexfile[0])
    fr_char_int = load_lexicon(lexfile[1])
    f=open(corpus)
    line=f.readline()
    encoder_input=[]
    decoder_input=[]
    decoder_target=[]
    while line:
        info=line.strip("\n").split("\t")
        eng=info[0]
        fr=info[1]
        e_1h=string_to_padded_one_hot(eng,eng_char_int,source_time,put_marker=False)
        encoder_input.append(e_1h)
        f_l_1h=string_to_padded_one_hot(fr,fr_char_int,target_time,put_marker=True,align='l')
        decoder_input.append(f_l_1h)
        f_r_1h = string_to_padded_one_hot(fr, fr_char_int, target_time, put_marker=True, align='r')
        decoder_target.append(f_r_1h)
        print("Processed line ",line)
        line=f.readline()
    f.close()
    print("Encoder Input Shape ",len(encoder_input),encoder_input[0].shape)
    print("Decoder Input Shape ", len(decoder_input),decoder_input[0].shape)
    print("Decoder Input Shape ", len(decoder_target),decoder_target[0].shape)
    source_time,source_tokens=encoder_input[0].shape[0],encoder_input[0].shape[1]
    target_time, target_tokens = decoder_input[0].shape[0], decoder_input[0].shape[1]
    f = open(config+"_extended", 'w')
    f.write("source_tokens," + str(source_tokens) + "\n")
    f.write("target_tokens," + str(target_tokens) + "\n")
    f.write("source_time," + str(source_time) + "\n")
    f.write("target_time," + str(target_time) + "\n")
    f.close()
    print("Modified parameters written")
    encoder_input=np.asarray(encoder_input)
    decoder_input=np.asarray(decoder_input)
    decoder_target=np.asarray(decoder_target)
    return encoder_input,decoder_input,decoder_target

def one_hot_to_string(onehot,char_int):
    string=""
    for vec in onehot:
        index=np.argmax(vec)
        char=char_int[index]
        #print(char)
        if(char!="<e>")and(char!="<s>")and(char!="<p>"):
            string+=char+" "
    return string

lexfiles=["Source_Words.txt","Target_Words.txt"]

#pick_random_sentences_frommain(eng_fr,root+"Test_1",1500)
read_reduced_corpus("time_corpus",'Config_v2')
#prepare_data(root+"Test_1_25.txt",lexfiles,"Config_v2.txt")
#pick_sequential_sentences_from_main(eng_fr,root+"/Corpus_1500",1500)
#read_reduced_corpus(root+"Corpus_1500",'Config_1500')
#prepare_data(root+"Corpus_1500",lexfiles,'Config_1500')