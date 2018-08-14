from __future__ import print_function
from Model import *
corpus="time_corpus"
encoder_input, decoder_input, decoder_target = prepare_data(corpus, lexfiles, 'Config_v2')
attrs=read_config_file('Config_v2_extended')
stime=attrs['source_time']
ttime=attrs['target_time']
stoken=attrs['source_tokens']
ttoken=attrs['target_tokens']

def train(mode):
    encoder_input = np.asarray(encoder_input)
    decoder_target = np.asarray(decoder_target)
    nw=BadhanauNMT(stime,ttime,stoken,ttoken,32,mode)
    nw.create_main()
    nw.create_trainer()
    nw.train([encoder_input,decoder_input],decoder_target,25,8)

def test(string,mode):
    nw = BadhanauNMT(stime, ttime, stoken, ttoken, 32, mode)
    nw.create_main()
    nw.create_translator()
    nw.predict(string,lexfiles)

mode='load'
#train(mode)
test("w2 m3 t1",mode)
