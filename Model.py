from __future__ import print_function
from PSMRNN import *
from keras.layers import Input,LSTM
from keras.models import Model
from ReadData_v2 import *
import matplotlib.pyplot as plt

class BadhanauNMT:
    def __init__(self,source_time,target_time,source_token,target_token,nodes,mode):
        self.source_time=source_time
        self.source_token=source_token
        self.target_time=target_time
        self.target_token=target_token
        self.nodes=nodes
        self.mode=mode
        print("Model prepared")

    def create_main(self):
        self.encoder_input=Input((self.source_time,self.source_token))
        self.decoder_input=Input((self.target_time,self.target_token))
        self.encoder_memory=LSTM(self.nodes,return_sequences=True,name='encoder_memory')
        self.decoder_memory=PAttentionCell_v2(self.nodes,self.nodes,self.target_token,name='decoder_memory')
        self.encoder = self.encoder_memory(self.encoder_input)

    def create_trainer(self):
        print("Creating Trainer")
        self.decoder = self.decoder_memory([self.decoder_input, self.encoder])
        self.model = Model([self.encoder_input, self.decoder_input], self.decoder)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if (self.mode == 'load'):
            self.model.load_weights("Weights")
            print("Weights loaded")
        self.model.summary()

    def create_translator(self):
        print("Creating Translator")
        self.translator_decoder = self.decoder_memory([self.decoder_input,self.encoder],predict=True)
        self.translator = Model([self.encoder_input,self.decoder_input],self.translator_decoder,name="Translator")
        self.translator.load_weights("Weights")
        print("Translator Created. Weights loaded")
        self.translator_alpha = PAttentionCell_v2(self.nodes, self.nodes, self.target_token, name='decoder_memory',return_alphas=True)
        self.probability_translator=self.translator_alpha([self.decoder_input, self.encoder])
        self.probability_model=Model([self.encoder_input,self.decoder_input],self.probability_translator,name="Translator_alpha")
        self.probability_model.load_weights("Weights")
        print("Translator Alpha Created. Weights loaded")


    def train(self,x,y,epochs,batch):
        for e in range(epochs):
            self.model.fit(x,y,batch_size=batch,validation_split=0.3)
            self.model.save_weights("Weights")

    def predict(self,string,lexfiles):
        source_lex=load_lexicon(lexfiles[0])
        tarrget_lex=load_lexicon(lexfiles[1])
        onehot_encoder=np.expand_dims(string_to_padded_one_hot(string,source_lex,self.source_time,put_marker=False),0)
        onehot_decoder=np.expand_dims(string_to_padded_one_hot(" ",tarrget_lex,self.target_time+1,put_marker=True,align='l'),0)
        print("Decoder input shape ",onehot_decoder.shape)
        #self.translator.load_weights("Weights")
        output=self.translator.predict([onehot_encoder,onehot_decoder])
        prob=self.probability_model.predict([onehot_encoder,onehot_decoder])
        outstring=one_hot_to_string(output[0],tarrget_lex)
        print("Output: ",outstring)
        print("Alphas: ",prob.shape)
        self.draw_alpha_map(string,outstring,prob[0])

    def draw_alpha_map(self,instring,outstring,alphas):
        instring=instring.split()
        inlength=len(instring)
        outstring=outstring.split()
        outlength=len(outstring)
        alphas = np.squeeze(alphas, -1)[:outlength,:inlength]
        alphas = np.transpose(alphas,[1,0])
        print(alphas.shape)
        f=plt.figure(figsize=(5,5))
        image=f.add_subplot(1,1,1)
        ih=image.imshow(alphas,interpolation='nearest',cmap='gray')
        image.set_yticks(range(inlength))
        image.set_yticklabels(instring,rotation=45)

        image.set_xticks(range(outlength))
        image.set_xticklabels(outstring)

        image.set_ylabel("Input Sequene")
        image.set_xlabel("Output Sequence")
        plt.show()
