from __future__ import print_function
import keras
from keras.layers import Layer,RNN
from keras.layers.recurrent import Recurrent
import keras.backend as K
from keras.initializers import RandomNormal

def P_time_distributed_dense(x,w,b):
    """
    :param x: a 3D input sequence (N,T,F)
    :param w: a weight matrix for every timestep
    :param b: a bias for every timestep
    :return: x dot w
    """
    #input_dim=K.shape(x)[-1]
    out=K.dot(x,w) #[N,e_ts,E] X [E,A] = [N,e_ts,A]
    out=K.bias_add(out,b) #[N,e_ts,A]+[A] = [N,e_ts,A]
    return out

class PLSTMCell(Layer):
    def __init__(self,nb_units,**kwargs):
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        super(PLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim=input_shape[-1]
        #self.timesteps=input_shape[-2]
        print("nodes = %d ,input_dim %d"%(self.nodes,self.input_dim))
        #inititate weights for Input gate
        self.Wki=self.add_weight(shape=[self.input_dim,self.nodes],initializer='orthogonal',name='W_ki')
        self.Wri=self.add_weight('W_ri',shape=[self.nodes,self.nodes],initializer='orthogonal')
        # inititate weights for Forget/Reset gate
        self.Wkf = self.add_weight('W_kf', shape=[self.input_dim, self.nodes], initializer='orthogonal')
        self.Wrf = self.add_weight('W_rf', shape=[self.nodes, self.nodes], initializer='orthogonal')
        # inititate weights for Update gate
        self.Wkc = self.add_weight('W_kc', shape=[self.input_dim, self.nodes], initializer='orthogonal')
        self.Wrc = self.add_weight('W_rc', shape=[self.nodes, self.nodes], initializer='orthogonal')
        # inititate weights for Output gate
        self.Wko = self.add_weight('W_ko', shape=[self.input_dim, self.nodes], initializer='orthogonal')
        self.Wro = self.add_weight('W_ro', shape=[self.nodes, self.nodes], initializer='orthogonal')
        # All gate weights initialized
        print("Weights Initiated")

    def call(self, inputs, states, training=None):
        print(states)
        self.h_tm1=states[0]
        self.c_tm1=states[1]
        self.inputs=inputs
        self.call_input_gate()
        self.call_forget_gate()
        self.call_output_gate()
        self.call_update_gate()
        #Now compute Updated C state
        C_reset=self.c_tm1*self.f
        C_new=self.i*self.c
        self.C_out=C_reset+C_new
        #Now compute Updated H state
        self.h_out=K.tanh(self.C_out)*self.o

        return self.h_out,[self.h_out,self.C_out]

    def call_input_gate(self):
        Iiw=K.dot(self.inputs,self.Wki)
        Irw=K.dot(self.h_tm1,self.Wri)
        i=Iiw+Irw
        self.i=K.hard_sigmoid(i)

    def call_forget_gate(self):
        Fiw=K.dot(self.inputs,self.Wkf)
        Frw=K.dot(self.h_tm1,self.Wrf)
        f=Fiw+Frw
        self.f=K.hard_sigmoid(f)

    def call_update_gate(self):
        Ciw=K.dot(self.inputs,self.Wkc)
        Crw=K.dot(self.h_tm1,self.Wrc)
        c=Ciw+Crw
        self.c=K.tanh(c)

    def call_output_gate(self):
        Oiw=K.dot(self.inputs,self.Wko)
        Orw=K.dot(self.h_tm1,self.Wro)
        o=Oiw+Orw
        self.o=K.hard_sigmoid(o)

class PAttentionCell(Recurrent):
    '''
    A=Attention size (nb_units)
    O=Output size (output_dim) Lexicon size
    E=Number of Encoder nodes (input_dim)
    D=Number of Decoder nodes
    e_ts=Encoder timestep
    '''

    def __init__(self,nb_units,output_dim,return_alphas=False,name="Attention",**kwargs):
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        self.output_dim=output_dim
        super(PAttentionCell, self).__init__(**kwargs)
        self.return_sequences=True
        self.return_alphas=return_alphas
        self.weight_constraint=keras.constraints.get(None)
        self.weight_initializer=keras.initializers.get('uniform')
        self.name=name

    def build(self, input_shape):
        self.input_dim=input_shape[-1]
        self.timesteps=input_shape[-2]
        print("nodes = %d ,input_dim %d"%(self.nodes,self.input_dim))
        #Weights for all the gates are initialized here
        '''
        every gate has four weights
            1)Wk=weight for input to gate (kernel)
            2)Wr=weight for recursive connection
            3)b=bias for this gate
            4)C=weight for context vector
        '''
        self.states = [None, None]
        #Weight matrix for initial state calculation
        self.Ws = self.add_weight('Ws',shape=(self.input_dim, self.nodes),initializer=self.weight_initializer)
        #inititate weights for Context vector Ct calculation
        self.Va=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='Va') #[A] to be multiplied with tanh(alignment output)
        self.Wa=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Wa')#[D,A] to be multiplied by Si-1 (Decoder hidden state)
        self.Ua=self.add_weight(shape=[self.input_dim,self.nodes],initializer=self.weight_initializer,name='Ua')#[E,A] to be multiplied with hj (Encoder output/annotation)
        self.ba=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='ba')#[A] Attention bias
        #initiate weights for Reset gate(r)
        self.Wkr=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkr')#[O,D]
        self.Urr=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urr')#[D,D]
        self.br=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='br')#[D]
        self.Cr=self.add_weight(shape=[self.input_dim,self.nodes],initializer=self.weight_initializer,name='Cr')#[E,D]
        # initiate weights for Update gate(z)
        self.Wkz=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkz')#[O,D]
        self.Urz=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urz')#[D,D]
        self.bz = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bz')#[D]
        self.Cz = self.add_weight(shape=[self.input_dim, self.nodes], initializer=self.weight_initializer, name='Cz')#[E,D]
        #initiate weights for Input gate(p)/proposal
        self.Wkp=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkp')#[O,D]
        self.Urp=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urp')#[D,D]
        self.bp = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bp')#[D]
        self.Cp = self.add_weight(shape=[self.input_dim, self.nodes], initializer=self.weight_initializer, name='Cp')#[E,D]
        #Initiate weights for Output gate(o)
        self.Wko=self.add_weight(shape=[self.output_dim,self.output_dim],initializer=self.weight_initializer,name='Wko')#[O,O]
        self.Uro=self.add_weight(shape=[self.nodes,self.output_dim],initializer=self.weight_initializer,name='Uro')#[D,O]
        self.bo = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bo')#[D]
        self.Co = self.add_weight(shape=[self.input_dim, self.output_dim], initializer=self.weight_initializer, name='Co')#[E,O]
        print("Weights Initiated")


    def call(self, input_sequence,mask=None, training=None, initial_state=None):
        #print(states)
        self.input_sequence=input_sequence#[N,e_ts,E]
        self.U_h=P_time_distributed_dense(self.input_sequence,self.Ua,self.ba)#[N,e_ts,A] also E=A (is that required ?)
        return super(PAttentionCell,self).call(input_sequence)

    def step(self, inputs, states):
        ytm,stm=states #states=[N,O],[N,D]
        #enc_hidden=dec_hidden
        #repeat the hidden state over all timesteps of encoder
        self.repeated_state=K.repeat(stm,self.timesteps) #N,e_ts,D
        #multiply the hidden state with Wa
        self._WaStm = K.dot(self.repeated_state,self.Wa) #[N,e_ts,D] X [D,A] = [N,e_ts,A]
        self._WaStm = K.tanh(self._WaStm+self.U_h)#[N,e_ts,A]+[N,e_ts,A] = [N,e_ts,A]
        Va_expanded=K.expand_dims(self.Va) #[A,1]
        self.e_ij = K.dot(self._WaStm,Va_expanded) #[N,e_ts,1]
        self.alpha_ij=K.softmax(self.e_ij,axis=1) #still [N,e_ts,1] one alpha_ij for every combinantion of input pos i and output position j
        #Now calculate context vector
        self.c_t=K.batch_dot(self.alpha_ij,self.input_sequence,axes=1)#[N,e_ts,1] X [N,e_ts,E] = [N,1,E]
        self.c_t=K.squeeze(self.c_t,1)#[N,E]
        self.call_reset_gate(ytm,stm)# get self.r
        self.call_update_gate(ytm,stm)# get self.Z_t
        self.call_input_gate(ytm,stm)#get self.S_t :this is new state from this step
        self.call_output_gate(ytm,stm)#get self.Y_t :this is new output token from this step
        if(self.return_alphas):
            return self.alpha_ij,[self.Y_t,self.S_t]
        else:
            return self.Y_t,[self.Y_t,self.S_t]#[N,O],states=[N,O],[N,D]

    #return self.S_t=[N,D]
    def call_input_gate(self,ytm,stm):#proposal
        r_stm=self.r*stm#[N,D]*[N,D]
        Iiw=K.dot(ytm,self.Wkp)#[N,O] X [O,D] = [N,D]
        Irw=K.dot(r_stm,self.Urp)#[N,D] X [D,D] = [N,D]
        Icw=K.dot(self.c_t,self.Cp)#[N,E] X [E,D] = [N,D]
        I=K.tanh(Iiw+Irw+Icw+self.bp)#[N,D]+[N,D]+[N,D]+[D] = [N,D]
        #new updated state
        self.S_t=((1-self.Z_t)*stm)+(self.Z_t*I)#[N,D]
        print("Proposal gate output ", K.int_shape(self.S_t))

    # self.r=returns [N,D]
    def call_reset_gate(self,ytm,stm):
        Riw=K.dot(ytm,self.Wkr)#[N,O] X [O,D] = [N,D] #previous predicted token
        Rrw=K.dot(stm,self.Urr)#[N,D] X [D,D] = [N,D]
        Rcw=K.dot(self.c_t,self.Cr)#[N,E] X [E,D] = [N,D]
        R=Riw+Rrw+Rcw+self.br#[N,D]+[N,D]+[D] = [N,D]
        self.r=K.sigmoid(R)#[N,D]
        print("Reset/Forget gate output ",K.int_shape(self.r))

    #returns self.Z_t=[N,D]
    def call_update_gate(self,ytm,stm):
        Ziw=K.dot(ytm,self.Wkz)#[N,O] X [O,D] = [N,D] #previous predicted token
        Zrw=K.dot(stm,self.Urz)#[N,D] X [D,D] = [N,D]
        Zcw=K.dot(self.c_t,self.Cz)#[N,E] X [E,D] =[N,D]
        z=Ziw+Zcw+Zrw+self.bz#[N,D]+[N,D]+[D] = [N,D]
        self.Z_t=K.sigmoid(z)#[N,D]
        print("Update gate output ", K.int_shape(self.Z_t))

    #return self.Y_t = [N,O]
    def call_output_gate(self,ytm,stm):
        Oiw=K.dot(ytm,self.Wko)#[N,O] X [O,O] = [N,O]
        Orw=K.dot(stm,self.Uro)#[N,D] X [D,O] = [N,O]
        Ocw=K.dot(self.c_t,self.Co)#[N,E] X [E,O] = [N,O]
        self.Y_t=K.sigmoid(Oiw+Orw+Ocw)#[N,O]
        print("Output gate output ", K.int_shape(self.Y_t))#[N,O] ?

    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = K.tanh(K.dot(inputs[:, 0], self.Ws))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.nodes,
            'return_probabilities': self.return_alphas
        }
        base_config = super(PAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_alphas:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

class PLSTM(RNN):
    def __init__(self,nodes,return_sequences=True,return_state=True,go_backwards=False,stateful=False,unroll=False,input_dim=None):
        if(input_dim is not None):
            cell=PLSTMCell(input_dim)
        else:
            cell=PLSTMCell(nodes)
        super(PLSTM,self).__init__(cell,return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(PLSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

class PSMLSTM(Recurrent):
    def __init__(self,nodes,name,return_sequence=False,initializer='uniform',**kwargs):
        self.nodes=nodes
        self.name=name
        self.return_sequences=return_sequence
        self.initializer=keras.initializers.get(initializer)
        super(PSMLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building:..Input shape ",input_shape)
        self.input_dim=input_shape[-1]
        self.input_time=input_shape[-2]
        self.states=[None,None]
        #Initial state weights
        self.Wsm1=self.add_weight('Wir',(self.input_dim,self.nodes),initializer=self.initializer)
        #Reset / Forget gate weights
        self.Wir=self.add_weight('Wir',(self.input_dim,self.nodes),initializer=self.initializer)
        self.Wrr=self.add_weight('Wrr',(self.nodes,self.nodes),initializer=self.initializer)
        self.br=self.add_weight('br',shape=[self.nodes],initializer=self.initializer)
        #Update gate weights
        self.Wiu=self.add_weight('Wiu',(self.input_dim,self.nodes),initializer=self.initializer)
        self.Wru = self.add_weight('Wru', (self.nodes, self.nodes), initializer=self.initializer)
        self.bu=self.add_weight('bu',[self.nodes],initializer=self.initializer)
        #Output gate weights
        self.Wio = self.add_weight('Wio', (self.input_dim, self.nodes), initializer=self.initializer)
        self.Wro = self.add_weight('Wro', (self.nodes, self.nodes), initializer=self.initializer)
        self.bo = self.add_weight('bo', [self.nodes], initializer=self.initializer)
        # Input gate weights
        self.Wii = self.add_weight('Wii', (self.input_dim, self.nodes), initializer=self.initializer)
        self.Wri = self.add_weight('Wri', (self.nodes, self.nodes), initializer=self.initializer)
        self.bi = self.add_weight('bi', [self.nodes], initializer=self.initializer)
        print("Build:All weights initialized for layer ",self.name)

    def reset_gate(self,xt,htm):
        Ri=K.dot(xt,self.Wir)#[N,I] x [I,H] = [N,H]
        Rr=K.dot(htm,self.Wrr)#[N,H] x [H,H] = [N,H]
        #print("reset: Inputs", K.int_shape(Ri),K.int_shape(Rr),K.int_shape(self.br))
        R=Ri+Rr+self.br#[N,H]+[N,H]+[H] = [N,H]
        return K.sigmoid(R)

    def input_gate(self,xt,htm):
        Ii=K.dot(xt,self.Wii)#[N,I] x [I,H] = [N,H]
        Ir = K.dot(htm, self.Wri)  #[N,H] x [H,H] = [N,H]
        I=Ii+Ir+self.bi #[N,H]+[N,H]+[H] = [N,H]
        return K.sigmoid(I)

    def update_gate(self,xt,htm):
        Ui = K.dot(xt, self.Wiu)  # [N,I] x [I,H] = [N,H]
        Ur = K.dot(htm, self.Wru)  # [N,H] x [H,H] = [N,H]
        U = Ui + Ur + self.bu  # [N,H]+[N,H]+[H] = [N,H]
        return K.tanh(U)

    def output_gate(self,xt,htm):
        Oi = K.dot(xt, self.Wio)  # [N,I] x [I,H] = [N,H]
        Or = K.dot(htm, self.Wro)  # [N,H] x [H,H] = [N,H]
        O = Oi + Or + self.bo  # [N,H]+[N,H]+[H] = [N,H]
        return K.sigmoid(O)


    def step(self, inputs, states):
        print("state shape, inputs shape", K.int_shape(states[0]),K.int_shape(inputs))
        Ctm,htm=states
        xt=inputs
        ig=self.input_gate(xt,htm)#[N,H]
        rg=self.reset_gate(xt,htm)#[N,H]
        ug=self.update_gate(xt,htm)#[N,H]
        og=self.output_gate(xt,htm)#[N,H]
        C_reset=Ctm*rg
        C_update=ig*ug
        C_t=C_reset+C_update
        h_t=K.tanh(C_t)*og
        return h_t,[C_t,h_t]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        print("Calling: Inputs shape ",K.int_shape(inputs[:,0]))
        return super(PSMLSTM, self).call(inputs)

    def get_initial_state(self, inputs):
        Sm1=K.tanh(K.dot(inputs[:,0],self.Wsm1))
        #Sm1=K.zeros(shape=[self.nodes])
        return [Sm1,Sm1]

    def compute_output_shape(self, input_shape):
        if(self.return_sequences):
            shape=(None,self.input_time,self.nodes)
        else:
            shape=(None,self.nodes)
        return shape

class PAttentionCell_v2(Recurrent):
    '''
    A=Attention size (nb_units)
    O=Output size (output_dim) Lexicon size
    E=Number of Encoder nodes (input_dim)
    D=Number of Decoder nodes
    e_ts=Encoder timestep
    '''

    def __init__(self,nb_units,annotation_dim,output_dim,return_alphas=False,name="Attention",**kwargs):
        super(PAttentionCell_v2, self).__init__(**kwargs)
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        self.output_dim=output_dim
        self.annotation_dim=annotation_dim
        self.return_sequences=True
        self.return_alphas=return_alphas
        self.weight_constraint=keras.constraints.get(None)
        self.weight_initializer=keras.initializers.get('uniform')
        self.name=name
        self.predict=False

    def build(self, input_shape):
        print("build: input shape",input_shape)
        self.input_dim=input_shape[0][-1]
        #self.timesteps=input_shape[-2]
        self.d_ts=input_shape[0][1]
        print("nodes = ,input_dim , annotation dim ",self.nodes,self.input_dim,self.annotation_dim)
        #Weights for all the gates are initialized here
        '''
        every gate has four weights
            1)Wk=weight for input to gate (kernel)
            2)Wr=weight for recursive connection
            3)b=bias for this gate
            4)C=weight for context vector
        '''
        self.states = [None, None]
        #Weight matrix for initial state calculation
        self.Ws = self.add_weight('Ws',shape=(self.input_dim, self.nodes),initializer=self.weight_initializer)
        #inititate weights for Context vector Ct calculation
        self.Va=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='Va') #[A] to be multiplied with tanh(alignment output)
        self.Wa=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Wa')#[D,A] to be multiplied by Si-1 (Decoder hidden state)
        self.Ua=self.add_weight(shape=[self.annotation_dim,self.nodes],initializer=self.weight_initializer,name='Ua')#[E,A] to be multiplied with hj (Encoder output/annotation)
        self.ba=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='ba')#[A] Attention bias
        #initiate weights for Reset gate(r)
        self.Wkr=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkr')#[O,D]
        self.Urr=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urr')#[D,D]
        self.br=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='br')#[D]
        self.Cr=self.add_weight(shape=[self.annotation_dim,self.nodes],initializer=self.weight_initializer,name='Cr')#[E,D]
        # initiate weights for Update gate(z)
        self.Wkz=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkz')#[O,D]
        self.Urz=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urz')#[D,D]
        self.bz = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bz')#[D]
        self.Cz = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Cz')#[E,D]
        #initiate weights for Input gate(p)/proposal
        self.Wkp=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkp')#[O,D]
        self.Urp=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urp')#[D,D]
        self.bp = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bp')#[D]
        self.Cp = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Cp')#[E,D]
        #Initiate weights for Output gate(o)
        self.Wko=self.add_weight(shape=[self.output_dim,self.output_dim],initializer=self.weight_initializer,name='Wko')#[O,O]
        self.Uro=self.add_weight(shape=[self.nodes,self.output_dim],initializer=self.weight_initializer,name='Uro')#[D,O]
        self.bo = self.add_weight(shape=[self.output_dim], initializer=self.weight_initializer, name='bo')#[D]
        self.Co = self.add_weight(shape=[self.annotation_dim, self.output_dim], initializer=self.weight_initializer, name='Co')#[E,O]
        print("Weights Initiated")


    def call(self, inputs,mask=None, training=None, initial_state=None,predict=False):
        #print(states)
        self.input_sequence = inputs[0]  # [N,d_ts,E]
        self.annotation = inputs[1]  # [N,e_ts,E]
        if(predict):
            self.predict=True
        self.e_ts=K.int_shape(self.annotation)[1]
        print("Call is OK. encoder time %d decoder time %d"%(self.e_ts,self.d_ts))
        #self.U_h=P_time_distributed_dense(self.annotation,self.Ua,self.ba)#[N,e_ts,A] also E=A (is that required ?)
        self.U_h = K.dot(self.annotation, self.Ua)  # [N,e_ts,E] X [E,A] = [N,e_ts,A]
        return super(PAttentionCell_v2,self).call(self.input_sequence)

    def step(self, inputs, states):
        print("Step:inputs shape ",K.int_shape(inputs))
        if(self.predict):
            ytm = inputs
            _, stm = states
        else:
            ytm, stm = states  # states=[N,O],[N,D]
        #enc_hidden=dec_hidden
        '''_____Computation of a() starts ____'''
        #repeat the hidden state over all timesteps of encoder
        self.repeated_state=K.repeat(stm,self.e_ts) #N,e_ts,D
        #multiply the hidden state with Wa
        self._WaStm = K.dot(self.repeated_state,self.Wa) #[N,e_ts,D] X [D,A] = [N,e_ts,A]
        #self.U_h=K.dot(self.annotation,self.Ua)#[N,e_ts,E] X [E,A] = [N,e_ts,A]
        self._WaStm = K.tanh(self._WaStm+self.U_h+self.ba)#[N,e_ts,A]+[N,e_ts,A] = [N,e_ts,A]
        Va_expanded=K.expand_dims(self.Va) #[A,1]
        self.e_ij = K.dot(self._WaStm,Va_expanded) #[N,e_ts,1]
        '''_____Computation of a() ends, Now find alpha_ij ____'''
        self.alpha_ij=K.softmax(self.e_ij,axis=1) #still [N,e_ts,1] one alpha_ij for every combinantion of input pos i and output position j
        print("Step:Alpha shape ",K.int_shape(self.alpha_ij))
        '''_____alpha_ij computed, Now compute Context vector Ct ____'''
        self.c_t=K.batch_dot(self.alpha_ij,self.annotation,axes=1)#[N,e_ts,1] X [N,e_ts,E] = [N,1,E]
        self.c_t=K.squeeze(self.c_t,1)#[N,E]
        '''_____Context vector found, Now compute gates____'''
        self.call_reset_gate(ytm,stm)# get self.r
        self.call_update_gate(ytm,stm)# get self.Z_t
        self.call_input_gate(ytm,stm)#get self.S_t :this is new state from this step
        self.call_output_gate(ytm,stm)#get self.Y_t :this is new output token from this step
        if(self.return_alphas):
            return self.alpha_ij,[self.Y_t,self.S_t]
        else:
            return self.Y_t,[self.Y_t,self.S_t]#[N,O],states=[N,O],[N,D]

    #return self.S_t=[N,D]
    def call_input_gate(self,ytm,stm):#proposal
        r_stm=self.r*stm#[N,D]*[N,D]
        Iiw=K.dot(ytm,self.Wkp)#[N,O] X [O,D] = [N,D]
        Irw=K.dot(r_stm,self.Urp)#[N,D] X [D,D] = [N,D]
        Icw=K.dot(self.c_t,self.Cp)#[N,E] X [E,D] = [N,D]
        I=K.tanh(Iiw+Irw+Icw+self.bp)#[N,D]+[N,D]+[N,D]+[D] = [N,D]
        #new updated state
        self.S_t=((1-self.Z_t)*stm)+(self.Z_t*I)#[N,D]
        print("Proposal gate output ", K.int_shape(self.S_t))

    # self.r=returns [N,D]
    def call_reset_gate(self,ytm,stm):
        Riw=K.dot(ytm,self.Wkr)#[N,O] X [O,D] = [N,D] #previous predicted token
        Rrw=K.dot(stm,self.Urr)#[N,D] X [D,D] = [N,D]
        Rcw=K.dot(self.c_t,self.Cr)#[N,E] X [E,D] = [N,D]
        R=Riw+Rrw+Rcw+self.br#[N,D]+[N,D]+[D] = [N,D]
        self.r=K.sigmoid(R)#[N,D]
        print("Reset/Forget gate output ",K.int_shape(self.r))

    #returns self.Z_t=[N,D]
    def call_update_gate(self,ytm,stm):
        Ziw=K.dot(ytm,self.Wkz)#[N,O] X [O,D] = [N,D] #previous predicted token
        Zrw=K.dot(stm,self.Urz)#[N,D] X [D,D] = [N,D]
        Zcw=K.dot(self.c_t,self.Cz)#[N,E] X [E,D] =[N,D]
        z=Ziw+Zcw+Zrw+self.bz#[N,D]+[N,D]+[D] = [N,D]
        self.Z_t=K.sigmoid(z)#[N,D]
        print("Update gate output ", K.int_shape(self.Z_t))

    #return self.Y_t = [N,O]
    def call_output_gate(self,ytm,stm):
        Oiw=K.dot(ytm,self.Wko)#[N,O] X [O,O] = [N,O]
        Orw=K.dot(stm,self.Uro)#[N,D] X [D,O] = [N,O]
        Ocw=K.dot(self.c_t,self.Co)#[N,E] X [E,O] = [N,O]
        self.Y_t=K.sigmoid(Oiw+Orw+Ocw+self.bo)#[N,O]
        print("Output gate output ", K.int_shape(self.Y_t))#[N,O] ?

    def get_initial_state(self, inputs):
        print('get_initial_state:inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = K.tanh(K.dot(inputs[:, 0], self.Ws))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.nodes,
            'return_probabilities': self.return_alphas
        }
        base_config = super(PAttentionCell_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_alphas:
            return (None, self.d_ts, self.e_ts)
        else:
            return (None, self.d_ts, self.output_dim)