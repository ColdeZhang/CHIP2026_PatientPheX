labels=(R/'lable.vocab').read_text().strip().split('\n');assert len(labels)==17266
rep=BERT_RepresentationLayer(str(R/'bioformer-base'),str(R/'lable.vocab'),lowercase=False)
enc=TFAutoModel.from_config(AutoConfig.from_pretrained(R/'bioformer-base',local_files_only=True))
a=tf.keras.layers.Input((32,),dtype=tf.int32);b=tf.keras.layers.Input((32,),dtype=tf.int32);c=tf.keras.layers.Input((32,),dtype=tf.int32)
y=tf.keras.layers.Dense(17266,activation='softmax')(enc(a,token_type_ids=b,attention_mask=c)[0][:,0,:]);model=tf.keras.Model([a,b,c],y);model.load_weights(str(R/'bioformer_PT_v1.2.h5'));assert model.count_params()==51380594 and len(model.weights)==137
