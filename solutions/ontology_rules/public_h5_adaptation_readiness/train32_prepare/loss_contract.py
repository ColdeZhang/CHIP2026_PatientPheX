"""CPU-only stable canonical-class mass objective and unchanged Dense head contract."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf

def validate_columns(columns,classes):
 if not columns or len(columns)!=len(set(columns)) or any(type(x) is not int or x<0 or x>=classes for x in columns):raise ValueError('invalid target columns')
 return columns

def canonical_mass_loss(logits,columns):
 validate_columns(columns,int(logits.shape[-1]))
 return tf.reduce_logsumexp(logits,axis=-1)-tf.reduce_logsumexp(tf.gather(logits,columns,axis=-1),axis=-1)

def main(out):
 t=time.monotonic();assert tf.__version__=='2.12.0';tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1);assert not tf.config.list_physical_devices('GPU')
 tf.random.set_seed(20261024);np.random.seed(20261024);D=17266;cases=[]
 for name,cols,extreme in [('single',[7],False),('equivalent_columns',[7,19,17264],False),('None',[17265],False),('wrong_class_extreme_single',[7],True),('wrong_class_extreme_equivalent',[7,19],True)]:
  value=np.random.default_rng(43).normal(size=(2,D)).astype('float64')
  if extreme:value.fill(-1000);value[:,0]=1000;value[:,cols]=-1000
  z=tf.Variable(value,dtype=tf.float64)
  with tf.GradientTape() as tape:loss=canonical_mass_loss(z,cols);objective=tf.reduce_mean(loss)
  grad=tape.gradient(objective,z).numpy();p=tf.nn.softmax(z).numpy();q=np.zeros_like(p);q[:,cols]=tf.nn.softmax(tf.gather(z,cols,axis=-1)).numpy();expected=(p-q)/len(value)
  checks=dict(finite=bool(np.isfinite(loss.numpy()).all() and np.isfinite(grad).all()),gradient_matches_stable_analytic=bool(np.allclose(grad,expected,rtol=1e-10,atol=1e-12)))
  if len(cols)==1:
   ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[cols[0]]*2,logits=z);checks['single_equals_native_logits_CE']=bool(np.allclose(loss.numpy(),ce.numpy(),rtol=1e-12,atol=1e-12))
  if extreme:checks['true_wrong_class_gradient_not_clipped_to_zero']=bool(np.max(np.abs(grad[:,cols]))>.2)
  cases.append(dict(name=name,columns=cols,loss=loss.numpy().tolist(),max_gradient_error=float(np.max(np.abs(grad-expected))),checks=checks))
  assert all(checks.values())
 invalid=[]
 for cols in [[],[-1],[17266],[1,1],[True]]:
  try:canonical_mass_loss(tf.zeros([1,D]),cols)
  except ValueError:invalid.append(dict(columns=cols,rejected=True))
  else:raise AssertionError('invalid target accepted')
 # Head-only interface proof: reuse the SAME original Dense variables, expose its
 # input from the original graph and multiply by its kernel/bias. No new Dense.
 inp=tf.keras.Input((5,),dtype=tf.float32);dense=tf.keras.layers.Dense(D,activation='softmax');model=tf.keras.Model(inp,dense(inp));hidden_model=tf.keras.Model(model.inputs,dense.input)
 x=tf.Variable(np.random.default_rng(2).normal(size=(3,5)).astype('float32'));before=[v.numpy().copy() for v in model.weights]
 with tf.GradientTape(persistent=True) as tape:
  hidden=hidden_model(x,training=True);logits=tf.matmul(hidden,dense.kernel)+dense.bias;prob=model(x,training=False)
  loss=tf.reduce_mean(canonical_mass_loss(logits,[17265]));native=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[17265]*3,logits=logits))
 variables=[x,dense.kernel,dense.bias];a=tape.gradient(loss,variables);b=tape.gradient(native,variables)
 head=dict(native_softmax_equals_same_dense_preactivation=bool(np.allclose(prob.numpy(),tf.nn.softmax(logits).numpy(),rtol=1e-6,atol=1e-8)),all_original_variables_identical_objects=len(model.weights)==2 and model.weights[0] is dense.kernel and model.weights[1] is dense.bias,weight_bytes_unchanged=all(np.array_equal(v.numpy(),old) for v,old in zip(model.weights,before)),activation_still_softmax=dense.activation is tf.keras.activations.softmax,single_loss_abs_diff=float(abs(loss.numpy()-native.numpy())),gradient_max_abs_differences=[float(tf.reduce_max(tf.abs(u-v)).numpy()) for u,v in zip(a,b)],gradients_finite=all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in a))
 assert head['native_softmax_equals_same_dense_preactivation'] and head['all_original_variables_identical_objects'] and head['weight_bytes_unchanged'] and head['activation_still_softmax'] and head['gradients_finite'];assert head['single_loss_abs_diff']<2e-6 and max(head['gradient_max_abs_differences'])<2e-6
 result=dict(status='PASS_CPU_SYNTHETIC_STABLE_LOSS_AND_DENSE_INTERFACE',tensorflow=tf.__version__,GPU_visible=[],classes=D,cases=cases,invalid_targets=invalid,head=head,seconds=time.monotonic()-t,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),public_model_or_weights_loaded=False,optimizer_or_fit=False,limitations=['Synthetic logits and small hidden-to-head graph only; no actual public encoder backward/memory/quality proof.','Training adapter must use original Dense variables and training=True hidden graph; original inference model and softmax unchanged.','No probability clipping; empty or invalid target columns fail before loss.'])
 with Path(out).open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
