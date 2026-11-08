import numpy as np, subprocess, sys
import gguf
from gguf import GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import dequantize
rng = np.random.default_rng(0)
D, NH, NKV, HD, E, K, FF, NL, CTX, BASE = 256, 4, 2, 64, 8, 2, 256, 2, 128, 10000.0
QD, KD = NH*HD, NKV*HD

# bytes_to_unicode
bs = list(range(33,127))+list(range(161,173))+list(range(174,256)); cs = bs[:]; n=0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n+=1
b2u = {b: chr(c) for b,c in zip(bs,cs)}
tokens = [b2u[b] for b in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
types = [1]*257 + [3]*3
V = len(tokens)

BLK = {Q.Q8_0:(32,34,3e-4), Q.Q4_K:(256,144,3e-5), Q.Q5_K:(256,176,1.5e-5), Q.Q6_K:(256,210,5e-6)}
def qtensor(qt, shape):           # returns (raw bytes array, dequantized float32)
    blck, nb, dscale = BLK[qt]
    rows = int(np.prod(shape[:-1])); nblk = shape[-1]//blck
    raw = rng.integers(0,256,(rows,nblk,nb),dtype=np.uint8)
    def put(off, val): raw[:,:,off:off+2] = (val.astype(np.float16)).view(np.uint8).reshape(rows,nblk,2)
    shp = (rows,nblk)
    if qt == Q.Q6_K: put(208, rng.uniform(.5,1.5,shp)*dscale)
    else:
        put(0, rng.uniform(.5,1.5,shp)*dscale)
        if qt in (Q.Q4_K,Q.Q5_K): put(2, rng.uniform(.5,1.5,shp)*dscale)
    raw = raw.reshape(rows, nblk*nb)
    f = dequantize(raw, qt).reshape(*shape).astype(np.float32)
    return raw.reshape(*shape[:-1], nblk*nb), f

w = GGUFWriter("tiny.gguf", "qwen3moe")
for k,v in [("block_count",NL),("context_length",CTX),("embedding_length",D),("attention.head_count",NH),("attention.head_count_kv",NKV),
            ("attention.key_length",HD),("attention.value_length",HD),("expert_count",E),("expert_used_count",K),("expert_feed_forward_length",FF)]:
    w.add_uint32(f"qwen3moe.{k}", v)
w.add_float32("qwen3moe.rope.freq_base", BASE); w.add_float32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
w.add_tokenizer_model("gpt2"); w.add_string("tokenizer.ggml.pre","qwen2")
w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(["a b"]); w.add_eos_token_id(V-2)

W = {}
def add_q(name, qt, shape):
    raw, f = qtensor(qt, shape); w.add_tensor(name, raw, raw_dtype=qt); W[name] = f
def add_f(name, arr):
    arr = arr.astype(np.float32); w.add_tensor(name, arr); W[name] = arr
add_q("token_embd.weight", Q.Q8_0, (V,D)); add_q("output.weight", Q.Q6_K, (V,D)); add_f("output_norm.weight", rng.uniform(.8,1.2,D))
for l in range(NL):
    p = f"blk.{l}."
    add_f(p+"attn_norm.weight", rng.uniform(.8,1.2,D)); add_f(p+"ffn_norm.weight", rng.uniform(.8,1.2,D))
    add_f(p+"attn_q_norm.weight", rng.uniform(.8,1.2,HD)); add_f(p+"attn_k_norm.weight", rng.uniform(.8,1.2,HD))
    add_q(p+"attn_q.weight", Q.Q5_K, (QD,D)); add_q(p+"attn_k.weight", Q.Q8_0, (KD,D))
    add_q(p+"attn_v.weight", Q.Q6_K, (KD,D)); add_q(p+"attn_output.weight", Q.Q4_K, (D,QD))
    add_f(p+"ffn_gate_inp.weight", rng.normal(0,.3,(E,D)))
    add_q(p+"ffn_gate_exps.weight", Q.Q5_K, (E,FF,D)); add_q(p+"ffn_up_exps.weight", Q.Q4_K, (E,FF,D)); add_q(p+"ffn_down_exps.weight", Q.Q6_K, (E,D,FF))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("weight absmax sample:", {k: float(np.abs(v).max()) for k,v in list(W.items())[:2]})

# ---------------- numpy reference ----------------
def rms(x,wt): return x/np.sqrt((x*x).mean(-1,keepdims=True)+1e-6)*wt
def rope(x, pos):                  # x (T,H,HD) NEOX
    inv = BASE**(-2*np.arange(HD//2)/HD); ang = pos[:,None]*inv[None]; c,s = np.cos(ang)[:,None], np.sin(ang)[:,None]
    a,b = x[...,:HD//2], x[...,HD//2:]; return np.concatenate([a*c-b*s, a*s+b*c],-1)
def ref(toks):
    T=len(toks); X = W["token_embd.weight"][toks]; pos=np.arange(T,dtype=np.float64)
    for l in range(NL):
        p=f"blk.{l}."; h=rms(X,W[p+"attn_norm.weight"])
        q=(h@W[p+"attn_q.weight"].T).reshape(T,NH,HD); k=(h@W[p+"attn_k.weight"].T).reshape(T,NKV,HD); v=(h@W[p+"attn_v.weight"].T).reshape(T,NKV,HD)
        q=rope(rms(q,W[p+"attn_q_norm.weight"]),pos); k=rope(rms(k,W[p+"attn_k_norm.weight"]),pos)
        o=np.zeros((T,NH,HD))
        for hh in range(NH):
            kv=hh//(NH//NKV); s=(q[:,hh]@k[:,kv].T)/np.sqrt(HD); s=np.where(np.tril(np.ones((T,T)))>0,s,-1e30)
            s=np.exp(s-s.max(-1,keepdims=True)); s/=s.sum(-1,keepdims=True); o[:,hh]=s@v[:,kv]
        X=X+o.reshape(T,QD)@W[p+"attn_output.weight"].T
        h=rms(X,W[p+"ffn_norm.weight"]); lg=h@W[p+"ffn_gate_inp.weight"].T; pr=np.exp(lg-lg.max(-1,keepdims=True)); pr/=pr.sum(-1,keepdims=True)
        out=np.zeros_like(X)
        for t in range(T):
            top=np.argsort(-pr[t])[:K]; ww=pr[t,top]/pr[t,top].sum()
            for e,wt in zip(top,ww):
                g=W[p+"ffn_gate_exps.weight"][e]@h[t]; u=W[p+"ffn_up_exps.weight"][e]@h[t]
                out[t]+=wt*(W[p+"ffn_down_exps.weight"][e]@((g/(1+np.exp(-g)))*u))
        X=X+out
    return rms(X[-1],W["output_norm.weight"])@W["output.weight"].T

ids=[5,17,200,33,99,3,150,256,42,77,8,9,10,11,12]
r=ref(ids)
print("ref   :", np.round(r[:8],5), "argmax", r.argmax())
ok=True
for b in ("1","4","64"):
    out=subprocess.run(["./qwen3moe","-m","tiny.gguf","-c","128","-b",b,"--logits-test",",".join(map(str,ids))],capture_output=True,text=True)
    line=out.stdout.strip().split("\n"); e=np.array(line[0].split(),dtype=np.float64)
    err=np.abs(e-r[:16]).max(); am=int(line[1].split()[1])
    print(f"batch {b:>2}: max|err|={err:.2e} argmax={am}", "OK" if err<2e-3*max(1,np.abs(r).max()) and am==r.argmax() else "FAIL")
    ok &= err<2e-3*max(1,np.abs(r).max())
print(subprocess.run(["./qwen3moe","-m","tiny.gguf","--tokenize","ab abab<|im_end|>\nhi"],capture_output=True,text=True).stdout.strip(), " <- expected 256 ... ids")
sys.exit(0 if ok else 1)
