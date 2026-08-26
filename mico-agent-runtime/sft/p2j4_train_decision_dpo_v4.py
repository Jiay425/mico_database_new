"""DPO v4: frozen-SFT reference log-probs plus fresh DPO LoRA.

The reference is materialized before policy training.  The reference branch is
an explicit frozen SFT adapter model and the SFT adapter is never updated.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
from typing import Any

def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _read(path: Path) -> list[dict[str, Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def _validate_dataset(train: list[dict[str,Any]], validation: list[dict[str,Any]]) -> None:
    required={"id","prompt","chosen","rejected","task_kind","goal_code","observation_flags","history_actions","candidate_actions","state_signature"}
    missing=[row.get("id") for row in train+validation if not required.issubset(row)]
    if missing: raise SystemExit(f"DPO_V4_INPUT_CONTRACT_MISSING:{missing[:5]}")
    train_states={row["state_signature"] for row in train}; val_states={row["state_signature"] for row in validation}
    overlap=train_states & val_states
    if overlap: raise SystemExit(f"DPO_V4_STATE_LEAKAGE:{len(overlap)}")
    for row in train+validation:
        if not row["goal_code"] or not isinstance(row["observation_flags"],list) or not isinstance(row["history_actions"],list) or not isinstance(row["candidate_actions"],list):
            raise SystemExit(f"DPO_V4_INPUT_CONTRACT_INVALID:{row.get('id')}")
        try:
            chosen = json.loads(row["chosen"])
            rejected = json.loads(row["rejected"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"DPO_V4_COMPLETION_JSON_INVALID:{row.get('id')}") from exc
        chosen_action = chosen.get("selected_action")
        rejected_action = rejected.get("selected_action")
        if chosen_action not in row["candidate_actions"] or rejected_action not in row["candidate_actions"] or chosen_action==rejected_action:
            raise SystemExit(f"DPO_V4_PAIR_CONTRACT_INVALID:{row.get('id')}")
        if row.get("chosen_action", chosen_action) != chosen_action or row.get("rejected_action", rejected_action) != rejected_action:
            raise SystemExit(f"DPO_V4_PAIR_METADATA_MISMATCH:{row.get('id')}")
def _render(tok: Any, prompt: list[dict[str,str]], completion: str, max_length: int) -> dict[str,list[int]]:
    p=tok.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True,enable_thinking=False); f=tok.apply_chat_template([*prompt,{"role":"assistant","content":completion}],tokenize=False,add_generation_prompt=False,enable_thinking=False); pi=tok(p,add_special_tokens=False)["input_ids"]; fi=tok(f,add_special_tokens=False)["input_ids"]
    if len(fi)>max_length or len(pi)>=len(fi): raise ValueError("DPO_V4_TOKEN_LENGTH_INVALID")
    return {"input_ids":fi,"attention_mask":[1]*len(fi),"labels":[-100]*len(pi)+fi[len(pi):]}
def _features(tok: Any, rows: list[dict[str,Any]], max_length: int) -> list[dict[str,Any]]:
    out=[]
    for row in rows:
        out.append({"id":row["id"],"chosen":_render(tok,row["prompt"],row["chosen"],max_length),"rejected":_render(tok,row["prompt"],row["rejected"],max_length)})
    return out
def _pad(items: list[dict[str,list[int]]], pad: int) -> dict[str,torch.Tensor]:
    width=max(len(x["input_ids"]) for x in items)
    fill={"input_ids":pad,"attention_mask":0,"labels":-100}
    return {k:torch.tensor([x[k]+([fill[k]]*(width-len(x[k]))) for x in items],dtype=torch.long,device="cuda") for k in ("input_ids","attention_mask","labels")}
def _logps(model: Any, batch: dict[str,torch.Tensor]) -> torch.Tensor:
    logits=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).logits[:,:-1]; labels=batch["labels"][:,1:]; mask=labels.ne(-100); safe=labels.masked_fill(~mask,0); return (F.log_softmax(logits,-1).gather(2,safe.unsqueeze(-1)).squeeze(-1)*mask).sum(-1)
def _batches(rows: list[Any], size: int): return [rows[i:i+size] for i in range(0,len(rows),size)]
def _trainable_state(model: Any) -> dict[str,torch.Tensor]:
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if "lora_" in k}
def _save_checkpoint(model: Any, optimizer: Any, output: Path, epoch: int, step: int, history: list[dict[str,Any]]) -> None:
    checkpoint=output/f"checkpoint-epoch-{epoch}"
    checkpoint.mkdir(parents=True,exist_ok=True)
    torch.save(_trainable_state(model),checkpoint/"policy-lora.pt")
    torch.save(optimizer.state_dict(),checkpoint/"optimizer.pt")
    (checkpoint/"trainer-state.json").write_text(json.dumps({"epoch":epoch,"step":step,"history":history},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def _load_checkpoint(model: Any, optimizer: Any, checkpoint: Path) -> tuple[int,int,list[dict[str,Any]]]:
    state=torch.load(checkpoint/"policy-lora.pt",map_location="cpu")
    current=model.state_dict()
    current.update({k:v.to(next(model.parameters()).device,dtype=current[k].dtype) for k,v in state.items() if k in current})
    model.load_state_dict(current,strict=False)
    optimizer.load_state_dict(torch.load(checkpoint/"optimizer.pt",map_location="cpu"))
    info=json.loads((checkpoint/"trainer-state.json").read_text(encoding="utf-8"))
    return int(info["epoch"]),int(info["step"]),list(info.get("history",[]))
def _adapter_hashes(adapter: Path) -> dict[str,str]:
    files=("adapter_config.json","adapter_model.safetensors")
    missing=[name for name in files if not (adapter/name).exists()]
    if missing: raise SystemExit("DPO_V4_SFT_ADAPTER_FILES_MISSING:"+",".join(missing))
    return {name:_sha(adapter/name) for name in files}
def _reference(model_path: str, adapter: Path, tok: Any, items: list[dict[str,Any]], batch_size: int, out: Path) -> dict[str,Any]:
    base=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda"); ref=PeftModel.from_pretrained(base,adapter,is_trainable=False); ref.eval(); values=[]
    with torch.no_grad():
        for batch in _batches(items,batch_size):
            c=_logps(ref,_pad([x["chosen"] for x in batch],tok.pad_token_id)); r=_logps(ref,_pad([x["rejected"] for x in batch],tok.pad_token_id))
            values.extend({"id":x["id"],"reference_chosen":float(a),"reference_rejected":float(b)} for x,a,b in zip(batch,c.tolist(),r.tolist()))
    del ref,base; torch.cuda.empty_cache(); out.write_text("\n".join(json.dumps(x) for x in values)+"\n",encoding="utf-8"); return {"count":len(values),"sha256":_sha(out)}
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--sft-adapter",type=Path,required=True); p.add_argument("--train",type=Path,required=True); p.add_argument("--validation",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--max-length",type=int,default=2048); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--epochs",type=int,default=2); p.add_argument("--learning-rate",type=float,default=5e-6); p.add_argument("--beta",type=float,default=.1); p.add_argument("--checkpoint-every",type=int,default=1); p.add_argument("--resume",type=Path); p.add_argument("--dry-run",action="store_true"); p.add_argument("--reuse-reference",action="store_true"); p.add_argument("--contract-only",action="store_true"); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.dry_run and a.resume is None and not a.reuse_reference: raise SystemExit("DPO_V4_OUTPUT_NOT_EMPTY")
    raw_train=_read(a.train); raw_val=_read(a.validation); _validate_dataset(raw_train,raw_val)
    if a.contract_only:
        a.output_dir.mkdir(parents=True,exist_ok=True)
        report={"schemaVersion":"p2j4-dpo-v4-trainer-contract-check-v1","status":"PASS","trainingStarted":False,"contractOnly":True,"trainCount":len(raw_train),"validationCount":len(raw_val),"trainStateCount":len({row["state_signature"] for row in raw_train}),"validationStateCount":len({row["state_signature"] for row in raw_val}),"model":a.model,"sftAdapter":str(a.sft_adapter),"optimizerLoopPresent":True,"checkpointAndResumePresent":True}
        (a.output_dir/"contract-check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False)); return 0
    global torch, F, LoraConfig, PeftModel, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit("DPO_V4_CUDA_BF16_REQUIRED")
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,use_fast=True); tok.padding_side="right"; tok.pad_token=tok.pad_token or tok.eos_token
    train=_features(tok,raw_train,a.max_length); val=_features(tok,raw_val,a.max_length); a.output_dir.mkdir(parents=True,exist_ok=True); ref_path=a.output_dir/"reference-logprobs.jsonl"; ref_manifest_path=a.output_dir/"reference-logprob-manifest.json"; adapter_hashes=_adapter_hashes(a.sft_adapter)
    if a.reuse_reference:
        if not ref_path.exists() or not ref_manifest_path.exists(): raise SystemExit("DPO_V4_FROZEN_REFERENCE_MISSING")
        frozen_manifest=json.loads(ref_manifest_path.read_text(encoding="utf-8"))
        expected={"trainSha256":_sha(a.train),"validationSha256":_sha(a.validation),"referenceFileSha256":_sha(ref_path),"sftAdapterHashes":adapter_hashes}
        if frozen_manifest.get("status")!="PASS" or any(frozen_manifest.get(key)!=value for key,value in expected.items()): raise SystemExit("DPO_V4_FROZEN_REFERENCE_HASH_MISMATCH")
        ref_info={"count":frozen_manifest.get("referenceCount"),"sha256":frozen_manifest["referenceFileSha256"]}
    else:
        ref_info=_reference(a.model,a.sft_adapter,tok,train+val,a.batch_size,ref_path)
    ref={x["id"]:x for x in _read(ref_path)}
    if len(ref)!=len(train)+len(val) or set(ref)!={x["id"] for x in train+val}: raise SystemExit("DPO_V4_FROZEN_REFERENCE_ID_SET_INVALID")
    base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda")
    sft=PeftModel.from_pretrained(base,a.sft_adapter,is_trainable=False)
    merged=sft.merge_and_unload()
    # PEFT leaves this bookkeeping attribute on the merged base model.  It no
    # longer represents live SFT adapter weights, but get_peft_model() treats
    # it as a second adapter unless it is explicitly removed.
    if hasattr(merged,"peft_config"):
        delattr(merged,"peft_config")
    if hasattr(merged,"peft_config"):
        raise SystemExit("DPO_V4_MERGED_SFT_ADAPTER_METADATA_PRESENT")
    merged.config.use_cache=False
    policy=get_peft_model(merged,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],task_type=TaskType.CAUSAL_LM))
    policy.enable_input_require_grads(); policy.gradient_checkpointing_enable(); policy.train()
    def loss(batch):
        pc=_logps(policy,_pad([x["chosen"] for x in batch],tok.pad_token_id)); pr=_logps(policy,_pad([x["rejected"] for x in batch],tok.pad_token_id)); rc=torch.tensor([ref[x["id"]]["reference_chosen"] for x in batch],device="cuda"); rr=torch.tensor([ref[x["id"]]["reference_rejected"] for x in batch],device="cuda"); return -F.logsigmoid(a.beta*((pc-pr)-(rc-rr))).mean()
    l=loss([train[0]]); l.backward(); grads=[x.grad for x in policy.parameters() if x.requires_grad and x.grad is not None]
    if not torch.isfinite(l) or not grads or not all(torch.isfinite(g).all() for g in grads): raise SystemExit("DPO_V4_DRY_FORWARD_BACKWARD_FAILED")
    manifest={"schemaVersion":"p2j4-dpo-v4-reference-logprob-manifest-v1","status":"PASS","referenceTopology":"Qwen3-8B Base + frozen SFT adapter","policyTopology":"merged SFT weights + fresh DPO LoRA","disableAdapterUsed":False,"referenceFile":str(ref_path),"referenceFileSha256":ref_info["sha256"],"referenceCount":ref_info["count"],"trainSha256":_sha(a.train),"validationSha256":_sha(a.validation),"sftAdapter":str(a.sft_adapter),"sftAdapterHashes":adapter_hashes,"dryForwardBackward":True,"dryLoss":float(l.item()),"trainingStarted":False}
    ref_manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if a.dry_run:
        print(json.dumps(manifest,ensure_ascii=False)); return 0
    optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=a.learning_rate)
    history=[]; start_epoch=0; step=0
    if a.resume is not None:
        start_epoch,step,history=_load_checkpoint(policy,optimizer,a.resume)
    randomizer=random.Random(42)
    for epoch in range(start_epoch,a.epochs):
        order=list(train); randomizer.shuffle(order); policy.train()
        for batch in _batches(order,a.batch_size):
            optimizer.zero_grad(set_to_none=True); batch_loss=loss(batch); batch_loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.0); optimizer.step(); step+=1; history.append({"epoch":epoch+1,"step":step,"loss":float(batch_loss.item())})
        policy.eval()
        with torch.no_grad():
            validation_losses=[float(loss(batch).item()) for batch in _batches(val,a.batch_size)]
        history.append({"epoch":epoch+1,"validationLoss":sum(validation_losses)/len(validation_losses)})
        if (epoch+1)%a.checkpoint_every==0: _save_checkpoint(policy,optimizer,a.output_dir,epoch+1,step,history)
    adapter_dir=a.output_dir/"adapter"; policy.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
    metrics={"trainingStarted":True,"model":a.model,"sftAdapter":str(a.sft_adapter),"trainCount":len(train),"validationCount":len(val),"epochs":a.epochs,"learningRate":a.learning_rate,"beta":a.beta,"checkpointEvery":a.checkpoint_every,"resumedFrom":str(a.resume) if a.resume else None,"history":history,"referenceManifest":str(a.output_dir/"reference-logprob-manifest.json")}
    (a.output_dir/"train_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(metrics,ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
