from pathlib import Path
import sys,re,json,sqlite3,time,uuid
from datetime import datetime,timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
from mock_llm import chat
OUT=ROOT/"outputs"; STATE=ROOT/"state"; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
POL=ROOT/"data/policies"

class PolicyRAG:
    def __init__(self,overlap=10):
        self.chunks=[]
        for f in sorted(POL.glob("*.txt")):
            words=f.read_text(encoding="utf-8").split()
            step=40-overlap
            for start in range(0,len(words),step):
                chunk=words[start:start+40]
                if not chunk: break
                self.chunks.append({"source":f.name,"text":" ".join(chunk)})
                if start+40>=len(words): break
        self.vec=TfidfVectorizer()
        self.matrix=self.vec.fit_transform([x["text"] for x in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vec.transform([query]); scores=cosine_similarity(q,self.matrix)[0]
        ids=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
        return [{"source":self.chunks[i]["source"],"score":float(scores[i]),"text":self.chunks[i]["text"]} for i in ids]
    def answer(self,question):
        hits=self.retrieve(question,3)
        context="\n".join(f"[{x['source']}]\n{x['text']}" for x in hits)
        user=f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        resp=chat([{"role":"system","content":"POLICY ASSISTANT: Answer only from supplied context. If absent, say I could not find that information in the policy documents."},{"role":"user","content":user}])
        return resp.choices[0].message.content,[x["source"] for x in hits]

def run():
    (OUT/"refund_ledger.jsonl").write_text("", encoding="utf-8")
    rag=PolicyRAG()
    evaldf=pd.read_csv(ROOT/"data/raw/policy_eval.csv")
    ev=[]
    for _,r in evaldf.iterrows():
        hits=rag.retrieve(r.question,3); ans,_=rag.answer(r.question)
        ev.append({"question":r.question,"expected_doc_found":r.expected_doc=="NONE" or r.expected_doc in [h["source"] for h in hits],
                   "expected_fact_found":r.expected_doc=="NONE" or r.expected_fact in ans,
                   "fallback":ans.startswith("I could not find")})
    pd.DataFrame(ev).to_csv(OUT/"policy_eval_results.csv",index=False)
    no=PolicyRAG(overlap=0); ev0=[]
    for _,r in evaldf.iterrows():
        hits=no.retrieve(r.question,3); ans,_=no.answer(r.question)
        ev0.append({"question":r.question,"expected_doc_found":r.expected_doc=="NONE" or r.expected_doc in [h["source"] for h in hits],
                    "expected_fact_found":r.expected_doc=="NONE" or r.expected_fact in ans,
                    "fallback":ans.startswith("I could not find")})
    pd.DataFrame(ev0).to_csv(OUT/"policy_eval_zero_overlap.csv",index=False)
    db=ROOT/"data/instamart.db"
    def get_order(order_id):
        with sqlite3.connect(db) as con:
            row=con.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone()
        return {"error":"order not found"} if row is None else dict(zip(["order_id","status","order_amount","delivery_minutes"],row))
    def search_policy(query): return {"results":rag.retrieve(query,3)}
    def calculate_refund(order_amount,delivery_minutes):
        return {"refund_inr":round(float(order_amount),2) if float(delivery_minutes)>45 else 0.0}
    def ledger(tool,order_id,refund_inr):
        rec={"timestamp":datetime.now(timezone.utc).isoformat(),"tool":tool,"order_id":order_id,"refund_inr":refund_inr}
        with (OUT/"refund_ledger.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(rec)+"\n")
        return {"refund_inr":refund_inr}
    def issue_refund(order_id,refund_inr):return ledger("issue_refund",order_id,refund_inr)
    def escalate_to_human(order_id,refund_inr):return ledger("escalate_to_human",order_id,refund_inr)
    toolmap={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,
             "issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
    spec=[{"name":"get_order","description":"Fetch a cleaned order.","parameters":{"order_id":"string"}},
          {"name":"search_policy","description":"Retrieve policy passages.","parameters":{"query":"string"}},
          {"name":"calculate_refund","description":"Calculate eligible refund.","parameters":{"order_amount":"number","delivery_minutes":"number"}},
          {"name":"issue_refund","description":"Record approved refund.","parameters":{"order_id":"string","refund_inr":"number"}},
          {"name":"escalate_to_human","description":"Send high-value refund to human lead.","parameters":{"order_id":"string","refund_inr":"number"}}]
    def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after=None,run_id=None):
        run_id=run_id or str(uuid.uuid4()); started=time.monotonic(); tokens=0; steps=[]; executed=[]
        msgs=[{"role":"system","content":"AGENT. Use tools to verify the order and apply refund policy."},{"role":"user","content":user_message}]
        def checkpoint():
            payload={"run_id":run_id,"scenario_id":scenario_id,"messages":msgs,"tokens_used":tokens,"step_number":len(steps),"tools_executed":executed,"steps":steps}
            (STATE/f"{run_id}.json").write_text(json.dumps(payload,default=str),encoding="utf-8")
        for _ in range(max_iterations):
            if time.monotonic()-started>=time_limit_seconds:
                return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: hand off to a human agent.","steps":steps,"llm_calls":len([m for m in msgs if m["role"]=="assistant"]),"tokens_used":tokens}
            resp=chat(msgs,tools=spec); tokens+=resp.usage.total_tokens
            if tokens>token_limit:
                return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: hand off to a human agent.","steps":steps,"llm_calls":len([m for m in msgs if m["role"]=="assistant"])+1,"tokens_used":tokens}
            m=resp.choices[0].message
            if not m.tool_calls:
                answer=m.content or "FALLBACK: hand off to a human agent."
                return {"status":answer.split(":")[0],"final_answer":answer,"steps":steps,"llm_calls":len([x for x in msgs if x["role"]=="assistant"])+1,"tokens_used":tokens}
            tc=m.tool_calls[0]; args=tc.arguments
            msgs.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}]})
            obs=toolmap[tc.name](**args); executed.append(tc.name)
            msgs.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(obs)})
            steps.append({"tool":tc.name,"arguments":args,"observation":obs}); checkpoint()
            if crash_after is not None and len(executed)==crash_after: raise RuntimeError("SIMULATED_CRASH")
        return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off to a human agent.","steps":steps,"llm_calls":max_iterations,"tokens_used":tokens}
    # Resume from the exact post-observation checkpoint, without repeating tools.
    def resume_agent(run_id):
        p=STATE/f"{run_id}.json"; saved=json.loads(p.read_text())
        msgs=saved["messages"]; tokens=saved["tokens_used"]; steps=saved["step_number"]; executed=saved["tools_executed"]
        started=time.monotonic()
        for _ in range(6):
            resp=chat(msgs,tools=spec); tokens+=resp.usage.total_tokens
            if tokens>6000:return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: hand off to a human.","steps":saved["steps"],"llm_calls":0,"tokens_used":tokens}
            m=resp.choices[0].message
            if not m.tool_calls:
                ans=m.content or ""; return {"status":ans.split(":")[0],"final_answer":ans,"steps":saved.get("steps",[]),"tools_executed":executed,"llm_calls":0,"tokens_used":tokens}
            tc=m.tool_calls[0]; obs=toolmap[tc.name](**tc.arguments); executed.append(tc.name)
            msgs.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":tc.arguments}]})
            msgs.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(obs)})
            saved.setdefault("steps",[]).append({"tool":tc.name,"arguments":tc.arguments,"observation":obs})
            saved.update({"messages":msgs,"tokens_used":tokens,"step_number":len(saved["steps"]),"tools_executed":executed})
            p.write_text(json.dumps(saved,default=str))
        return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off.","steps":saved.get("steps",[]),"tools_executed":executed,"llm_calls":0,"tokens_used":tokens}

    scenarios=pd.read_csv(ROOT/"data/raw/agent_scenarios.csv")
    runs=[]
    for _,r in scenarios.iterrows():
        result=run_agent(r.message,r.scenario_id)
        result["scenario_id"]=r.scenario_id
        runs.append(result)
    with (OUT/"agent_runs.jsonl").open("w") as f:
        for x in runs:f.write(json.dumps(x,default=str)+"\n")
    # Required stop-safe demonstrations.
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1.message,"S1_TOKEN_DEMO",token_limit=200)
    timeout_demo=run_agent(s1.message,"S1_TIMEOUT_DEMO",time_limit_seconds=0)
    (OUT/"agent_demos.json").write_text(json.dumps({"token_limit_demo":token_demo,"timeout_demo":timeout_demo},default=str,indent=2))
    # Crash immediately after second tool observation, then resume.
    crash_id="S1_CRASH_RESUME"
    try: run_agent(s1.message,"S1",crash_after=2,run_id=crash_id)
    except RuntimeError: pass
    resumed=resume_agent(crash_id)
    (OUT/"crash_resume_result.json").write_text(json.dumps(resumed,default=str,indent=2))
    return runs

if __name__=="__main__": run()
