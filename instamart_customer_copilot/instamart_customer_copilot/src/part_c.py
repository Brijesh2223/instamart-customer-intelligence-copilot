from pathlib import Path
import sys,re,json,random,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
from mock_llm import est_tokens,TransientLLMError
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
CATS={"billing","delivery","technical","fraud"}
EMAIL=re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE=re.compile(r"\b[6-9]\d{9}\b")
ORDER=re.compile(r"ORD-\d{6}")
ZERO="CLASSIFY. Allowed categories: billing, delivery, technical, fraud. Return JSON only: {\"category\": \"...\", \"reason\": \"...\"}."
FEW=ZERO+"\nEXAMPLES:\nTicket: unauthorized transaction on my card\n{\"category\":\"fraud\",\"reason\":\"unauthorized transaction\"}\nTicket: app crashes on login\n{\"category\":\"technical\",\"reason\":\"app login crash\"}"

def redact(t):
    counts=[]
    for pat,rep in [(EMAIL,"[EMAIL]"),(PHONE,"[PHONE]"),(ORDER,"[ORDER_ID]")]:
        t,n=pat.subn(rep,t); counts.append(n)
    return t,counts

def classify_batch(prompt,budget=None,output=None):
    df=pd.read_csv(ROOT/"data/raw/tickets.csv",keep_default_na=False)
    seg=pd.read_csv(OUT/"customer_segments.csv")
    segmap=seg.set_index("customer_id")[["segment_name","is_premium"]].to_dict("index")
    mock_llm.reset(); random.seed(42)
    running=0; rows=[]; retry_count=0; red_counts=[0,0,0]
    for _,r in df.iterrows():
        tid,cid,raw=r.ticket_id,r.customer_id,r.text
        if not raw.strip(): status,cat,tok="skipped_empty","",0
        elif est_tokens(raw)>400: status,cat,tok="flagged_long","",0
        else:
            safe,counts=redact(raw)
            red_counts=[a+b for a,b in zip(red_counts,counts)]
            assert not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
            if budget is not None and running>=budget: status,cat,tok="budget_exhausted","",0
            else:
                messages=[{"role":"system","content":prompt},{"role":"user","content":safe}]
                resp=None
                for n in range(4):
                    try:
                        resp=mock_llm.chat(messages); break
                    except TransientLLMError:
                        if n==3: raise
                        wait=.05*(2**n)*random.uniform(.9,1.15)
                        print(f"retry ticket={tid} retry={n} wait={wait:.5f}s")
                        retry_count+=1; time.sleep(wait)
                tok=resp.usage.total_tokens; running+=tok
                content=resp.choices[0].message.content or ""
                try:
                    obj=json.loads(content); candidate=obj.get("category")
                    if candidate in CATS: status,cat="classified_json",candidate
                    else: status,cat="flagged_invalid_output",""
                except Exception:
                    words=re.findall(r"[a-z]+",content.lower())
                    found=[x for x in CATS if x in words]
                    if len(found)==1: status,cat="classified_keyword",found[0]
                    else: status,cat="flagged_invalid_output",""
        info=segmap.get(cid,{})
        premium=bool(info.get("is_premium",False))
        if status in {"skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted"}:
            route,priority="human_review","NORMAL"
        elif cat=="fraud": route,priority="security_team","HIGH"
        elif premium: route,priority="premium_desk","HIGH"
        else: route,priority="support_queue","NORMAL"
        rows.append({"ticket_id":tid,"customer_id":cid,"status":status,"category":cat,
          "segment_name":info.get("segment_name",""),"priority":priority,"route":route,"tokens_used":tok})
    result=pd.DataFrame(rows)
    if output: result.to_csv(output,index=False)
    print("redaction email/phone/order_id/total:",*red_counts,sum(red_counts))
    print("statuses:",result.status.value_counts().to_dict(),"tokens:",running,"retries:",retry_count)
    return result

def run():
    classify_batch(ZERO,6000,OUT/"triage_budget_demo.csv")
    return classify_batch(FEW,100000,OUT/"triage_results.csv")
if __name__=="__main__": run()
