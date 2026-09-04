from inspect_ai.log import read_eval_log
import glob, sys
f=sorted(glob.glob(sys.argv[1]))[-1]
s=read_eval_log(f).samples[0]
for m in s.messages:
    if m.role=='assistant':
        c=m.content
        if isinstance(c,list):
            for x in c:
                if getattr(x,'type',None)=='reasoning' and x.reasoning.strip(): print('  [CoT]',x.reasoning.strip()[:500])
                elif getattr(x,'type',None)=='text' and x.text.strip(): print('  [TEXT]',x.text.strip()[:500])
        for tc in (m.tool_calls or []): print('  [TOOLCALL]',tc.function,'|',str(tc.arguments)[:220])
    elif m.role=='tool':
        print('TOOL:', str(m.content)[:160].replace(chr(10),' '))
    elif m.role=='user':
        print('USER:', str(m.content)[:120].replace(chr(10),' '))
sc=s.scores['exfil_asr']
print('\nSCORE',sc.value,sc.answer,'|',sc.explanation.split(chr(10))[0])
