#!/usr/bin/env python3
"""Real-time development arena: shared Hikari 24-row/SRS/attack/input rules.

Both TBP engines ponder continuously on one worker. Six public pieces, 3s
countdown, 60Hz, entry delay3 / clear delay38 frames. Garbage arrives after60
frames, cancels FIFO and rises only on a nonclear lock, at most8 rows per lock.
This is development evaluation, not the reserved1000-seed evaluation.
"""
import argparse, collections, hashlib, json, pathlib, queue, random, subprocess, threading, time

KINDS = list('IOTLJSZ')

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

class Engine:
    def __init__(self, command, log):
        self.log = open(log, 'w'); self.err = open(str(log)+'.stderr', 'w')
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1)
        self.messages = queue.Queue(); self.closed = False
        def read():
            for line in self.p.stdout:
                try: self.messages.put(json.loads(line))
                except Exception as e: self.messages.put({'type':'transport_error','error':str(e)})
            self.messages.put({'type':'transport_error','error':'engine stdout closed'})
        threading.Thread(target=read, daemon=True).start()
        self.info = self.get(); assert self.info['type']=='info', self.info
        self.send({'type':'rules','randomizer':'seven_bag'}); assert self.get()['type']=='ready'
    def send(self, msg):
        self.log.write(json.dumps({'time':time.monotonic(),'in':msg})+'\n'); self.log.flush()
        self.p.stdin.write(json.dumps(msg)+'\n'); self.p.stdin.flush()
    def poll(self):
        try: msg=self.messages.get_nowait()
        except queue.Empty: return None
        self.log.write(json.dumps({'time':time.monotonic(),'out':msg})+'\n'); self.log.flush()
        if msg.get('type')=='transport_error' or msg.get('type')=='error': raise RuntimeError(msg)
        return msg
    def get(self):
        end=time.monotonic()+15
        while time.monotonic()<end:
            msg=self.poll()
            if msg is not None: return msg
            time.sleep(.001)
        raise RuntimeError('engine response timed out')
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.send({'type':'quit'});self.p.wait(timeout=3)
        except Exception:self.p.kill();self.p.wait()
        self.log.close();self.err.close()

class Rules:
    def __init__(self, exe):
        self.p=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def call(self, board, pieces, move=None):
        self.p.stdin.write(json.dumps({'op':'plan' if move else 'legal','board':board,'queue':pieces,'move':move})+'\n');self.p.stdin.flush()
        v=json.loads(self.p.stdout.readline())
        if 'error' in v:raise RuntimeError(v['error'])
        return v
    def close(self):self.p.terminate();self.p.wait()

class Bag:
    def __init__(self, seed):self.rng=random.Random(seed);self.remaining=[]
    def draw(self):
        if not self.remaining:self.remaining=KINDS.copy();self.rng.shuffle(self.remaining)
        return self.remaining.pop()

def cancel(pending, amount):
    while amount and pending:
        take=min(amount,pending[0]['rows']);pending[0]['rows']-=take;amount-=take
        if not pending[0]['rows']:pending.popleft()
    return amount

def rise(board, pending, frame):
    raised=0;death=False
    while pending and pending[0]['due']<=frame and raised<8:
        item=pending[0];n=min(item['rows'],8-raised)
        death |= any(board['rows'][-n:])
        board['rows']=[1023 ^ (1<<item['hole'])]*n+board['rows'][:-n]
        item['rows']-=n;raised+=n
        if not item['rows']:pending.popleft()
    return raised, bool(death)

def start_message(player):
    b=player['board']
    cells=[[('G' if mask&(1<<x) else None) for x in range(10)] for mask in b['rows']]+[[None]*10 for _ in range(16)]
    return {'type':'start','board':cells,'queue':player['queue'],'hold':b['hold'],'combo':b['ren'],
            'back_to_back':b['back_to_back'],'randomizer':{'type':'seven_bag','bag_state':player['bag'].remaining.copy()}}

def player(name, seed):
    bag=Bag(seed)
    return {'name':name,'bag':bag,'queue':[bag.draw() for _ in range(6)],'board':{'rows':[0]*24,'hold':None,'bag':0,'ren':0,'back_to_back':False},
            'pending':collections.deque(),'ready':0,'request':None,'active':None,'pieces':0,'lines':0,'attack':0,'sent':0,'received':0,'dead':False}

def game(args, number, seed, order, rules, out):
    engines=[];players=[player(name,seed) for name in order];events=[];reason='time_limit';max_lag=0;lag_sum=0;frames=0
    try:
        for side,p in enumerate(players):
            cmd=[str(args.hikari),'tbp','--threads','1'] if p['name']=='hikari' else [str(args.cold_clear)]
            eng=Engine(cmd,out/f'game-{number}-side-{side}.jsonl');engines.append(eng);eng.send(start_message(p))
        time.sleep(args.countdown)
        base=time.monotonic();holes=[random.Random(seed^0xAAA),random.Random(seed^0xBBB)]
        for frame in range(args.seconds*60):
            target=base+frame/60;now=time.monotonic()
            if now<target:time.sleep(target-now)
            lag=max(0,time.monotonic()-target);max_lag=max(max_lag,lag);lag_sum+=lag;frames=frame+1
            outgoing=[0,0]
            for side,(p,eng) in enumerate(zip(players,engines)):
                if p['active'] and p['active']['lock_frame']<=frame:
                    plan=p.pop('active');p['active']=None
                    p['board']=plan['board'];p['pieces']+=1;p['lines']+=plan['lines'];p['attack']+=plan['attack'];p['dead']|=plan['death']
                    eng.send({'type':'play','move':plan['move']})
                    for _ in range(plan['consumed']):
                        p['queue'].pop(0);piece=p['bag'].draw();p['queue'].append(piece);eng.send({'type':'new_piece','piece':piece})
                    outgoing[side]=cancel(p['pending'],plan['attack']);p['sent']+=outgoing[side]
                    raised=0
                    if not plan['lines']:
                        raised,death=rise(p['board'],p['pending'],frame);p['received']+=raised;p['dead']|=death
                    p['ready']=frame+3+(38 if plan['lines'] else 0)
                    events.append({'frame':frame,'side':side,'move':plan['move'],'path':plan['path'],'board':p['board'].copy(),'queue':p['queue'].copy(),
                                   'attack':plan['attack'],'sent':outgoing[side],'received':raised,'lines':plan['lines'],'pieces':p['pieces']})
                    if raised:
                        eng.send({'type':'stop'});eng.send(start_message(p))
                msg=eng.poll()
                if msg is not None:
                    if msg['type']!='suggestion' or p['request'] is None:raise RuntimeError(f'unexpected message {msg}')
                    p['request']=None;moves=msg.get('moves',[])
                    if not moves:
                        p['dead']=not rules.call(p['board'],p['queue'])['legal']
                        if not p['dead']:raise RuntimeError('empty suggestion despite reachable placements')
                    else:
                        mv=moves[0];plan=rules.call(p['board'],p['queue'],mv)
                        plan.update(move=mv,lock_frame=frame+plan['frames']);p['active']=plan
                if not p['dead'] and p['active'] is None and p['request'] is None and frame>=p['ready']:
                    if not rules.call(p['board'],p['queue'])['legal']:p['dead']=True
                    else:eng.send({'type':'suggest'});p['request']=frame
                if p['request'] is not None and frame-p['request']>180:raise RuntimeError(f'{p["name"]} no suggestion after3s')
            for side,amount in enumerate(outgoing):
                if amount:players[1-side]['pending'].append({'rows':amount,'due':frame+60,'hole':holes[side].randrange(10)})
            if any(p['dead'] for p in players):reason='top_out';break
        dead=[p['dead'] for p in players];winner=None if dead[0]==dead[1] else players[1 if dead[0] else 0]['name']
        result={'game':number,'seed':seed,'order':order,'winner':winner,'reason':reason,'seconds':frames/60,'wall_seconds':time.monotonic()-base,
                'max_host_lag_seconds':max_lag,'mean_host_lag_seconds':lag_sum/frames,'players':[{k:p[k] for k in ['name','pieces','lines','attack','sent','received','dead','board']} for p in players],
                'events':events,'engine_info':[e.info for e in engines]}
        save(out/f'game-{number}.json',result);return result
    finally:
        for eng in engines:eng.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--hikari',type=pathlib.Path,required=True);ap.add_argument('--cold-clear',type=pathlib.Path,required=True);ap.add_argument('--rules',type=pathlib.Path,required=True)
    ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--games',type=int,default=4);ap.add_argument('--seconds',type=int,default=180);ap.add_argument('--countdown',type=float,default=3);ap.add_argument('--seed',type=int,default=1611001)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    args.hikari=args.hikari.resolve();args.cold_clear=args.cold_clear.resolve();args.rules=args.rules.resolve()
    manifest={'rules':__doc__,'seconds':args.seconds,'countdown':args.countdown,'seed':args.seed,'games':args.games,'workers_each':1,
              'binaries':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in [('hikari',args.hikari),('cold_clear',args.cold_clear),('rules',args.rules)]}}
    save(out/'manifest.json',manifest);rules=Rules(args.rules);results=[]
    try:
        for i in range(args.games):
            order=['hikari','cold-clear-2'] if i%2==0 else ['cold-clear-2','hikari']
            result=game(args,i,args.seed+i//2,order,rules,out);results.append({k:v for k,v in result.items() if k!='events'})
            summary={'games':len(results),'hikari_wins':sum(r['winner']=='hikari' for r in results),'cold_clear_wins':sum(r['winner']=='cold-clear-2' for r in results),'draws':sum(r['winner'] is None for r in results),'results':results}
            save(out/'summary.json',summary);print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    except Exception as e:
        save(out/'error.json',{'error':str(e),'completed_games':len(results)});raise
    finally:rules.close()

if __name__=='__main__':main()
