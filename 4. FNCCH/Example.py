'''
Example run of FunctionalNetwork
@ Frankie Yeung (2021 May)

1. load spike time stamps
2. FNCCH reconstruction
3. filter unphysical reconstructions
'''
import FunctionalNetwork as fn

args = {
    "correlation-window": 200,                  # CCH over [-w/2,+w/2]
    "spike-timestamp-file": "DIV25_spks.txt",   # spk timestamp file to load
    #### experimental params by default ########################################
    "deltaT": 1/7.06,                           # physical time unit (ms)
    "deltaD": 42,                               # physical distance unit (micron)
    "tThres": 1,                                # time threshold: deltaT * abs(tau*) has to be at least tThres
    "vThres": 400                               # distance threshold: deltaD * (grid distance) / tau* can be at most vThres
}

def main():
    #### 1. load spike time stamps
    w = args["correlation-window"]
    n = fn.FunctionalNetwork(args["spike-timestamp-file"])
    #### 2. FNCCH reconstruction
    n.reconstruct(w=w)
    #### 3. filter unphysical reconstructions
    n.applyPhysicalFilter(w=w,
        deltaT=args["deltaT"],
        deltaD=args["deltaD"],
        tThres=args["tThres"],
        vThres=args["vThres"])

if __name__=="__main__":
    main()
