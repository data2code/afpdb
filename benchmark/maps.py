from afpdb import util
import multiprocessing,time

def map(f, tasks, n_CPU=0, progress=True, min_interval=30):
    """min_interval, do not report progress more frequent that 30 sec"""
    if len(tasks)==0: return []
    if n_CPU==0:
        n_CPU=max(multiprocessing.cpu_count()-2, 1)
    if n_CPU<=1:
        if not progress:
            return [ f(args) for args in tasks ]
        else:
            out=[]
            pg=util.Progress(len(tasks))
            i_start=time.time()
            for i, args in enumerate(tasks):
                out.append(f(args))
                if (time.time()-i_start)>=min_interval or i==len(tasks)-1:
                    pg.check(i+1)
                    i_start=time.time()
            return out
    n_CPU=min(n_CPU, len(tasks))
    pl=multiprocessing.Pool(n_CPU)
    if progress:
        # https://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call
        out=[]
        pg=util.Progress(len(tasks))
        i_start=time.time()
        #for i, _ in enumerate(pl.imap(f, tasks), 1):
        for i, _ in enumerate(pl.imap(f, tasks)):
            out.append(_)
            #print(">>>", i, time.time(), i_start, time.time()-i_start)
            if (time.time()-i_start)>=min_interval or i==len(tasks)-1:
                pg.check(i+1)
                i_start=time.time()
    else:
        out=pl.map(f, tasks)
    pl.close()
    pl.join()
    return out
