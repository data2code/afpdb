#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import print_function
import shlex, subprocess
import numpy as np
import pandas as pd
import math
import os
import shutil
import re
import sys
import six
from six.moves import range
from six.moves import zip
#import setting
import datetime

def is_python3():
    return sys.version_info[0]>=3

### OS/IO
def no_warning():
    """Call this at the beginnig of your script to suppress warning, when we use db.py."""
    import warnings
    warnings.simplefilter("ignore")

def no_buffer():
    """Force autoflush for stdout"""
    # no way to get unbuffer stdout working in python3, so I change buffer size to one
    if is_python3():
        sys.stdout = os.fdopen(sys.__stdout__.fileno(), 'w', 1)
    else:
        sys.stdout = os.fdopen(sys.fileno(), 'w', 0)

def module_exists(module_name):
    # http://stackoverflow.com/questions/5847934/how-to-check-if-python-module-exists-and-can-be-imported
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

def _unix(exe, l_print=True, l_error=True, l_block=True, error_to_stdout=True):
    '''Execute a unix command and return results in a list\nExample: unix(["ls", "-l"]
    exe: str, command line string, subject to command line trojan, so be responsible
    l_print: boolean, default False. return string if l_print=False, else print and no return
    l_block, by default True, waits for the command to finish
        If false, return immediately with the Popen instance
    '''
    #import commands
    #print 'This function is deprecated, please use commands.getstatusoutput() instead!\n'
    if type(exe) is list:
         exe=" ".join(exe) #shlex.split(exe)
    #err, out = commands.getstatusoutput(exe)
    #print(">>>>>>>>>>>>", exe)
    #https://stackoverflow.com/questions/11495783/redirect-subprocess-stderr-to-stdout
    if l_block:
        out,err = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
    else:
        if is_python3():
            #see https://stackoverflow.com/questions/50573169/leave-a-process-running-after-closing-python
            return subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, start_new_session=True)
        else:
            return subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    #print(exe, ' +++ ',  out, ' +++ ', err, len(err), err is not None)
    if l_print:
        if l_error and err is not None and len(err)>0:
            print("ERROR> "+err.decode('utf-8', 'ignore'))         #YfZ
        print(out.decode('utf-8', 'ignore'))

    if l_error and err is not None and len(err)>0:
        #print("ERROR> "+"".join(err))
        if error_to_stdout:
            out+=err
    if is_python3():
        return out.decode('utf-8', 'ignore') # + '***' + str(err) #out.splitlines()
    else:
        return str(out)

def unix(exe, l_print=True, l_error=True, l_block=True, error_to_stdout=True, sleep_time=0.5):
    '''Execute a unix command and return results in a list\nExample: unix(["ls", "-l"]
    exe: str, command line string, subject to command line trojan, so be responsible
    l_print: boolean, default False. return string if l_print=False, else print and no return
    l_block, by default True, waits for the command to finish
        If false, return immediately with the Popen instance
    '''
    if type(exe) is list:
         exe=" ".join(exe) #shlex.split(exe)
    if l_block:
        ON_POSIX='posix' in sys.builtin_module_names
        p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0, close_fds=ON_POSIX)
        p_keep=p
    else:
        #see https://stackoverflow.com/questions/50573169/leave-a-process-running-after-closing-python
        if is_python3():
            # seems we need to pass None to stdout, stderr
            # otherwise, cmd >/dev/null works, but cmd may fail
            return subprocess.Popen(exe, stdout=None, stderr=None, shell=True, start_new_session=True)
        else:
            return subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    #https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
    from threading import Thread
    if is_python3():
        from queue import Queue, Empty
    else:
        from Queue import Queue, Empty
    import time

    sys.stdout.flush()
    done=[False, False]
    p=[p.stdout, p.stderr]
    q=[Queue(), Queue()]
    out=[ [], [] ]

    def enqueue(p, q, i):
        for line in iter(p[i].readline, b''):
            q[i].put(line.decode('utf-8', 'ignore'))
        p[i].close()
        done[i]=True
    for i in range(2):
        t=Thread(target=enqueue, args=(p, q, i))
        t.daemon=True
        t.start()

    try:
        while True:
            l_data=False
            for i in range(2):
                while True:
                    try:
                        line=q[i].get_nowait()
                    except Empty:
                        break
                    else:
                        out[i].append(line)
                        l_data=True
                        if l_print:
                            if i==1 and l_error:
                                print("ERROR> "+line.strip())
                            else:
                                print(line.strip())
                        if i==1 and error_to_stdout:
                            out[0].append(line)
            if sum(done)>1: break
            time.sleep(sleep_time)
    except KeyboardInterrupt as e: # this will catch Ctrl-C
        p_keep.kill()
    return "".join(out[0])


def unix2(exe, l_print=True):
    '''Execute a unix command and return results in a list\nExample: unix(["ls", "-l"]
    exe:'''
    if type(exe) is str:
        exe=exe.split()
    if l_print:
        subprocess.call(exe, shell=False)
    else:
        return subprocess.check_output(exe)
    return

def to_ascii(line):
    '''Remove non-printable chars'''
    if line is None: return ''
    if not is_python3():
        line=str(line)
    if type(line) is str:
        line=line.encode('utf-8').decode('ascii', 'ignore')
    else:
        line=line.decode('ascii', 'ignore')
    return line

def memory_usage(l_gc=True):
    """Print out memory usage, run garbage collection before that."""
    import gc
    import psutil
    import os

    def display(x):
        if x<1000:
            return str(x)
        if x<1e6:
            return "%.1fk" % (x/1e3)
        if x<1e9:
            return "%.1fm" % (x/1e6)
        return "%.1fg" % (x/1e9)

    if l_gc: gc.collect()
    p=psutil.Process(os.getpid())
    if 'memory_info' in dir(p):
        x=p.memory_info()
    else:
        x=p.memory_info_ex()
    print(("rss: %s, vms: %s, shared: %s" % (display(x.rss), display(x.vms), display(x.shared))))
    return x


def gzip_it(s_file):
    """gzip a disk file: s_file"""
    #if is_python3():
    #    import subprocess as commands                #add by YfZ 02/02/2018
    #else:
    #    import commands
    if os.path.exists(s_file):
        if os.path.exists(s_file+".gz"):
            os.remove(s_file+".gz")
        #commands.getstatusoutput('gzip '+s_file)
        unix('gzip '+s_file)

def ungzip_it(s_file):
    """ungzip the disk file s_file. Expecting s_file to ends with .gz"""
#    import commands
    if s_file.endswith(".gz") and os.path.exists(s_file):
#        commands.getstatusoutput('gunzip -f '+s_file)
        unix('gunzip -f '+s_file)

def empty_dir(folder):
    """Delete all files and subfolders in the folder"""
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            else:
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def list_files(folder):
    """Return list of files under a folder"""
    return [ x for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x)) ]

def lines_in_file(s_file):
    """return number of lines in file s_file"""
    n=0
    with open(s_file) as f:
        for x in f:
            n+=1
    return n

def save_list(s_file, S_lines, s_end=""):
    """Save S_lines to s_file, each line ends with character s_end.
    s_file: str, file name
    S_lines: list[str], lines of text
    s_end: str, append to line, default ''. You may want to set it to '\n' to provide breaks."""

    # sometimes strings arrive in unicode format, so convert to str first
    if is_python3():
        if type(S_lines) is not list:
            S_lines=[S_lines]
    else:
        if type(S_lines) is unicode:
            S_lines = [ S_lines.encode('utf8')]
        elif type(S_lines) is str:
            S_lines=[S_lines]
    if is_python3():
        f=open(s_file, 'w', encoding="utf-8")
    else:
        f=open(s_file, 'w')
    for s in S_lines:
        f.write(s+s_end)
    f.flush()
    f.close()

def _read_list(s_file, encoding="utf-8"):
    """Read s_file into a list of strings, remove line breaks"""
    S=[]
    if s_file.endswith('.gz'):
        import gzip
        with gzip.open(s_file, mode='rt', encoding=encoding) as f:
            S=[s.rstrip('\r\n') for s in f]
    else:
        if is_python3():
            with open(s_file, encoding=encoding) as f:
                S=[s.rstrip('\r\n') for s in f]
        else:
            with open(s_file) as f:
                S=[s.rstrip('\r\n') for s in f]
    return S

def read_list(s_file, encoding="utf-8"):
    """Read s_file into a list of strings, remove line breaks"""
    if s_file.endswith('.gz'):
        import gzip
        with gzip.open(s_file, mode='rb') as f:
            s=f.read()
    else:
        with open(s_file, mode='rb') as f:
            s=f.read()
    s=s.decode(encoding, 'ignore')
    S=s.splitlines()
    return S

def save_string(s_file, s):
    """Save a string s into file s_file"""
    with open(s_file, 'w') as f:
        f.write(s)
        f.flush()

def read_string(s_file, encoding="utf-8"):
    """Read s_file into a long string"""
    s_file = str(s_file)
    s=""
    if s_file.endswith('.gz'):
        import gzip
        with gzip.open(s_file, 'rb') as f:
            s=str(f.read())
    else:
        if is_python3():
            with open(s_file, 'r', encoding="utf-8") as f:
                s=f.read()
        else:
            with open(s_file, 'r') as f:
                s=f.read()
    return s

def read_table_file(fn, **kwargs):
    """filter is method that takes a chunk (dataframe) and return a filtered chunk
        a typical chunk is
        lambda chunk: chunk[chunk['#tax_id'].isin(SyncDB.SUPPORTED_SPECIES)]

        top: return when we found top # of records
        max_chunks: return if reached, regardless of how many records were found
        This is for debugging purpose for very long files, such as uniprot
        We can return early to debug the program

        continuous: True/False, whether data records are stored as a continuous cluster
            e.g., if data are sorted by tax_id and we filter for a tax_id
            that means we can return early when we no longer find a new record
            continuous=True means return if get a zero-size chunk after non-zero chunk(s)
    """
    print(f"read_table_file> {fn}")
    opt={'sep':"\t", 'chunksize':500000, 'keep_default_na':False, 'low_memory':False, 'index_col':False, 'continuous':False}
    filter = None
    top =  max_chunks = None
    if kwargs is not None: opt.update(kwargs)
    if 'filter' in opt:
        filter=opt['filter']
        del opt['filter']
    if 'top' in opt:
        top=opt['top']
        del opt['top']
    if 'max_chunks' in opt:
        max_chunks=opt['max_chunks']
        del opt['max_chunks']
    l_continuous=False
    if 'continuous' in opt:
        l_continuous=opt['continuous']
        del opt['continuous']
    iter_csv=pd.read_csv(fn, **opt)
    data=[]
    cnt=0
    n_cnt=0
    empty=None
    for chunk in iter_csv:
        cnt+=1
        if empty is None: empty=chunk[:0]
        if filter is not None: chunk=filter(chunk)
        n_cnt+=len(chunk)
        if len(chunk):
            data.append(chunk)
            print("o", end="")
            sys.stdout.flush()
        else:
            print(".", end="")
            sys.stdout.flush()
            if n_cnt>0 and l_continuous: break
        if top is not None and n_cnt>=top: break
        if max_chunks is not None and cnt>=max_chunks: break
    print("")
    if len(data):
        t=pd.concat(data)
    else:
        t=empty
    return t

def read_csv(s_file, *args, **kwargs):
    ''':rtype :pandas.core.frame.DataFrame'''
    """Read csv either in .csv or in .csv.gz.
    args, kwargs will be passed to pandas.read_csv()"""
    # pandas can read .gz, .zip directly
    return pd.read_csv(s_file, *args, **kwargs)

def csv_header(s_file):
    # peek to get the headers
    t=read_csv(s_file, nrows=1)
    return t.header()

def to_csv(df, s_file, **kwargs):
    kw=kwargs.copy()
    df.to_csv(s_file, **kw)
    gzip_it(s_file)

def format_path(s_path, OS=None):
    """format file path in the format run by the current OS, either Unix posix style or windows nt style
    OS: str, default None, autodetect current platform, otherwise 'win' or else 'unix'"""
    OS = sys.platform if OS is None else OS
    if (OS.startswith('win')):
        # win32 for windows
        s = re.sub(r'^\\(?=[^\\])', r'\\\\', re.sub('/', r'\\', s_path))
        if hasattr(setting, 'win_folder_rename'):
            s=setting.win_folder_rename(s)
    else:
        # darwin for Mac OSX, cygwin for windows/cygwin, linux for Linux
        s = re.sub('^//', '/', re.sub(r'\\', '/', s_path))
    return s

def dir_sep():
    """return '/' or '\\' depending on the current OS."""
    # use os.sep instead
    return '/' if (os.name in ('posix')) else '\\'

### array/list

def index(elm, lst):
    """return int index where elm appears in a list lst. -1 if not found."""
    try:
        if type(list)!=list:
            lst=list(lst)
        idx=lst.index(elm)
        return idx
    except ValueError:
        return -1

def index_all(elm, lst):
    """return list[int] all positions where elm appears in lst. Empty list if not found"""
    if type(list)!=list:
        lst=list(lst)
    return [i for i,e in enumerate(lst) if e==elm]

def rm_na(lst):
    return [x for x in lst if not pd.isnull(x)]

def unique(lst, is_rm_na=False):
    """return the unique elements of a list"""
    if is_rm_na:
        if type(lst) is np.ndarray:
            return list(np.unique(lst[~np.isnan(lst)]))
        return list(set(rm_na(lst)))
    if type(lst) is np.ndarray:
        return list(np.unique(lst))
    return list(set(lst))

def unique2(lst, is_rm_na=False):
    """Unique a string list, but preserve the order. This is useful for taking the top X number of unique entries"""
    out=[]
    c_seen={}
    l_na_seen=False
    for x in lst:
        if x in c_seen: continue
        if pd.isnull(x):
            if is_rm_na or l_na_seen: continue
            l_na_seen=True
        c_seen[x]=True
        out.append(x)
    return out

def unique_count(lst, is_rm_na=False):
    """dict of unique element counts in list, keys are elements and values are counts"""
    c={}
    l_na_seen=False
    na_cnt=0
    na_val=None
    for x in lst:
        if pd.isnull(x):
            if is_rm_na: continue
            na_cnt+=1
            if not l_na_seen:
                l_na_seen=True
                na_val=x
            continue
        c[x]=c.get(x,0)+1
    if na_cnt>0: c[na_val]=na_cnt
    return c

### debug
MSG_PREFIX=""

def error_msg(s_msg):
    """Print an error message and quit"""
    #raise Exception(MSG_PREFIX+s_msg)
    raise Exception(MSG_PREFIX+"ERROR> "+str(s_msg))
    #sys.stdout.flush()

def warn_msg(s_msg):
    """Print a warning message"""
    print(MSG_PREFIX+"WARN> "+s_msg)
    sys.stdout.flush()

def info_msg(s_msg):
    """Print an information message"""
    print(MSG_PREFIX+"INFO> "+s_msg)
    sys.stdout.flush()

def debug_msg(s_msg):
    """Print a debugging message"""
    print(MSG_PREFIX+"DEBUG> "+s_msg)
    sys.stdout.flush()

def uniform_subset(N, n):
    """When a dataset has too many entries N, we want to only compute a subset of n
    Returns an index array of size n
    Returns 0 .. N, if n>=N"""
    if n>=N: return np.arange(N);
    return np.unique(np.array(np.linspace(0, N-1, n), dtype=int))
### formatting

def sarray2sarray(S, s_null=''):
    """Convert an array/list to a new str array, replace NULL elements by s_null"""
    return [s_null if pd.isnull(s) else str(s) for s in S]

def r2i2s(r, s_null=''):
    """Convert a float to int then to str, replace NULL by s_null"""
    try:
        r=float(r)
    except:
        r=None
    return s_null if pd.isnull(r) else str(int(round(r)))

def rarray2iarray(S, s_null=None):
    """Convert an real array/list to iarray, replace NULL by s_null"""
    return [s_null if pd.isnull(s) else int(s) for s in S]

def sarray2iarray(S):
    """Convert an string array/list to iarray, replace non-int by"""
    out=[]
    for s in S:
        try:
            r=int(float(s))
        except:
            r=None
        out.append(r)
    return out

def iarray2sarray(S, s_null=''):
    """Convert an int array/list to str array, replace NULL by s_null"""
    return [s_null if pd.isnull(s) else str(int(s)) for s in S]

def rarray2sarray(R, s_format='%.2g', s_null=''):
    """Convert float array to str array, with format s_format, replace NULL by s_null"""
    return [s_null if np.isnan(r) else s_format % r for r in R]

def sarray2rarray(S):
    """Convert a str array to a np.array"""
    R=np.empty(len(S))
    for i,s in enumerate(S):
        try:
            r=float(s)
        except:
            r=np.nan
        R[i]=r
    return R

def df2sdf(df, s_format='%.2g', s_null='',l_all_str=False):
    """Convert a dataframe to str dataframe, so all numbers are strings with s_format. This is used before to_csv(), so that we get rid of all unnecessary long digits in the .csv file"""
    df2=df.copy()
    for s in header(df2):
        if df[s].dtype != np.dtype(object):
            #print 'ssss'
            #print df[s].dtype
            #if df[s].dtype in (np.dtype(int), np.int64, np.int32):
            if np.issubdtype(df[s].dtype, np.integer):
                df2[s]=[s_null if pd.isnull(x) else (str(x) if l_all_str else x) for x in df[s]]
            elif np.issubdtype(df[s].dtype, np.bool_):
                df2[s] = ["True" if x else "False" for x in df[s]]
            elif np.issubdtype(df[s].dtype, np.floating):
                df2[s]=[s_null if pd.isnull(x) else s_format % x for x in df[s]]
            #elif np.issubdtype(df[s].dtype, np.number) or np.issubdtype(df[s].dtype, np.unsignedinteger):
            #    if s_format.startswith('int'):
            #        true_s_format = s_format.replace('int','')
            #        df2[s] = [ s_null if pd.isnull(x) else (str(int(x))  if abs(int(x)-x)< 1e-5 else true_s_format %x)  for x in df[s]]
            elif np.issubdtype(df[s].dtype, np.flexible):
                df2[s]=[s_null if pd.isnull(x) else s_format % x for x in df[s]]
            else:
                warn_msg("Unrecognize dtype: "+str(df[s].dtype)+" for column "+s)
        else: # assume string column
            df2.loc[pd.isnull(df[s]), s]=''
    return df2

def no_nan(R):
    """New array by removing NULLs"""
    valid = pd.notnull(R)
    return R[valid]

def stdv(R):
    """STDV by removing NULL first"""
    return np.std(no_nan(R), ddof=1)

def mean(R):
    """Mean by removing NULL first"""
    return np.mean(no_nan(R))

def median(R):
    """Median by removing NULL first"""
    return np.median(no_nan(R))

### calculation
def distance(a, b, R_weight=None, metrics='PEARSON', has_null=True):
    """Calculate distance between two np.arrays.
    a, b: np.array
    R_weight: np.array, default None, weighting vector
    metrics: str, 'PEARSON' or 'MANHATTAN', or 'EUCLIDEAN'
    has_null: wether input vector may contain NULL"""
    a2, b2, w2 = a, b, R_weight
    if has_null:
        valid = pd.notnull(a) & pd.notnull(b)
        if not valid.all():
            a2 = a[valid]
            b2 = b[valid]
            w2 = R_weight[valid] if R_weight is not None else None
    if len(a2)<=1 and metrics=='PEARSON':
        return 0.5
    if w2 is None:
        if hasattr(metrics, '__call__'):
            return metrics(a, b)
        elif metrics=='MANHATTAN':
            return np.sum(np.abs(a2-b2))
        elif metrics=='EUCLIDEAN':
            return math.sqrt(np.sum((a2-b2)**2))
        elif metrics=='PEARSON' or metrics=='BUILD_IN':
            r=np.corrcoef(a2, b2)[0, 1]
            return 0.5 if np.isnan(r) else abs(1.0-r)/2
    else:
        n=np.sum(w2)
        if hasattr(metrics, '__call__'):
            return metrics(a2, b2, w2)
        elif metrics=='MANHATTAN':
            return np.sum(np.abs(a2-b2)*w2)/n
        elif metrics=='EUCLIDEAN':
            return math.sqrt(np.sum(np.power(a2-b2, 2)*w2)/n)
        elif metrics=='PEARSON' or metrics=='BUILD_IN':
            sumX=np.sum(a2*w2)
            sumY=np.sum(b2*w2)
            sumXX=np.sum(a2*a2*w2)
            sumYY=np.sum(b2*b2*w2)
            xy=np.sum(a2*b2*w2)-sumX*sumY/n
            r=xy/math.sqrt((sumXX-sumX*sumX/n)*(sumYY-sumY*sumY/n))
            return 0.5 if np.isnan(r) else abs(1.0-r)/2

def pearson(X,Y,R_weight=None):
    """
    Calculates Pearson correlation between two sets of vectors X and Y,
    i.e. between all pairs (x,y) where x is from X, and y is from Y.
    Rows are vectors, columns are features.
    X: np.array, LxN (or can be 1-D if there's only one vector)
    Y: np.array, MxN (or 1-D)
    R_weight: np.array 1xN or 1-D, default None, weighting vector
    Returns np.array LxM, where entry ij is correlation between X[i] and Y[j]
    """
    A, B, w = X, Y, R_weight
    if A.ndim == 1: A = A.reshape(1,len(A))
    N = A.shape[1]
    if B.ndim == 1: B = B.reshape(1,N)
    L = A.shape[0]
    M = B.shape[0]
    if w is None:
        n = float(N)
        sumX=np.sum(A,axis=1)
        sumY=np.sum(B,axis=1)
        sumXX=np.sum(A*A,axis=1)
        sumYY=np.sum(B*B,axis=1)
        xy=A.dot(B.T)-sumX.reshape(L,1)*sumY.reshape(1,M)/n
    else:
        w = np.array(w)
        if w.ndim == 1: w = w.reshape(1,N)
        n = float(np.sum(w))
        sumX=np.sum(A*w,axis=1)
        sumY=np.sum(B*w,axis=1)
        sumXX=np.sum(A*A*w,axis=1)
        sumYY=np.sum(B*B*w,axis=1)
        xy=A.dot(B.T)-sumX.reshape(L,1)*sumY.reshape(1,M)/n
    r=xy.astype(np.float64)/np.sqrt(((sumXX-sumX*sumX/n).reshape(L,1)*(sumYY-sumY*sumY/n).reshape(1,M)).astype(np.float64))
    r = np.where(np.isnan(r),0,r)
    r = np.where(r > -1, r, -1)
    r = np.where(r < 1, r, 1)
    return r

### DataFrame
def header(df):
    """Obtain column names from a dataframe"""
    return list(df.columns) #.values)

def col_index(df, s_col):
    return index(s_col, header(df))

def rename2(df, columns):
    """rename column headers, similar to the build-in DataFrame.rename, but always inplace
    and use no additional memory. Default rename seems to create a new copy of table."""
    df.columns=[ columns.get(x, x) for x in header(df) ]

def move_before(df, S_col, ref_col):
    """Move columns (or one column) S_col before ref_col)"""
    S_header=list(df.columns)
    if type(S_col) is str: S_col=[S_col]
    S=[x for x in S_header if x not in S_col]
    i_pos=index(ref_col, S)
    S[i_pos:i_pos]=S_col
    return df.reindex(columns=S)

def move_after(df, S_col, ref_col):
    S_header=list(df.columns)
    if type(S_col) is str: S_col=[S_col]
    S=[x for x in S_header if x not in S_col]
    i_pos=index(ref_col, S)
    S[i_pos+1:i_pos+1]=S_col
    return df.reindex(columns=S)

def move_column(df, S_col, i_pos):
    """Move dataframe column(s) S_col to i_pos position
        Can take one column name or a list
    """
    S_header=list(df.columns)
    return move_before(df, S_col, S_header[i_pos])

def melt(t, S_colsToCollapse):
    """Melt a dataframe
    S_colsToCollapse: list[str], column to be used to collapse
    return dataframe: contains a column 'variable' and 'value'. Original column names in 'variable'"""
    t2=t[S_colsToCollapse].stack()
    t2.name='value'
    S=list(t2.index.names)
    S[-1]='variable'
    # FrozenList, cannot be changed
    #t2.index.names[-1]='variable'
    t2.index.names=S
    t2=t2.reset_index('variable')
    t3=t.drop(S_colsToCollapse, axis=1)
    T=t3.join(t2)
    T.index=list(range(len(T)))
    return T

def split(df, n_chunk=1, chunk_size=0, key=None):
    """split a table/list/2d-array etc into n chunks of roughly equal size, but not gauranteed. This can be used to split an input data into multiple pieces, send to workers, then you need to assemble the output back into one piece by yourself.
    n_chunk: int, number of pieces to break the data into, default 1
    chunk_size: int, default 0, maximum size per piece. 0 means no limit.
    It attempts to break the input into n_chunk, but if too large (defined by chunk_size), it will create more pieces, so each is not bigger than chunk_size,  You may specify one of n_chunk and chunk_size, or both
    If key is provided, it is an array/list, where we need to make sure rows with the same key must stay together.
        e.g., if key is GROUP_ID, then rows with the same GROUP_ID will not be split into two chunks
        However, you need to make sure rows with the same GROUP_ID must be next to each other in df
        Note: we assume on average each key has about the same number of records, if not, results may not be optimal
    """
    n=len(df)
    if key is not None:
        key=np.array(key)
        IDX=np.concatenate([[-1], np.where(key[:-1]!=key[1:])[0], [n-1]])
        IDX=[(a+1,b+1) for a,b in zip(IDX[:-1], IDX[1:])]
        IDX=split(IDX, n_chunk=n_chunk, chunk_size=chunk_size)
        out=[df[X[0][0]:X[-1][1]].copy() if isinstance(df, pd.DataFrame) else df[X[0][0]:X[-1][1]] for X in IDX]
        return out
    sz=int(math.ceil(n*1.0/n_chunk)) if n_chunk>0 else n
    if chunk_size==0:
        chunk_size=sz
    else:
        chunk_size=min(chunk_size, sz)
    chunk_size=max(1, chunk_size)
    out=[]
    for iB in range(0, n, chunk_size):
        iE=min(iB+chunk_size, n)
        out.append(df[iB:iE].copy() if isinstance(df, pd.DataFrame) else df[iB:iE])
    return out

def split_exact(df, n_chunk=1):
    """Return exactly n_chunks (if fewer, some are empty), may not be even size, if not dividable"""
    n=len(df)
    sz=int(math.floor(n*1.0/n_chunk))
    remainder=n % n_chunk
    out=[]
    iB=0
    for i in range(0, n_chunk):
        if remainder==0 or iB>=remainder:
            chunk_size=sz
        else:
            chunk_size=sz+1
        iE=min(iB+chunk_size, n)
        out.append(df[iB:iE])
        iB=iE
    return out

def html(df, colors=None, tag_table=None, tag_tr=None, tag_th=None, tag_td=None, portrait=True, i_max=0, callback=None):
    """Convert a dataframe to HTML for visualization.
    colors: list[str], 3 elements to define colors for odd, even row background and header background.
        it can also be a dict, where keys are 'odd', 'even', 'header'
    tag_table: dict, defining attribute name-value pairs for TABLE tag
    tag_tr, tag_th, tag_td: dict, name-value pairs for TR, TH, and TD tags
    portrait: boolean, default True, if specify, HTML table is in portrait (columns are columns), good for a long table
        in landscape mode, table columns are HTML rows (good for a fat table, few rows, many columns)
    i_max, int, default 0. If >0, only show up to i_max records. This is to handle huge tables
    callback: a callback method that can modify tags for individual table cell
        the method expect to take the following arguments
            callback(tag, row_idx, col_idx, col_name, df)
        returns a modify tag dict

        Example (color expensive items as orange, otherwise blue:
        t=pd.read_csv('Product.csv')
        t=t[:6]
        def callback(tag, r, c, col, df):
            # header cell has row index -1, so ignore them
            if r>-1 and col=='UnitPrice':
                tag['style']= 'background-color:#fc8d59;' if df.ix[r,c]>20 else 'background-color:#91bfdb;'
            return tag
        print html(t, ["#D4D4BF","#ECECE4","#CCCC99"], callback=callback)
    """
    (S, s_tr, s_td, s_th) = ([], "TR", "", "TH")
    if (tag_table is None or type(tag_table) is not dict):
        tag_table = { 'class': "data_table" }
    if (tag_tr is None or type(tag_tr) is not dict):
        tag_tr = {}
    if (tag_th is None or type(tag_th) is not dict):
        tag_th = {}
    if (tag_td is None or type(tag_td) is not dict):
        tag_td = {}

    def tag2str(tag):
        s="";
        for k,v in tag.items():
            if v=='': continue
            if k=='':
                s +=" "+v
                #for backward compatibility, in case the tag is a str
            elif v!='':
                s+= ' %s="%s"' % (k, v)
        return s

    def merge_tag(old, usr):
        for k,v in usr.items():
            if k in old:
                if v=='':
                    pop(old[k])
                elif (k=='style' and 'background-color:' in old[k]):
                    old[k]=v
                elif v!='':
                    old[k]+=' '+v
            else:
                old[k]=v

    S.append("<TABLE")
    for k,v in tag_table.items():
        S.append(' '+k+'="'+v+'"')
    S.append(">\n")
    if portrait: S.append("  <THEAD>\n")
    S_header=df.columns
    l_colorByClass = False
    BG_COLOR=["#D4D4BF","#ECECE4","#CCCC99"]
    CELL_CLASSES=["data_table_odd","data_table_even","data_table_header"]
    if colors is None:
        BG_COLOR=None
    if (type(colors) is dict):
        l_colorByClass = True
        if 'even' in colors: CELL_CLASSES[1] = colors['even']
        if 'odd' in colors: CELL_CLASSES[0] = colors['odd']
        if 'header' in colors: CELL_CLASSES[2] = colors['header']
    elif (type(colors) is list and len(colors)==3):
        BG_COLOR = colors
    s_tr=tag2str(tag_tr)
    s_th=tag2str(tag_th)
    nRow=len(df)
    nCol=len(df.columns)

    if portrait:
        clr=""
        if l_colorByClass:
            if CELL_CLASSES[2]: clr=' class='+CELL_CLASSES[2]
        elif BG_COLOR is not None and BG_COLOR[2]!='':
            clr=' style="background-color:%s;"' % BG_COLOR[2]
        S.append('    <TR'+s_tr+clr+'>\n')
        for i,s in enumerate(S_header):
            th=tag2str(callback(tag_th.copy(), -1, i, s, df)) if callback is not None else s_th
            S.append('      <TH'+th+'>'+s+'</TH>\n')
        S.append('    </TR>\n')
        S.append('  </THEAD>\n')
        S.append('  <TBODY>\n')
        for i in range(nRow):
            clr=""
            if l_colorByClass:
                if CELL_CLASSES[i%2]: clr=' class='+CELL_CLASSES[i%2]
            elif BG_COLOR is not None and BG_COLOR[i%2]!='':
                clr=' style="background-color:%s;"' % BG_COLOR[i%2]
            S.append('    <TR'+s_tr+clr+'>\n')
            if i_max>0 and i>=i_max:
                S.append('      <TD COLSPAN="'+str(nCol)+'">'+str(nRow-i_max)+' records not displayed ...</TD>\n    </TR>\n')
                break
            for j in range(nCol):
                td={}
                if S_header[j] in tag_td: td = tag_td[S_header[j]]
                if type(td) is str: td={'':td}
                s_td=tag2str(callback(td.copy(), i, j, S_header[j], df) if callback is not None else td)
                S.append('      <TD'+s_td+'>')
                S.append('&nbsp;' if (pd.isnull(df.iloc[i,j]) or str(df.iloc[i,j])=='') else str(df.iloc[i,j]))
                S.append('</TD>\n')
            S.append('    </TR>\n')
        S.append('  </TBODY>\n')
    else:
        for i in range(nCol):
            S.append('  <TR>\n')
            th={}
            if l_colorByClass:
                if CELL_CLASSES[2]: th['class']=CELL_CLASSES[2]
            elif BG_COLOR is not None and BG_COLOR[2]!='':
                th['style']="background-color:%s;" % BG_COLOR[2]
            if callback is not None:
                th=tag2str(callback(th.copy(), -1, i, S_header[i], df))
            else:
                merge_tag(th, tag_th)
                th=tag2str(th)
            S.append('    <TH'+th+'>'+S_header[i]+'</TH>\n')
            td={}
            if S_header[i] in tag_td: td = tag_td[S_header[i]]
            if type(td) is str: td={'':td}
            for j in range(nRow):
                if i_max>0 and j>=i_max:
                    if i==0:
                        S.append('      <TD ROWSPAN="'+str(nCol)+'">'+str(nRow-i_max)+' records not displayed ...</TD>\n')
                    break
                clr={}
                if l_colorByClass:
                    if CELL_CLASSES[j%2]: clr['class']=CELL_CLASSES[j%2]
                elif BG_COLOR is not None and BG_COLOR[j%2]!='':
                    clr['style']="background-color:%s;" % BG_COLOR[j%2]
                merge_tag(clr, td)
                if callback is not None:
                    s_td=tag2str(callback(clr, j, i, S_header[i], df))
                else:
                    s_td=tag2str(clr)
                S.append('    <TD'+s_td+'>')
                S.append('&nbsp;' if (pd.isnull(df.iloc[j,i]) or str(df.iloc[j,i])=='') else str(df.iloc[j,i]))
                S.append('</TD>\n')
            S.append('  </TR>\n')
    S.append('</TABLE>')
    return "".join(S)

# output table in HTML format, with table orientation rotated,
# so that each HTML table row is a column in the table
# This is useful for a slim table (few columns but many rows)
def html2(df, colors=None, tag_table=None, tag_tr=None, tag_th=None, tag_td=None, callback=None):
    """Convert dataframe to HTML in portrait mode"""
    return html(df, colors=colors, tag_table=tag_table, tag_tr=tag_tr, tag_th=tag_th, tag_td=tag_td, portrait=False, callback=callback)

def html3(df, portrait=True, theme="blue", border=1):
    colors=['#eff3ff', '#bdd7e7', '#6baed6']
    if theme=="gold":
        colors=["#D4D4BF","#ECECE4","#CCCC99"]
    elif theme=="orange":
        colors=["#feedde", "#fdbe85", "#fd8d3c"]
    elif theme=="green":
        colors=["#edf8e9", "#bae4b3", "#74c476"]
    elif theme=="purple":
        colors=["#f2f0f7", "#cbc9e2", "#9e9ac8"]
    elif theme=="red":
        colors=["#fee5d9", "#fcae91", "#fb6a4a"]
    elif theme=="gray":
        colors=["#f7f7f7", "#cccccc", "#969696"]
    return html(df, colors=colors, tag_table={'border':str(border)}, portrait=portrait)

def sample_table(l_text=False, nrows=10, ncols=5):
    """For testing convenient, generate a random table to play with
    l_text: boolean, default False, if true, add a text column
    return dataframe, a table of 10 rows, 5 numeric columns"""
    STR="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    L=list(STR)
    if ncols<=26:
        S_col=L[:ncols]
    elif ncols<=26*27:
        S_col=L[:]
        for x in L:
            S_col+=[ _ + "x" for _ in L]
            if len(S_col)>=ncols: break
        S_col=S_col[:ncols]
    else:
        error_msg('Cannot handle more than 702 columns for now')
    t=pd.DataFrame(np.random.randn(nrows,ncols), columns=S_col)
    if l_text:
        L=list(STR.lower())
        if nrows<=26:
            S_row=L[:nrows]
        else:
            S_row=L*(int(nrows/26)+1)
            S_row=S_row[:nrows]
        t['text']=S_row
    return t

def value_type(val):
    """return str: 'r', 'i', or 's' type of variable"""
    if hasattr(val, 'dtype'):
        if np.issubdtype(val.dtype, np.integer): return "i"
        if np.issubdtype(val.dtype, np.floating): return "r"
        if np.issubdtype(val.dtype, np.bool_): return "l"
        return "s"
    else:
        if isinstance(val, int): return "i"
        if isinstance(val, float): return "r"
        if isinstance(val, datetime.date): return "d"
        return "s"

def col_type(df, s_col):
    """s_col: str, column name
    return str: 'r', 'i', or 's' type of column data"""
    #if df[s_col].dtype != np.dtype(object):
    #https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    if np.issubdtype(df[s_col].dtype, np.integer): return "i"
    if np.issubdtype(df[s_col].dtype, np.floating): return "r"
    if np.issubdtype(df[s_col].dtype, np.bool_): return "l"
    if np.issubdtype(df[s_col].dtype, np.datetime64): return "d"
    return "s"

def col_types(df, l_dict=False):
    """return list[str], column types, 'i', 'r', or 's'"""
    S=df.header()
    if l_dict:
        return { x:col_type(df, x) for x in S }
    else:
        return [ col_type(df, s) for s in S ]

def summary(df, n_sample=0):
    """Provide key statistics of a table. If a table has too many rows, use n_sample to randomly pick a subset.
    n_sample: int, defult 0 means all rows. Otherwise, sample n_sample rows for statistics.
    return dataframe, a summary of key statistics for each data column. If table is too large, better off to sample first."""
    if n_sample>0 and n_sample<len(df):
        t=df.loc[df.index[np.random.permutation(len(df))[:n_sample]]]
    else:
        t=df
    S_type=col_types(t)
    #S_min=t.min() may fail when the column is a mix of str and float
    #S_max=t.max()
    S_header=header(t)
    n=len(S_header)
    I_unique=np.zeros(n)
    S_min=['']*n
    S_max=['']*n
    S_most_popular=['']*n
    I_popular=np.zeros(n)
    S_missing=np.zeros(n)
    R_mean=np.zeros(n)
    R_stdv=np.zeros(n)
    for i,x in enumerate(S_header):
        S_min[i]=t[x][pd.notnull(t[x])].min()
        S_max[i]=t[x][pd.notnull(t[x])].max()
        c=unique_count(t[x])
        if len(c)==0: return pd.DataFrame()
        I_unique[i]=len(c)
        L=[(k,v) for k,v in c.items()]
        sorted(L, key=lambda x: -x[1])
        S_most_popular[i]=L[0][0]
        I_popular[i]=L[0][1]
        S_missing[i]=sum(pd.isnull(t[x]))
        if S_type[i]!='s':
            R_mean[i]=t[x].mean()
            R_stdv[i]=t[x].std()
    t_sum=pd.DataFrame(data=list(zip(S_header, S_type, S_missing, S_min, S_max, I_unique, S_most_popular, I_popular, R_mean, R_stdv)),
            columns=['Column', 'Type', '#NULL', 'Min', 'Max', '#Unique', 'MostPopular', '#Popular', 'Mean', 'STDV'])
    return t_sum

class Spotfire:
    """Spotfire-related utility class"""

    @staticmethod
    def new_dxp(template_file, output_file, data_file):
        """Use template_file .dxp, to generate a new file output_file, where the
        datafile inside is changed to point to data_file
        template_file: str, a .dxp template file
        output_file: str, new file name, if not ends with .dxp, .dxp will be appended
        data_file: str, the data file to be used for visualization
        Example: new_dxp('NCC.dxp', 'new.dxp', '~/NCC.csv')"""
        if not os.path.exists(template_file):
            error_msg('Template dose not exist: %s!' % template_file)
        if not output_file.lower().endswith('.dxp'):
            output_file+='.dxp'
        import zipfile
        s_target_file='AnalysisDocument.xml'
        zin=zipfile.ZipFile(template_file, 'r')
        zout=zipfile.ZipFile(output_file, 'w')
        for x in zin.infolist():
            data=zin.read(x.filename)
            if x.filename == s_target_file:
                pat=r'<Field Name="FilePath"><String Id="\d+" Value="(.*?)"\s+/>'
                if is_python3():
                    data=str(data, 'utf-8')
                m=re.search(pat, data)
                if m:
                    # make sure it's windows file path format
                    data_file=format_path(data_file, 'win')
                    oldfile=data[m.start(1):m.end(1)]
                    data=data[:m.start(1)]+data_file+data[m.end(1):]
                    #util.info_msg('Replace old file: %s with new file: %s' % (oldfile, data_file))
                else:
                    error_msg('Tag for FilePath is not found in %s!' % s_target_file);
            zout.writestr(x, data)
        zout.close()
        zin.close()

def to_csv_string(df,**kwargs):
    if is_python3():
        from io import StringIO
        s=StringIO()
    else:
        import StringIO
        s = StringIO.StringIO()
    df.to_csv(s,**kwargs)
    return s.getvalue()

def mask(df, key, value):
    return df[df[key] == value]

def display(df, headers='keys', tablefmt='simple', l_print=True):
    """tablefmt: simple, psql, fancy_grid"""
    from tabulate import tabulate
    if type(df) is pd.Series:
        X=[(k,v) for k,v in df.items()]
        df=pd.DataFrame(X, columns=['Column','Value'])
    s=tabulate(df, headers=headers, tablefmt=tablefmt)
    if l_print:
        print(s)
    else:
        return s

def intersect(a, b, rm_na=False):
    """ return the intersection of two lists """
    if rm_na:
        a=rm_na(a)
        b=rm_na(b)
    return list(set(a) & set(b))

def union(a, b, rm_na=False):
    """ return the union of two lists """
    if rm_na:
        a=rm_na(a)
        b=rm_na(b)
    return list(set(a) | set(b))

def minus(a, b, rm_na=False):
    """ show whats in list b which isn't in list a """
    if rm_na:
        a=rm_na(a)
        b=rm_na(b)
    s = set(b)
    return [x for x in a if x not in b]
    #return list(set(a).difference(set(b)))

def minus_set(a, b, rm_na=False):
    """ show whats in list b which isn't in list a """
    if rm_na:
        a=rm_na(a)
        b=rm_na(b)
    return list(set(a).difference(set(b)))

pd.DataFrame.move_column = move_column
pd.DataFrame.move_before= move_before
pd.DataFrame.move_after= move_after
pd.DataFrame.header = header
pd.DataFrame.col_index = col_index
pd.DataFrame.rename2 = rename2
pd.DataFrame.split = split
pd.DataFrame.melt = melt
pd.DataFrame.csv = df2sdf
pd.DataFrame.html = html
pd.DataFrame.html2 = html2
pd.DataFrame.col_type = col_type
pd.DataFrame.col_types= col_types
pd.DataFrame.summary = summary
pd.DataFrame.display = display
pd.Series.display = display
pd.DataFrame.mask = mask

import time
class StopWatch(object):
    """StopWatch is used to measure time lapse between check() calls
    sw=util.StopWatch()
    calc_A()
    sw.check('A calculated')
    """

    def __init__(self, s_prompt=""):
        """Start the timer"""
        self.start=time.time()
        self.prompt="%s> " % s_prompt
        print(self.prompt+"Start timer ...")

    def check(self, s_msg, l_reset=True):
        """Print the time lapse since start, in addtion to message
        s_msg: str, message to print
        l_reset: boolean, default True. reset timer, so next check() will report the time past between now and next time.
        otherwise, timer is not reset."""
        x=time.time()
        dur=x-self.start
        print(MSG_PREFIX+self.prompt+"Passed: %.1f secs, %s" % (dur, s_msg))
        if l_reset: self.start=x
        return dur

class Progress(object):
    """Progress can monitor the progress of a lengthy computation."""

    def __init__(self, n, func=None, position=0):
        """n: int, total number of items to be process
        func: method. A function that takes an item count [0-n] as input and return percentage [0-1.0]
            e.g., lambda x: (x*1.0/n)*(x/n) is good for O(n2) algorithms
            default, None, linear algorithm
            position: progress bar position, passed to tqdm
        """
        import tqdm
        # see lock issue https://github.com/tqdm/tqdm/issues/461
        tqdm.tqdm.get_lock().locks = []

        self.start=time.time()
        self.n=n # total amount of work
        self.func = func
        # a function that convert i,n into percent progress
        if self.func is None:
            self.func=lambda i: i*1.0/n
        if type(self.func) is str:
            if self.func=='O(n)':
                self.func=lambda i: i*1.0/n
            elif self.func=='O(n2)':
                self.func=lambda i: (i*1.0/n)**2
        self.pg=tqdm.tqdm(total=self.n, position=position)
        #print(("Start processing %d records ..." % self.n))

    def check(self, i, s_msg=''):
        """i: int, index of the current item being processed
        s_msg: str, message to print
        return: print progress statistics, estimate the finish time."""
        i_pass=(time.time()-self.start)/60
        pct=max(self.func(i),1e-6)
        i_remain=abs((1-pct)*(i_pass/pct))
        # percentage, time used, additional time required
        #print(("Processed %.1f%%, used %.1fmin, remain %.1fmin. %s" % (pct*100, i_pass, i_remain, s_msg)))
        if i>self.pg.n: # set progress to 0 may lock the underlyig library
            self.pg.update(max(i-self.pg.n, 0))

def dump_object(obj, s_file='untitled'):
    if os.name == 'nt':
        return dump_object_nt(obj, s_file)
    """use to dump any object
    s_file: str, filename, defaults to 'untitled', filename will be appended with suffix '.pickle'
    no return"""
    if is_python3():
        import pickle
    else:
        import cPickle as pickle
    import gzip
    ext=os.path.splitext(s_file)[1].lower()
    if ext not in ('.gz','.pickle'):
        s_file+='.pickle'
    try:
        tmp=s_file+"_tmp"
        if is_python3():
            if ext=='.gz':
                f=gzip.open(s_file, 'wb')
            else:
                f=open(tmp, 'wb')
        else:
            if ext=='.gz':
                f=gzip.open(s_file, 'w')
            else:
                f=open(tmp, 'w')
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)
        pickle.dump(obj, f)
        fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        # minimize the overlap of mutliple processes trying to read/write to the same file
        if os.path.exists(tmp):
            os.rename(tmp, s_file)
        f.close()

def dump_object_nt(obj, s_file='untitled'):
    """use to dump any object
    s_file: str, filename, defaults to 'untitled', filename will be appended with suffix '.pickle'
    no return"""
    import pickle
    import gzip
    ext=os.path.splitext(s_file)[1].lower()
    if ext not in ('.gz','.pickle'):
        s_file+='.pickle'
    if ext=='.gz':
        f=gzip.open(s_file, 'wb')
    else:
        f=open(s_file, 'wb')
    pickle.dump(obj, f)


def load_object(s_file):
    if os.name == 'nt':
        return load_object_nt(s_file)
    """load a python object from a pickle file (resulted from a previous dump.
    s_file: str, full pickle file name
    return object
    Example: HCIObject.load('cache/mydump.pickle')"""
    if is_python3():
        import pickle
    else:
        import cPickle as pickle
    import gzip
    ext=os.path.splitext(s_file)[1].lower()
    if ext not in ('.gz','.pickle','.pkl'):
        s_file+='.pickle'
    import fcntl
    if is_python3():
        if ext=='.gz':
            f=gzip.open(s_file, 'rb')
        else:
            f=open(s_file, 'rb')
        #https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            x=pickle.load(f)
        except:
            # see https://stackoverflow.com/questions/35879096/pickle-unpicklingerror-could-not-find-mark
            try:
                f.seek(0)
                x=pickle.load(f, encoding='latin1')
            except Exception as e:
                x=None
                print(e)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
    else:
        if ext=='.gz':
            f=gzip.open(s_file, 'r')
        else:
            f=open(s_file)
        #https://stackoverflow.com/questions/29587179/load-pickle-filecomes-from-python3-in-python2
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            x=pickle.load(f)
        except:
            try:
                f.seek(0)
                x=pickle.loads(f)
            except:
                x=None
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
    return x

def load_object_nt(s_file):
    import pickle
    import gzip
    ext=os.path.splitext(s_file)[1].lower()
    if ext not in ('.gz','.pickle'):
        s_file+='.pickle'
    if ext=='.gz':
        f=gzip.open(s_file, 'rb')
    else:
        f=open(s_file, 'rb')
    try:
        x=pickle.load(f)
    except:
        # see https://stackoverflow.com/questions/35879096/pickle-unpicklingerror-could-not-find-mark
        try:
            f.seek(0)
            x=pickle.load(f, encoding='latin1')
        except Exception as e:
            x=None
            print(e)
    finally:
        f.close()
    return x


def fetch_url(url, params=None, method="POST", verify=False):
    """for post method, params should be a dictionary"""
    from urllib.parse import urlencode
    import requests
    params = urlencode(params) if params is not None else None
    if method=="GET":
        if params is not None:
            r = requests.get(url+"?"+params, verify=verify)
        else:
            r = requests.get(url, verify=verify)
    else:
        r = requests.post(url, data=params)
    return (r.text, r.status_code)


#http://a-ma.us/wp/2011/05/getpost-to-a-url-in-python/
def _obsolete_fetch_url(url, params, method="POST"):
    """Fetch URL page"""
    if is_python3():
        from urllib.parse import urlencode
        from urllib.request import urlopen
    else:
        from urllib import urlencode, urlopen
    params = urlencode(params)
    if method=="GET":
        f = urlopen(url+"?"+params)
    else:
        # Usually a POST
        f = urlopen(url, params)
    return (f.read(), f.code)
#url = "http://google.com"
#method = "POST"
#params = {"Param1": "Value1"}
# Fetch the content and response code
#[content, response_code] = fetch_url(url, params, method)
# Check if the server responded with a success code (200)
#if (response_code == 200):
#  print content
#else:
#  print response_code

def get_url(s_url, l_text=False, s_file=None):
    try:
        if is_python3():
            import requests
            resp=requests.get(s_url, verify=False, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'})
            s=resp.content
            #s=urllib.request.urlopen(
            #    urllib.request.Request(s_url,
            #        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'},
            #        verify=False)
            #    ).read()
            if l_text:
                s=s.decode('utf-8', 'ignore')
        else:
            import urllib2
            s= urllib2.urlopen(s_url).read()
        if s_file is not None:
            #print(s_file, s)
            if l_text:
                #print(s_file, s)
                save_string(s_file, s)
            else:
                with open(s_file, 'bw') as f:
                    f.write(s)
                    f.flush()
        return s
    except Exception as e:
        print(e)
        return None

def mkdir(path):
    #https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    # don't throw except if dir already exists.
    # in parallel code, there is a small chance where both threads don't think dir exists
    # and try to create it
    if is_python3(): return os.makedirs(path, exist_ok=True)
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def email(fromaddr, toaddr, subject, body, HOST='localhost', attachments=None, html=True):
    """ Do not use this one
        Email via smtp directly
        HOST='localhost', if sent from host
        HOST='172.17.0.1' if sent within docker container
        # see configuration http://satishgandham.com/2016/12/sending-email-from-docker-through-postfix-installed-on-the-host/
        toaddr: comma space concatenated if there are multiple
    """
    #https://stackoverflow.com/questions/3362600/how-to-send-email-attachments
    import smtplib
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate

    #msg = MIMEText(msg, 'html' if html else 'plain')
    msg = MIMEMultipart()
    msg['from'] = fromaddr
    msg['subject'] = subject
    toaddr=COMMASPACE.join(toaddr) if type(toaddr) is list else toaddr
    msg['to'] = toaddr
    msg['Date']= formatdate(localtime=True)
    msg.attach(MIMEText(body, 'html' if html else 'plain'))
    if attachments is not None and type(attachments) is str:
        attachments=[attachments]

    for f in attachments or []:
        with open(f, "rb") as fil:
            part=MIMEApplication( fil.read(), Name=os.path.basename(f) )
        part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
        msg.attach(part)

    smtp = smtplib.SMTP(HOST)
    smtp.set_debuglevel(0) #https://docs.python.org/3/library/smtplib.html
    #print(fromaddr, toaddr, msg.as_string(), attachments)
    # https://stackoverflow.com/questions/8856117/how-to-send-email-to-multiple-recipients-using-python-smtplib
    # SMTP.sendmail, on the other hand, sets up the "envelope" of the message for the SMTP protocol. It needs a Python list of strings, each of which has a single address.
    smtp.sendmail(fromaddr, toaddr.split(COMMASPACE), msg.as_string())
    smtp.close()

def sendmail(fromaddr, toaddr, subject, body):
    """Using sendmail command"""
    from email.mime.text import MIMEText
    from subprocess import Popen, PIPE
    from email.utils import COMMASPACE, formatdate

    msg = MIMEText(body)
    msg["From"] = fromaddr
    msg['subject'] = subject
    toaddr=COMMASPACE.join(toaddr) if type(toaddr) is list else toaddr
    msg["To"] = toaddr
    msg["Subject"] = subject
    msg["Content-Type"] = "text/html"

    p = Popen(["sendmail", "-t", "-oi"], stdin=PIPE)
    p.communicate(msg.as_bytes())

def webmail(fromaddr, toaddr, subject, body, html=True):
    import requests,urllib
    html="Y" if html else "N"
    subject=urllib.parse.quote(subject)
    body=urllib.parse.quote(body)
    resp=requests.get(f'https://uslj.prd.nibr.novartis.net/apps/crontab/pub/email?from={fromaddr}&to={toaddr}&subject={subject}&body={body}&html={html}', verify=False)
    return resp.text.startswith('ok')

if __name__=="__main__":
    t=pd.read_csv('Product.csv')
    t=t[:6]
    def callback(tag, r, c, col, df):
        if r>-1 and col=='UnitPrice':
            tag['style']= 'background-color:#fc8d59;' if df.loc[r,c]>20 else 'background-color:#91bfdb;'
        if r>-1 and col=='Discontinued':
            tag['style']= 'background-color: #999999;' if df.loc[r,c] else 'background-color:#af89dc;'
        return tag

    print((html(t, ["#D4D4BF","#ECECE4","#CCCC99"], callback=callback)))
