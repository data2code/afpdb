#!/usr/bin/env python
import pymol2
import pymol.util as pyutil
import os,ssl
from .afpdb import util
os.environ["http_proxy"] = "http://nibr-proxy.global.nibr.novartis.net:2011"
os.environ["https_proxy"] = os.environ["http_proxy"]
os.environ["ftp_proxy"] = os.environ["http_proxy"]
ssl._create_default_https_context = ssl._create_unverified_context

class PyMOL:

    def __init__(self):
        self.p=pymol2.PyMOL()
        self.p.start()

    def __del__(self):
        self.close()

    def close(self):
        if self.p is not None and hasattr(self.p, '_stop'):
            self.p.stop()
        self.p=None

    def __call__(self, cmd, *args, **kwargs):
        # use API
        try:
            f=getattr(self.p.cmd, cmd)
        except:
            print(f"Invalid PyMOL command: {cmd}")
        return f(*args, **kwargs)

    def cmd(self, cmd_str):
        """cmd_str can be one command or multi-line string
        empty line or a line starts with # will be ignored"""
        # use pymol command line string
        def is_empty(s): s=='' or s.startswith('#')
        out=None
        if "\n" in cmd_str:
            for _cmd in cmd_str.split("\n"):
                _cmd=_cmd.strip()
                if is_empty(_cmd): continue
                out=self.p.cmd.do(_cmd)
            return out
        cmd_str=cmd_str.strip()
        if not is_empty(cmd_str):
            out=self.p.cmd.do(cmd_str)
        return out

    def script(self, fn):
        S=util.read_list(fn)
        out=None
        for cmd_str in S:
            cmd_str=cmd_str.strip()
            if cmd_str=='' or cmd_str.stratswith('#'): continue
            out=self.cmd(cmd_str)
        return out

if __name__=="__main__":
    # see example_pymol.py in the same folder for a better example
    x=PyMOL()
    # using API
    out=x.cmd("""
fetch 8a47, myobj
save x.pdb
color red
select a, /myobj//A
color white, a
save x.png
""")
