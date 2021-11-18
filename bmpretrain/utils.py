import sys
from typing import Any, Dict, Iterable
from .global_var import config

ALIGN = 4
ROW_WIDTH = 60

def round_up(x, d):
    return (x + d - 1) // d * d

def print_dict(title : str, content : Dict[str, Any], file=sys.stdout):
    max_kw_len = max([ len(kw) for kw in content.keys() ])
    max_kw_len = round_up(max_kw_len + 3, 4)

    raw_content = ""

    for kw, val in content.items():
        raw_content += kw + " :" + " " * (max_kw_len - len(kw) - 2)
        raw_val = "%s" % val
        
        len_val_row = ROW_WIDTH - max_kw_len
        st = 0
        if len(raw_val) == 0:
            raw_val = " "
        while st < len(raw_val):
            if st > 0:
                raw_content += " " * max_kw_len
            raw_content += raw_val[st: st + len_val_row] + "\n"
            st += len_val_row
    
    print_block(title, raw_content, file)


def print_block(title : str, content : str, file=sys.stdout):
    left_title = (ROW_WIDTH - len(title) - 2) // 2
    right_title = ROW_WIDTH - len(title) - 2 - left_title
    
    print("=" * left_title + " " + title + " " + "=" * right_title, file=file)
    print(content, file=file)
    
def print_rank(*args, rank=0, **kwargs):
    if config["rank"] == rank:
        print(*args, **kwargs)
