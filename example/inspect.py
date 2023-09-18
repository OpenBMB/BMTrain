from contextlib import contextmanager
from bmtrain import CheckpointBlock
import sys

@contextmanager
def custom_redirection(fileobj):
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

def look_var(layer, _, output):
    try:
        print(f"{layer.__name__}: {output.min()}")
    except:
        print(f"{layer.__name__}:{output[0].min()}")

def lookup_output(model,layers=set()):
    
    for key,layer in model.named_modules():
        layer.__name__ = key
        if layer not in layers:
            layers.add(layer)
        else:
            continue
        if len(layer._modules) !=0:
            layer.register_forward_hook(look_var)
            lookup_output(layer,layers)
        else:
            layer.register_forward_hook(look_var)
