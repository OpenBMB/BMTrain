

DEBUG_VARS = {}

def clear():
    global DEBUG_VARS
    DEBUG_VARS = {}

def set(key, value):
    global DEBUG_VARS
    DEBUG_VARS[key] = value

def get(key):
    global DEBUG_VARS
    return DEBUG_VARS[key]

def append(key, value):
    global DEBUG_VARS
    if key not in DEBUG_VARS:
        DEBUG_VARS[key] = []
    DEBUG_VARS[key].append(value)

def extend(key, value):
    global DEBUG_VARS
    if key not in DEBUG_VARS:
        DEBUG_VARS[key] = []
    DEBUG_VARS[key].extend(value)