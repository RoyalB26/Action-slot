"""Canonical TACO benchmark action labels and definitions."""

# 64 Actor Action classes + 1 Background class (Total: 65)
TACO_ACTOR_LABELS = [
    'c:z1-z2', 'c:z1-z3', 'c:z1-z4',
    'c:z2-z1', 'c:z2-z3', 'c:z2-z4',
    'c:z3-z1', 'c:z3-z2', 'c:z3-z4',
    'c:z4-z1', 'c:z4-z2', 'c:z4-z3',
    'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
    'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
    'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
    'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
    'b:z1-z2', 'b:z1-z3', 'b:z1-z4',
    'b:z2-z1', 'b:z2-z3', 'b:z2-z4',
    'b:z3-z1', 'b:z3-z2', 'b:z3-z4',
    'b:z4-z1', 'b:z4-z2', 'b:z4-z3',
    'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
    'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
    'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
    'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
    'p:c1-c2', 'p:c1-c4', 
    'p:c2-c1', 'p:c2-c3', 
    'p:c3-c2', 'p:c3-c4', 
    'p:c4-c1', 'p:c4-c3', 
    'p+:c1-c2', 'p+:c1-c4', 
    'p+:c2-c1', 'p+:c2-c3', 
    'p+:c3-c2', 'p+:c3-c4', 
    'p+:c4-c1', 'p+:c4-c3',
    'bg'
]

# 7 Ego Vehicle Action classes
TACO_EGO_LABELS = [
    "Stop",
    "Go / Accelerate",
    "Turn Left",
    "Turn Right",
    "Lane Change Left",
    "Lane Change Right",
    "Slow Down / Brake",
]
