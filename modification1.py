import numpy as np

def unitStep(v):
  if v >= 0:
    return 1
  else:
    return 0

def perceptronModel(x, w, b):
  v = np.dot(w, x) + b
  y = unitStep(v)
  return y

def NOT_logicFunction(x):
  wNOT = -1
  bNOT = 0.5
  return perceptronModel(x, wNOT, bNOT)

def AND_logicFunction(x):
  w = np.array([1, 1])
  bAND = -1.5
  return perceptronModel(x, w, bAND)

def OR_logicFunction(x):
  w = np.array([1, 1])
  bOR = -0.5
  return perceptronModel(x, w, bOR)

# Unique modification: Adding a NAND logic function
def NAND_logicFunction(x):
  w = np.array([-1, -1])  # The weights for NAND (inverse of AND)
  bNAND = 1.5  # A positive bias value for NAND
  return perceptronModel(x, w, bNAND)

# Modify XOR to use NAND gate
def XOR_logicFunction(x):
  y1 = NAND_logicFunction(x)  # Use NAND instead of AND-OR combination
  y2 = OR_logicFunction(x)
  final_x = np.array([y1, y2])
  finalOutput = AND_logicFunction(final_x)
  return finalOutput

# Test the modified XOR function
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])

print("XOR({}, {}) = {}".format(0, 1, XOR_logicFunction(test1)))
print("XOR({}, {}) = {}".format(1, 1, XOR_logicFunction(test2)))
print("XOR({}, {}) = {}".format(0, 0, XOR_logicFunction(test3)))
print("XOR({}, {}) = {}".format(1, 0, XOR_logicFunction(test4)))