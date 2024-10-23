import sys, os
print(sys.path)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'LMRRfactory')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'LMRRfactory/src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'cantera/build/python')))
print(sys.path)