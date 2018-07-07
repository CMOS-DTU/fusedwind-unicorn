import numpy as np
import copy

# The following are helper functions to create a custom interface
#################################################################

def create_interface():

    return {'output': {}, 'input': {}}

def set_variable(inner_dict, variable):

    inner_dict[variable['name']]=copy.deepcopy(variable)

def set_input(fifc, variable):

    set_variable(fifc['input'], variable)

def set_output(fifc, variable):

    set_variable(fifc['output'], variable)

def extend_interface(base, extension):

    for k, v in extension['input'].items():
        set_input(base, v)

    for k, v in extension['output'].items():
        set_output(base, v)

    return base

'''
# Consider adding to simplify including inputs into FUSED_Objects
def fusedvar(name,val,desc='',shape=None):

    return {'name' : name, 'val' : val, 'desc' : desc, 'shape' : shape}
'''

# The following are helper functions to help objects implement interfaces
#########################################################################

class FUSED_Object(object):

    def __init__(self):

        super(FUSED_Object,self).__init__()
        self.interface = create_interface()

    def implement_fifc(self, fifc, **kwargs):

        for k, v in fifc['input'].items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'])

            # Add our parameter
            self.add_input(**v)

        for k, v in fifc['output'].items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'])

            # add out output
            self.add_output(**v)

    def add_input(self, **kwargs):

        set_input(self.interface, kwargs)

    def add_output(self, **kwargs):

        set_output(self.interface, kwargs)
