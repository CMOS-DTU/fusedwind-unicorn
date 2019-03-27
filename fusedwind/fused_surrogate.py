from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import linear_model
from sklearn.externals import joblib
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
import numpy as np
# CMOS Thoughts on the surrogate capabilities:
#   The surrogate object should when finished be able to connect to independent variables and outputs. The "Model" part is what the power user defines. This should take a numpy array(matrix) and spit out predictions(vector) in a function .get_prediction(np_input) further more it should include a field(list) calles .input_names which allows the surrogate object to know where to connect which inputs. Notice that the order of this list is essential!
# The model might have an output name. Otherwise this is simply 'prediction'
# Example of very simple surrogate:

#class simple_surrogate(object):
#    def __init__(self):
#        self.input_names = ['x','y']
#        self.output_name = ['prediction']
#
#    def get_prediction(np_input):
#        return [sum(np_input,axis=1)]


class FUSED_Surrogate(FUSED_Object):
    def __init__(self, object_name_in='unnamed_surrogate_object',state_version_in=None,model_in=None):
        super(FUSED_Surrogate, self).__init__(object_name_in, state_version_in)

        self.model = None
        self.model_input_names = None
        self.model_output_names = ['prediction']
        self.output_name = None

        if not model_in is None:
            self.set_model(model_in)

    def _build_interface(self):
        if self.model is None:
            print('No model connected yet. The interface is still empty')
        else:
            for var in self.model_output_names:
                self.add_output(var)
            for var in self.model_input_names:
                self.add_input(var)
    
    #Class to set the model. The model_obj is described above.
    def set_model(self,model_obj):
        self.model = model_obj
        if not hasattr(self.model,'input_names'):
            raise Exception('The model has no attribute .input_names. Add this to the model object to couple it to fused_wind')
        self.model_input_names = self.model.input_names
        if hasattr(self.model,'output_names'):
            self.model_output_names = self.model.output_names
        else:
            print('The model has no outputname. The output name of the fused-object is by default \'prediction\'')

    #Fused wind way of getting output:
    def compute(self,inputs,outputs):
        if self.model is None:
            raise Exception('No model has been connected')

        np_array = np.empty((1,len(inputs)))
        for n, var in enumerate(self.model_input_names):
            if not var in inputs:
                raise Exception('The name {} was not found in the input variables. Syncronize the connected independent variables and the model.input_names to ensure that the model is connect correctly')
            np_array[0,n] = inputs[var]
        prediction = self.model.get_prediction(np_array)
        for n, var in enumerate(self.model_output_names):
            outputs[var] = prediction[n]

## ----------- The rest of the classes are CMOS customized surrogates. They can be used as default, but are a bit complicated to modify and NOT seen as an integrated part of fused wind.

class Multi_Fidelity_Surrogate(object):
    
    def __init__(self, cheap=None, exp=None, intersections=None):
        if not cheap[1]  == exp[1] and cheap[2] == exp[2]:
            raise Exception('The input and output keys should be the same and in the same order.. Other cases are not implemented yet!!')
        else:
            self.input_names = cheap[1]
            self.output_name = cheap[2]

        self.output_names = ['prediction','sigma']

        self.data_set_object_cheap = cheap[0]
        self.data_set_object_exp = exp[0]

        built = 'False'

    def build_model(self):
        self.cheap_input = self.data_set_object_cheap.get_numpy_array(self.input_names)
        self.cheap_output = self.data_set_object_cheap.get_numpy_array(self.output_name)

        self.exp_input = self.data_set_object_exp.get_numpy_array(self.input_names)
        self.exp_output = self.data_set_object_cheap.get_numpy_array(self.output_name)

        ###########
        self.exp_correction = self.exp_output
        ###########

        self.cheap_linear_model = Linear_Model(self.cheap_input,self.cheap_output)
        self.cheap_linear_model.build()
        
        linear_prediction_cheap = self.cheap_linear_model.get_prediction(self.cheap_input)
        linear_remainder = self.cheap_output-linear_prediction_cheap

        self.cheap_GP_model = Kriging_Model(self.cheap_input,linear_remainder)
        self.cheap_GP_model.build()

        self.exp_correction_linear_model = Linear_Model(self.exp_input,self.exp_correction)
        self.exp_correction_linear_model.build()

        linear_prediction_correction = self.exp_correction_linear_model.get_prediction(self.exp_input)
        linear_remainder_correction = self.exp_correction-linear_prediction_correction

        self.exp_correction_GP_model = Kriging_Model(self.exp_input,linear_remainder_correction)
        self.exp_correction_GP_model.build()

        built = 'True'
 
    #Fast way of getting a prediction on a np-array data set: 
    def get_prediction_matrix(self,input):
        prediction = self.cheap_linear_model.get_prediction(input)+self.cheap_GP_model.get_prediction(input)+self.exp_correction_linear_model.get_prediction(input)+self.exp_correction_GP_model.get_prediction(input)
        return prediction

    #Single prediction for the fused wind coupling:
    def get_prediction(self,input):
        prediction = self.cheap_linear_model.get_prediction(input)+self.cheap_GP_model.get_prediction(input)+self.exp_correction_linear_model.get_prediction(input)+self.exp_correction_GP_model.get_prediction(input)
        sigma = self.cheap_GP_model.get_sigma(input)**2+self.exp_correction_GP_model.get_sigma(input)**2
        sigma = sigma**0.5
        return [prediction,sigma]

#Single fidelity surrogate object:
class Single_Fidelity_Surrogate(object):

    def __init__(self, input=None, output=None, dataset=None):
        self.input = input
        self.output = output

        self.output_names = ['prediction','sigma']
        
        built = 'False'

    def build_model(self):
        self.linear_model = Linear_Model(self.input,self.output)
        self.linear_model.build()

        linear_prediction = self.linear_model.get_prediction(self.input)
        linear_remainder = self.output-linear_prediction

        self.GP_model = Kriging_Model(self.input,linear_remainder)
        self.GP_model.build()

        built = 'True'

    def get_prediction_matrix(self,input):
        prediction = self.linear_model.get_prediction(input)+self.GP_model.get_prediction(input)
        return prediction

    def get_prediction(self,input):
        prediction = self.linear_model.get_prediction(input)+self.GP_model.get_prediction(input)
        sigma = self.GP_model.get_sigma(input)
        return [prediction,sigma]

    def do_LOO(self, extra_array_input=None, extra_array_output=None):
        error_array = do_LOO(self, extra_array_input, extra_array_output)
        return error_array

def do_LOO(self, extra_array_input=None, extra_array_output=None):
    '''
    This function creates a Leave One Out error calculation on the input/output data of the model.
    The function takes an extra data array of test-cases.
    '''
    full_input = self.input
    full_output = self.output  
    error_array = []

    #Doing leave one out test:
    for ind in range(len(self.input[:,0])):
        test_input = np.array([full_input[ind]])
        test_output = full_output[ind]

        self.input = np.delete(full_input,ind,0)
        self.output = np.delete(full_output,ind)

        self.build_model()
        predicted_output = self.get_prediction(test_input)[0]

        error_array.append(np.abs(predicted_output-test_output))

    self.input = full_input
    self.output = full_output
    self.build_model()
    
    #Testing error on extra points:
    if not extra_array_input is None:
        for ind, test_input in enumerate(extra_array_input):
            TI = np.array([test_input])
            TO = extra_test_array_output[ind]
            predicted_output = self.get_prediction(TI)[0]
            error_array.append(np.abs(predicted_output-TO))

    return np.mean(error_array)

class Linear_Model(linear_model.LinearRegression):
 
    def __init__(self,input=None,output=None,linear_order=3):
        super(Linear_Model, self).__init__()
        self.input = input
        self.output = output
        self.linear_order = linear_order
        self.is_build = 'False'

    def set_input(self,input=None):
        self.input = input

    def set_output(self,output=None):
        self.output = output

    def build(self):
        if self.input is None or self.output is None:
            print('#ERR# Input or output is not defined')
        else:
            self.covariates = self.create_covariates(self.input,self.linear_order)
            self.fit(self.covariates,self.output)
            self.is_build = 'True'

    def get_prediction(self,input):
        if input is self.input:
            return self.predict(self.covariates)
        else:
            covariates = self.create_covariates(input,self.linear_order)
            return self.predict(covariates)

    def create_covariates(self,input_matrix,order):
        # -------------- Linear model covariate building ------------------
        # Number of variables:
        n_var = np.size(input_matrix[0,:])
        orderGroups = {}
        orderGroups[1] = np.array(range(1,n_var+1))
        orderGroups[1] = orderGroups[1][np.newaxis,:]
        # insert constant and first order term:
        covariates = np.concatenate([np.vstack(np.ones(len(input_matrix[:,0]))),input_matrix],axis=1)
        # Going up through the orders of the covariates:
        for order_loop in range(2,order+1):
            # Going through all the existing covariates to add one factor:
            for covariate_loop in  range(0,np.size(orderGroups[order_loop-1][0,:])):
                # Inserting a variable at the end of the existing covariates. This creates dublicates, which are removed on a later stage:
                for variable_loop in range(0,n_var):
                    testGroup = np.vstack(np.sort(np.concatenate([orderGroups[order_loop-1][:,covariate_loop],[variable_loop+1]])))
                    #Adding the first number to the testgroup:
                    if not order_loop in orderGroups:
                        orderGroups[order_loop] = testGroup
                        covariate = covariates[:,testGroup[0]]
                        for k in testGroup[1:]:
                            covariate = covariate*covariates[:,k]
                        covariate = np.concatenate([covariates,covariate],axis=1)
                    else:
                        #Running through the existing covariates of same order to see if it is a duplicate:
                        for kernels in range(0,np.size(orderGroups[order_loop][0,:])):
                            if np.array_equal(testGroup, np.vstack(orderGroups[order_loop][:,kernels])):
                                break
                            #If we are at the end of the line:
                            if kernels is np.size(orderGroups[order_loop][0,:])-1:
                                orderGroups[order_loop] = np.concatenate([orderGroups[order_loop],testGroup],axis=1)
                                covariate = covariates[:,testGroup[0]]
                                for k in testGroup[1:]:
                                    covariate = covariate*covariates[:,k]
                                covariates = np.concatenate([covariates,covariate],axis=1)
        return covariates

class Kriging_Model(GaussianProcessRegressor):
    
    def __init__(self,input=None,output=None):
        super(Kriging_Model, self).__init__(input,output)
        self.input = input
        self.output = output

        self.kernel = C(1.0, (1e-2,1e2))*RBF(10,(1e-3,1e3))
        self.n_restarts_optimizer = 9
        self.optimizer = 'fmin_l_bfgs_b'
        self.alpha = 1e-6
        self.normalize_y = True
        self.copy_X_train = False
        self.random_state = None
        self.is_build = 'False'

    def build(self):
        if self.input is None or self.output is None:
            print('#ERR# Input for Kriging model not defined')
        else:
            self.fit(self.input,self.output)
        self.is_build = 'True'

    def get_prediction(self,input):
        return self.predict(input)

    def get_sigma(self,input):
        prediction, sigma = self.predict(input,return_std=True)
        return sigma
