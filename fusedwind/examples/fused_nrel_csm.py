"""
FUSED csm example

Copyright (c) NREL. All rights reserved.
"""

# NREL CSM model set
from nrelcsm.nrel_csm import aep_csm, tcc_csm, bos_csm, opex_csm, fin_csm

# FUSED helper functions and interface defintions
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
from fusedwind.windio_plant_costs import fifc_aep, fifc_tcc_costs, fifc_bos_costs, fifc_opex, fifc_finance

import numpy as np

### FUSED-wrapper file 
class aep_csm_fused(FUSED_Object):

    def __init__(self, object_name_in='unnamed_dummy_object'):

        super(aep_csm_fused, self).__init__(object_name_in)

        self.aep_csm_assembly = aep_csm()

    def _build_interface(self):

        self.implement_fifc(fifc_aep) # pulls in variables from fused-wind interface (not explicit)

        # Add model specific inputs
        self.add_input(**{'name': 'max_tip_speed', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'max_power_coefficient', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'opt_tsr', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'cut_in_wind_speed', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'cut_out_wind_speed', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'air_density', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'max_efficiency', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'thrust_coefficient', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'soiling_losses', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'array_losses', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'availability', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'shear_exponent', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'wind_speed_50m', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'weibull_k', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'hub_height', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'altitude', 'val' : 0.0, 'type' : float})

        # Add model specific outputs
        self.add_output(**{'name': 'rated_wind_speed', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rated_rotor_speed', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rotor_thrust', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rotor_torque', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'power_curve', 'val' : np.zeros(161), 'type' : float, 'shape' : (161,)})
        self.add_output(**{'name': 'gross_aep', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'capacity_factor', 'val' : 0.0, 'type' : float})

    def compute(self, inputs, outputs):


        self.aep_csm_assembly.compute(inputs['machine_rating'], inputs['max_tip_speed'], inputs['rotor_diameter'], inputs['max_power_coefficient'], inputs['opt_tsr'],
                inputs['cut_in_wind_speed'], inputs['cut_out_wind_speed'], inputs['hub_height'], inputs['altitude'], inputs['air_density'],
                inputs['max_efficiency'], inputs['thrust_coefficient'], inputs['soiling_losses'], inputs['array_losses'], inputs['availability'],
                inputs['turbine_number'], inputs['shear_exponent'], inputs['wind_speed_50m'], inputs['weibull_k'])

        outputs['rated_wind_speed'] = self.aep_csm_assembly.aero.rated_wind_speed
        outputs['rated_rotor_speed'] = self.aep_csm_assembly.aero.rated_rotor_speed
        outputs['rotor_thrust'] = self.aep_csm_assembly.aero.rotor_thrust
        outputs['rotor_torque'] = self.aep_csm_assembly.aero.rotor_torque
        outputs['power_curve'] = self.aep_csm_assembly.aero.power_curve
        outputs['gross_aep'] = self.aep_csm_assembly.aep.gross_aep
        outputs['net_aep'] = self.aep_csm_assembly.aep.net_aep
        outputs['capacity_factor'] = self.aep_csm_assembly.aep.capacity_factor


### FUSED-wrapper file 
class tcc_csm_fused(FUSED_Object):

    def __init__(self, offshore=False, advanced_blade=True, drivetrain_design='geared', \
                       crane=True, advanced_bedplate=0, advanced_tower=False, object_name_in='unnamed_dummy_object'):

        super(tcc_csm_fused, self).__init__(object_name_in)

        self.offshore = offshore 
        self.advanced_blade = advanced_blade 
        self.drivetrain_design = drivetrain_design 
        self.crane = crane 
        self.advanced_bedplate = advanced_bedplate  
        self.advanced_tower = advanced_tower

        self.tcc = tcc_csm()

    def _build_interface(self):

        self.implement_fifc(fifc_tcc_costs) # pulls in variables from fused-wind interface (not explicit)

        # Add model specific inputs
        self.add_input(**{'name': 'rotor_thrust', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'rotor_torque', 'val' : 0.0, 'type' : float}) 
        self.add_input(**{'name': 'year', 'val' : 2010, 'type' : int})
        self.add_input(**{'name': 'month', 'val' : 12, 'type' : int})

        # Add model specific outputs
        self.add_output(**{'name': 'rotor_cost', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rotor_mass', 'val' : 0.0, 'type' : float}) 
        self.add_output(**{'name': 'turbine_mass', 'val' : 0.0, 'type' : float}) 

    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        rotor_diameter = inputs['rotor_diameter']
        hub_height = inputs['hub_height']
        blade_number = inputs['blade_number']
        rotor_thrust = inputs['rotor_thrust']
        rotor_torque = inputs['rotor_torque']
        year = inputs['year']
        month = inputs['month']

        self.tcc.compute(rotor_diameter, machine_rating, hub_height, rotor_thrust, rotor_torque, \
                year, month, blade_number, self.offshore, self.advanced_blade, self.drivetrain_design, \
                self.crane, self.advanced_bedplate, self.advanced_tower)

        # Outputs
        outputs['turbine_cost'] = self.tcc.turbine_cost 
        outputs['turbine_mass'] = self.tcc.turbine_mass
        outputs['rotor_cost'] = self.tcc.rotor_cost
        outputs['rotor_mass'] = self.tcc.rotor_mass
        
        return outputs


### FUSED-wrapper file 
class bos_csm_fused(FUSED_Object):

    def __init__(self, object_name_in='unnamed_dummy_object'):

        super(bos_csm_fused, self).__init__(object_name_in)

        self.implement_fifc(fifc_bos_costs) # pulls in variables from fused-wind interface (not explicit)

        self.bos = bos_csm()

    def _build_interface(self):

        # Add model specific inputs
        self.add_input(**{'name': 'sea_depth', 'val' : 0.0, 'type' : float}) # = Float(20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.add_input(**{'name': 'year', 'val' : 2010, 'type' : int}) # = Int(2009, iotype='in', desc='year for project start')
        self.add_input(**{'name': 'month', 'val' : 12, 'type' : int}) # = Int(12, iotype = 'in', desc= 'month for project start')
        self.add_input(**{'name': 'multiplier', 'val' : 0.0, 'type' : float}) # = Float(1.0, iotype='in')

        # Add model specific outputs
        self.add_output(**{'name': 'bos_breakdown_development_costs', 'val' : 0.0, 'type' : float}) #  = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
        self.add_output(**{'name': 'bos_breakdown_preparation_and_staging_costs', 'val' : 0.0, 'type' : float}) #  = Float(desc='Site preparation and staging')
        self.add_output(**{'name': 'bos_breakdown_transportation_costs', 'val' : 0.0, 'type' : float}) #  = Float(desc='Any transportation costs to site / staging site') #BOS or turbine cost?
        self.add_output(**{'name': 'bos_breakdown_foundation_and_substructure_costs', 'val' : 0.0, 'type' : float}) # Float(desc='Foundation and substructure costs')
        self.add_output(**{'name': 'bos_breakdown_electrical_costs', 'val' : 0.0, 'type' : float}) # Float(desc='Collection system, substation, transmission and interconnect costs')
        self.add_output(**{'name': 'bos_breakdown_assembly_and_installation_costs', 'val' : 0.0, 'type' : float}) # Float(desc='Assembly and installation costs')
        self.add_output(**{'name': 'bos_breakdown_soft_costs', 'val' : 0.0, 'type' : float}) # = Float(desc='Contingencies, bonds, reserves, decommissioning, profits, and construction financing costs')
        self.add_output(**{'name': 'bos_breakdown_other_costs', 'val' : 0.0, 'type' : float}) # = Float(desc='Bucket for any other costs not captured above')

    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        rotor_diameter = inputs['rotor_diameter']
        hub_height = inputs['hub_height']
        RNA_mass = inputs['RNA_mass']
        turbine_cost = inputs['turbine_cost']

        turbine_number = inputs['turbine_number']
        sea_depth = inputs['sea_depth']
        year = inputs['year']
        month = inputs['month']
        multiplier = inputs['multiplier']

        self.bos.compute(machine_rating, rotor_diameter, hub_height, RNA_mass, turbine_cost, turbine_number, sea_depth, year, month, multiplier)

        # Outputs
        outputs['bos_costs'] = self.bos.bos_costs #  = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')
        #self.add_output(bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
        outputs['bos_breakdown_development_costs'] = self.bos.bos_breakdown_development_costs #  = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
        outputs['bos_breakdown_preparation_and_staging_costs'] = self.bos.bos_breakdown_preparation_and_staging_costs #  = Float(desc='Site preparation and staging')
        outputs['bos_breakdown_transportation_costs'] = self.bos.bos_breakdown_transportation_costs #  = Float(desc='Any transportation costs to site / staging site') #BOS or turbine cost?
        outputs['bos_breakdown_foundation_and_substructure_costs'] = self.bos.bos_breakdown_foundation_and_substructure_costs # Float(desc='Foundation and substructure costs')
        outputs['bos_breakdown_electrical_costs'] = self.bos.bos_breakdown_electrical_costs # Float(desc='Collection system, substation, transmission and interconnect costs')
        outputs['bos_breakdown_assembly_and_installation_costs'] = self.bos.bos_breakdown_assembly_and_installation_costs # Float(desc='Assembly and installation costs')
        outputs['bos_breakdown_soft_costs'] = self.bos.bos_breakdown_soft_costs  # = Float(desc='Contingencies, bonds, reserves, decommissioning, profits, and construction financing costs')
        outputs['bos_breakdown_other_costs'] = self.bos.bos_breakdown_other_costs # = Float(desc='Bucket for any other costs not captured above')

        return outputs


### FUSED-wrapper file 
class opex_csm_fused(FUSED_Object):

    def __init__(self, object_name_in='unnamed_dummy_object'):
        super(opex_csm_fused, self).__init__(object_name_in)

        self.implement_fifc(fifc_opex)

        self.opex = opex_csm()

    def _build_interface(self):

        # Add model specific inputs
        self.add_input(**{'name': 'sea_depth', 'val' : 0.0, 'type' : float}) # #20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.add_input(**{'name': 'year', 'val' : 2010, 'type' : int}) # = Int(2009, iotype='in', desc='year for project start')
        self.add_input(**{'name': 'month', 'val' : 12, 'type' : int}) # iotype = 'in', desc= 'month for project start') # units = months
        self.add_input(**{'name': 'net_aep', 'val' : 0.0, 'type' : float}) # units = 'kW * h', iotype = 'in', desc = 'annual energy production for the plant')

        # Add model specific outputs
        self.add_output(**{'name': 'opex_breakdown_preventative_opex', 'val' : 0.0, 'type' : float}) # desc='annual expenditures on preventative maintenance - BOP and turbines'
        self.add_output(**{'name': 'opex_breakdown_corrective_opex', 'val' : 0.0, 'type' : float}) # desc='annual unscheduled maintenance costs (replacements) - BOP and turbines'
        self.add_output(**{'name': 'opex_breakdown_lease_opex', 'val' : 0.0, 'type' : float}) # desc='annual lease expenditures'
        self.add_output(**{'name': 'opex_breakdown_other_opex', 'val' : 0.0, 'type' : float}) # desc='other operational expenditures such as fixed costs'

    def compute(self, inputs, outputs):

        self.opex.compute(inputs['sea_depth'], inputs['year'], inputs['month'],
                          inputs['turbine_number'], inputs['machine_rating'], inputs['net_aep'])

        outputs['avg_annual_opex'] = self.opex.avg_annual_opex
        outputs['opex_breakdown_preventative_opex'] = self.opex.opex_breakdown_preventative_opex
        outputs['opex_breakdown_corrective_opex'] = self.opex.opex_breakdown_corrective_opex
        outputs['opex_breakdown_lease_opex'] = self.opex.opex_breakdown_lease_opex
        outputs['opex_breakdown_other_opex'] = self.opex.opex_breakdown_other_opex


### FUSED-wrapper file 
class fin_csm_fused(FUSED_Object):

    def __init__(self,fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0, object_name_in='unnamed_dummy_object'):

        super(fin_csm_fused, self).__init__(object_name_in)

        self.fin = fin_csm(fixed_charge_rate, construction_finance_rate, tax_rate, discount_rate, \
                      construction_time, project_lifetime)

    def _build_interface(self):

        self.implement_fifc(fifc_finance) # pulls in variables from fused-wind interface (not explicit)

        self.add_input(**{'name': 'sea_depth', 'val' : 0.0, 'type' : float}) # #20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')

        self.add_output(**{'name': 'lcoe', 'val' : 0.0, 'type' : float}) 

    def compute(self, inputs, outputs):

        turbine_cost = inputs['turbine_cost']
        turbine_number = inputs['turbine_number']
        bos_costs = inputs['bos_costs']
        avg_annual_opex = inputs['avg_annual_opex']
        net_aep = inputs['net_aep']
        sea_depth = inputs['sea_depth']

        self.fin.compute(turbine_cost, turbine_number, bos_costs, avg_annual_opex, net_aep, sea_depth)

        # Outputs
        outputs['coe'] = self.fin.coe 
        outputs['lcoe'] = self.fin.lcoe 

        return outputs


### Full NREL cost and scaling model LCOE assembly and problem execution
#########################################################################

def example_lcoe():

    TCC = tcc_csm_fused(object_name_in='TCC') # object name in???
    AEP = aep_csm_fused(object_name_in='AEP')
    BOS = bos_csm_fused(object_name_in='BOS')
    OPEX = opex_csm_fused(object_name_in='OPEX')
    FIN = fin_csm_fused(object_name_in ='FIN')
    
    MR = Independent_Variable(5000.0, 'machine_rating',object_name_in='MR') # would be nice to add a group of Ind Vars together like in OMDAO
    RD = Independent_Variable(126.0, 'rotor_diameter', object_name_in='RD')
    HH = Independent_Variable(90.0, 'hub_height', object_name_in='HH')
    TN = Independent_Variable(100.0, 'turbine_number', object_name_in='TN')
    Y = Independent_Variable(2009.0, 'year', object_name_in='Y')
    M = Independent_Variable(12.0, 'month', object_name_in='M')
    SD = Independent_Variable(20.0, 'sea_depth', object_name_in='SD')

    # would be nice to be able to connect multiple obejcts at the same time
    AEP.connect(MR)
    AEP.connect(RD)
    AEP.connect(HH)
    AEP.connect(TN)
    AEP.connect(Y)
    AEP.connect(M)
    AEP.connect(SD)

    TCC.connect(MR)
    TCC.connect(RD)
    TCC.connect(HH)
    TCC.connect(TN)
    TCC.connect(Y)
    TCC.connect(M)
    TCC.connect(SD)

    OPEX.connect(MR)
    OPEX.connect(RD)
    OPEX.connect(HH)
    OPEX.connect(TN)
    OPEX.connect(Y)
    OPEX.connect(M)
    OPEX.connect(SD)

    BOS.connect(MR)
    BOS.connect(RD)
    BOS.connect(HH)
    BOS.connect(TN)
    BOS.connect(Y)
    BOS.connect(M)
    BOS.connect(SD)

    FIN.connect(MR)
    FIN.connect(RD)
    FIN.connect(HH)
    FIN.connect(TN)
    FIN.connect(Y)
    FIN.connect(M)
    FIN.connect(SD)

    # source and destination counterintuitive; connecting would seem like appending
    TCC.connect(AEP)
    OPEX.connect(AEP)
    FIN.connect(AEP)
    BOS.connect(TCC)
    FIN.connect(TCC)
    FIN.connect(BOS)
    FIN.connect(OPEX)

    # Now I want to set a bunch of model specific inputs that I dont want to specify as independent variables
    '''
    # set inputs
    # simple test of module
    # Turbine inputs
    prob['rotor_diameter'] = 126.0
    prob['blade_number'] = 3
    prob['hub_height'] = 90.0    
    prob['machine_rating'] = 5000.0

    # Rotor force calculations for nacelle inputs
    maxTipSpd = 80.0
    maxEfficiency = 0.90201
    ratedWindSpd = 11.5064
    thrustCoeff = 0.50
    airDensity = 1.225

    ratedHubPower  = prob['machine_rating'] / maxEfficiency 
    rotorSpeed     = (maxTipSpd/(0.5*prob['rotor_diameter'])) * (60.0 / (2*np.pi))
    prob['rotor_thrust']  = airDensity * thrustCoeff * np.pi * prob['rotor_diameter']**2 * (ratedWindSpd**2) / 8
    prob['rotor_torque'] = ratedHubPower/(rotorSpeed*(np.pi/30))*1000
    
    prob['year'] = 2009
    prob['month'] = 12

    # AEP inputs
    prob['max_tip_speed'] = 80.0 #Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    prob['max_power_coefficient'] = 0.488 #Float(iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
    prob['opt_tsr'] = 7.525 #Float(iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
    prob['cut_in_wind_speed'] = 3.0 #Float(units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
    prob['cut_out_wind_speed'] = 25.0 #Float(units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
    prob['altitude'] = 0.0 #Float(units = 'm', iotype='in', desc= 'altitude of wind plant')
    prob['air_density'] = 1.225 #Float(units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
    prob['max_efficiency'] = 0.902 #Float(iotype='in', desc = 'maximum efficiency of rotor and drivetrain - at rated power')
    prob['thrust_coefficient'] = 0.5 #Float(iotype='in', desc='thrust coefficient at rated power')
    prob['soiling_losses'] = 0.0
    prob['array_losses'] = 0.1
    prob['availability'] = 0.941
    prob['turbine_number'] = 100
    prob['shear_exponent'] = 0.1
    prob['wind_speed_50m'] = 8.02
    prob['weibull_k']= 2.15

    # Finance, BOS and OPEX inputs
    prob['RNA_mass'] = 256634.5 # RNA mass is not used in this simple model
    prob['sea_depth'] = 20.0
    prob['multiplier'] = 1.0
    '''

    work_flow_objects = get_execution_order([TCC, AEP, BOS, OPEX, FIN, MR, RD, HH, TN, Y, M, SD])

    print('Calculate LCOE for default values')
    print(AEP.get_output_value())

if __name__ == '__main__':

    example_lcoe()

