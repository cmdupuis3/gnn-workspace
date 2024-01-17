
from dataclasses import dataclass
from typing import Iterable

@dataclass
class Scenario:
    conv_var: Iterable[str]
    input_var: Iterable[str]
    target: Iterable[str]
    name: str

target = ['U', 'V']

sc1  = Scenario(['SSH'],                     ['TAUX','TAUY'],     target, name='SSH_2ps_pointTAUXTAUY_3l_402010_nf80')
sc2  = Scenario(['SSH'],                     ['X'],               target, name='SSH_2ps_pointX_3l_402010_nf80')
sc3  = Scenario(['SSH','SST'],               ['X'],               target, name='SSHSST_2ps_pointX_3l_402010_nf80')
sc4  = Scenario(['SSH'],                     ['X','TAUX','TAUY'], target, name='SSH_2ps_pointXTAUXTAUY_3l_402010_nf80')
sc5  = Scenario(['SSH','SST'],               ['X','TAUX','TAUY'], target, name='SSHSST_2ps_pointXTAUXTAUY_3l_402010_nf80')
sc6  = Scenario(['SSH','X','Y','Z'],         ['TAUX','TAUY'],     target, name='SSHXYZ_2ps_pointTAUXTAUY_3l_402010_nf80')
sc7  = Scenario(['SSH','X'],                 ['TAUX','TAUY'],     target, name='SSHX_2ps_pointTAUXTAUY_3l_402010_nf80')
sc8  = Scenario(['SSH','SST','X','Y','Z'],   ['TAUX','TAUY'],     target, name='SSHSSTXYZ_2ps_pointTAUXTAUY_3l_402010_nf80')
sc9  = Scenario(['SSH','SST','X'],           ['TAUX','TAUY'],     target, name='SSHSSTX_2ps_pointTAUXTAUY_3l_402010_nf80')
sc10 = Scenario(['SSH','TAUX','TAUY'],       ['X'],               target, name='SSHTAUXTAUY_2ps_pointX_3l_402010_nf80')
sc11 = Scenario(['SSH','SST','TAUX','TAUY'], ['X'],               target, name='SSHSSTTAUXTAUY_2ps_pointX_3l_402010_nf80')
sc12 = Scenario(['SSH','SST'],               ['X','dx','dy','TAUX','TAUY'], target, name='SSHSST_2ps_pointXdxdyTAUXTAUY_3l_402010_nf80')
