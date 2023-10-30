from . import extractors
from . import readers
from . import matchers
from . import evaluators

'''
compo_name:什么操作
model_name:该种操作下何种算法,如提取特征点是SIFT or superPoint or ...
config:config
'''
def load_component(compo_name,model_name,config):
    if compo_name =='extractor':
        component = load_extractor(model_name, config)
    elif compo_name=='reader':
        component=load_reader(model_name,config)
    elif compo_name=='matcher':
        component=load_matcher(model_name,config)
    elif compo_name=='evaluator':
        component=load_evaluator(model_name,config)
    else:
        raise NotImplementedError
    return component





#提取特征点
def load_extractor(model_name,config):
    if model_name == 'root':
        extractor = extractors.ExtractSIFT(config)
    else:
        raise NotImplementedError
    return extractor
    


def load_matcher(model_name,config):
    if model_name=='SGM':
        matcher=matchers.GNN_Matcher(config,'SGM')
    elif model_name=='SG':
        matcher=matchers.GNN_Matcher(config,'SG')
    elif model_name=='NN':
        matcher=matchers.NN_Matcher(config)
    else:
        raise NotImplementedError
    return matcher

def load_reader(model_name,config):
    if model_name=='standard':
        reader=readers.standard_reader(config)
    else:
        raise NotImplementedError
    return reader

def load_evaluator(model_name,config):
    if model_name=='AUC':
        evaluator=evaluators.auc_eval(config)
    elif model_name=='FM':
        evaluator=evaluators.FMbench_eval(config)
    return evaluator