from . import extractors

'''
compo_name:什么操作
model_name:该种操作下何种算法,如提取特征点是SIFT or superPoint or ...
config:config
'''
def load_component(compo_name,model_name,config):
    if compo_name =='extractor':
        component = load_extractor(model_name, config)
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
    