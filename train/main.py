import yaml
from config import get_config, print_unage




def main(config, model_config):
    # if config.modle_name=='SGM': #选择模型(如果有多个模型)
        # model_config = SGM_Modle(model_config)

    #dataloader
    
        

if __name__ == '__main__':

    config, unparsed = get_config()
    with open(config.config_path) as f:
        model_config = yaml.safe_load(f)
        print(model_config)
    
    if len(unparsed) > 0:
        print_unage()
        exit(1)

    main(config, model_config)