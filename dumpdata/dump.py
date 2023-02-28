import argparse
import yaml
import os 
import sys
#将字符转换为布尔值
def str2bool(v):
    return v.lower() in ("true", "1")

parser = argparse.ArgumentParser(description="dump data")
# parser.add_argument("--config_path", type=str, default='/home/freesix/SGM-1/dumpdata/configs/gl3d.yaml')
parser.add_argument("--config_path", type=str, default='configs/gl3d.yaml')


args = parser.parse_args()#获取配置
config_path = args.config_path

#这个是读取文件绝对路径，以便仿真时从argparse中直接传入yaml文件路径, 当然传入yaml文件路径为绝对路径也可
fileRealPath = os.path.split(os.path.realpath(__file__))[0]
# ROOT_DIR = os.path.join(fileRealPath, '..')
# sys.path.insert(0, fileRealPath)
Config_path = os.path.join(fileRealPath, config_path)
#用来获取dump对象并重新命名，如此处导入的就是gl3d.yaml中data_name，即导入gl3d_train中gl3d类
def get_dumper(name):
    print(name)
    mod = __import__('dumper.{}'.format(name), fromlist=[''])
    print(mod)
    print(getattr(mod, name))
    return getattr(mod, name)



if __name__ == '__main__':
    with open(Config_path, 'r') as f:
        print(f)
        config = yaml.safe_load(f)

    dataset = get_dumper(config['data_name'])(config)

    dataset.initialize()#类中的initialize方法(获取数列,创c建相应文件和文件夹)
    if config['extractor']['extract']:
        dataset.dump_feature()#提取特征并保存
    dataset.format_dump_data()#将得到的数据的所有相关信息统一保存