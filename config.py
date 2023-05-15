import os
import argparse

class Config(object):
    """model config"""
    def __init__(self):
        self.mode = None
        self.pretrained_model = None
        self.train_data_path = None
        self.eval_data_path = None
        self.test_data_paht = None
        self.output_path = None
        self.per_device_train_batch_size = None
        self.per_device_eval_batch_size = None
        self.num_train_epochs = None
        self.learning_rate = None
        self.parser = self.setup_parser()
        # 将参数转换成字典形式
        self.args = vars(self.parser.parse_args())
        # 使用args字典更新__dict__字典，__dict__字典包含了当前类中属性和对应值的字典形式的键值对
        self.__dict__.update(self.args)
        
    def setup_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-mode', help='train/eval/test', type=str, default='train')
        parser.add_argument('-pretrained_model', help='pretrained model name or path', type=str, default='./PLM/bert-base-chinese')
        parser.add_argument('-train_data_path', help='train data path', type=str, default='./datasets/qa_train.jsonl')
        parser.add_argument('-eval_data_path', help='eval data path', type=str, default='./datasets/qa_dev.jsonl')
        parser.add_argument('-test_data_path', help='test data path', type=str)
        parser.add_argument('-output_path', help='output model path', type=str, default='./output/')
        parser.add_argument('-per_device_train_batch_size', help='train batch size per device', type=int, default=1)
        parser.add_argument('-per_device_eval_batch_size', help='eval batch size per device', type=int, default=1)
        parser.add_argument('-num_train_epochs', type=int, default=1)
        parser.add_argument('-learning_rate', help='learning rate', type=float, default=1e-5)
        parser.add_argument("-save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
        parser.add_argument("-save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
        return parser

if __name__ == '__main__':
    config = Config()
    print(config.mode)