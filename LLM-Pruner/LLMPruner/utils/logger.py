import os
import sys
import time
import codecs
import logging


class LoggerWithDepth():
    def __init__(self, env_name, config, root_dir = 'runtime_log', overwrite = True, setup_sublogger = True, rank = None):
        if os.path.exists(os.path.join(root_dir, env_name)) and not overwrite:
            raise Exception("Logging Directory {} Has Already Exists. Change to another name or set OVERWRITE to True".format(os.path.join(root_dir, env_name)))
        
        self.env_name = env_name
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, env_name)
        self.overwrite = overwrite
        self.rank = rank

        self.format = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                        "%Y-%m-%d %H:%M:%S")

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        if setup_sublogger:
            #sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            self.setup_sublogger(config)

    def setup_sublogger(self,sub_config):
        self.sub_dir = self.log_dir

        # Setup File/Stream Writer
        log_format=logging.Formatter("%(asctime)s - %(levelname)s :       %(message)s", "%Y-%m-%d %H:%M:%S")
        
        self.writer = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.sub_dir, "Result.log"))
        fileHandler.setFormatter(log_format)
        self.writer.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_format)
        self.writer.addHandler(consoleHandler)
        
        self.writer.setLevel(logging.INFO)  

    def log(self, info):
        self.writer.info(info)

    def write_description_to_folder(self, file_name, config):
        with codecs.open(file_name, 'w') as desc_f:
            desc_f.write("- Training Parameters: \n")
            for key, value in config.items():
                desc_f.write("  - {}: {}\n".format(key, value))
