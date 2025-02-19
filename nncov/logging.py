
import logging
import os
def get_logger(name: str, path: str, to_stdout=False):
    logger = logging.getLogger(name)
    # 移除根日志记录器的所有处理器
    for handler in logger.root.handlers[:]:
        logger.root.removeHandler(handler)
    # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s %(levelname)s]: %(message)s')
    
    # 创建一个handler，用于写入日志文件
    if os.path.exists(path):
        os.remove(path)
    fh = logging.FileHandler(path, mode='w+', encoding='utf-8')
    # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
    fh.setFormatter(formatter)
    # 给logger对象绑定文件操作符
    logger.addHandler(fh)
    
    if to_stdout:
        # 再创建一个handler用于输出到控制台
        ch = logging.StreamHandler()
        # 为控制台操作符绑定格式（可以绑定多种格式例ch.setFormatter(formatter2)）
        ch.setFormatter(formatter)
        # 给logger对象绑定文件操作符
        logger.addHandler(ch)
    
    # 定义日志输出层级
    logger.setLevel(logging.DEBUG)
    return logger
