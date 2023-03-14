//文件说明
boundary为边界识别模块，论文《A joint model for entity boundary detection and entity span recognition》
entity_detection为实体识别模块，论文《A Controlled Attention for Nested Named Entity Recognition》，train_entity 文件运行实体识别代码
Data为数据文件，add_cues_data为加入控制注意力标签后的数据格式文件，
boundary_data为边界信息文件，其中均为正确边界，test数据集边界信息并未给出，可依据boundary模块自行运行输出。
config文件为参数配置文件
log文件为日志文件
