import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)

#读入CSV文件
'''
要读取的CSV格式的文件路径
"D:\cgs\File\CSVtb_api_2019_04_01_11_44_23.csv"
"D:\cgs\File\CSVtb_api_apk_map_2019_04_01_11_44_23.csv"
"D:\cgs\File\CSVtb_apk_2019_04_01_11_44_23.csv"
"D:\cgs\File\CSVtb_authority_2019_04_01_11_44_23.csv"
"D:\cgs\File\CSVtb_authority_apk_map_2019_04_01_11_44_23.csv"

'''

'''
读入CSV文件
'''
def init():
    pd.set_option('display.width', None)
    tb_api = pd.read_csv("D:/cgs/File/CSV/tb_api_2019_04_01_13_39_19.csv")
    tb_api_apk_map = pd.read_csv("D:/cgs/File/CSV/tb_api_apk_map_2019_04_01_13_39_19.csv")
    tb_apk = pd.read_csv("D:/cgs/File/CSV/tb_apk_2019_04_01_13_39_19.csv")
    tb_authority = pd.read_csv("D:/cgs/File/CSV/tb_authority_2019_04_01_13_39_19.csv")
    tb_authority_apk_map = pd.read_csv("D:/cgs/File/CSV/tb_authority_apk_map_2019_04_01_13_39_19.csv")

    # 显示读入的原始数据
    #print(tb_api)
    #print(tb_api_apk_map)
    #print(tb_apk)
    #print(tb_authority)
    #print(tb_authority_apk_map)

    #合并表,目标：将权限特征、api特征、应用属性放在一张表中
    #这张表中有apkid apiid apicontent apiMD5
    tb1=pd.merge(tb_api_apk_map,tb_api,on=["api_id","api_id"])
    #将tb1与tb_apk表进行合并
    #这张表中有apk_id packageName apiid apicontent apiMD5
    tb2=pd.merge(tb1,tb_apk,on=["apk_id","apk_id"])
    #将tb_authority和tb_authority_apk_map表合并
    # 这个表中有 apkID authorityID 权限内容 权限MD5
    tb3=pd.merge(tb_authority,tb_authority_apk_map,on=["authority_id","authority_id"])
    # 将tb2和tb3合并，根据apkid
    tb4=pd.merge(tb2,tb3,on=["apk_id","apk_id"])


    #显示合并后的表
    # print(tb4)
#     找到api 权限和属性之间的关系
    table=pd.crosstab(tb4["api_content"],tb4["apk_attribute"])

#     输出最终的数据
    print(table)

init()

