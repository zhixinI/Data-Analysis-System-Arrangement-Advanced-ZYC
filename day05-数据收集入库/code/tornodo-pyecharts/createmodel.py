'''
@name : createmodel.py
创建数据库模型的;
'''
from sqlalchemy.ext.declarative import declarative_base  #父类模型
from sqlalchemy.dialects.mysql import DECIMAL,FLOAT,INTEGER,DATETIME,VARCHAR  #数据类型
from sqlalchemy import Column  #创建字段
from sqlalchemy import create_engine

Base = declarative_base()
metadata = Base.metadata   #元类

#中国综合数据
class chinaTotal(Base):
    #指定表名称
    __tablename__="chinaTotal"
    #主键
    id = Column(INTEGER,primary_key=True,autoincrement=True)
    #累计确诊
    confirm = Column(INTEGER,comment="累计确诊")
    #累计死亡
    dead = Column(INTEGER,comment="累计死亡")
    #累计治愈
    heal = Column(INTEGER,comment="累计治愈")
    #现有确诊
    nowConfirm = Column(INTEGER,comment="现有确诊")
    #现有疑似
    suspect = Column(INTEGER,comment="现有疑似")
    #现有重症
    nowSevere = Column(INTEGER,comment="现有重症")
    #境外输入
    importedCase = Column(INTEGER,comment="境外输入")
    #更新时间
    create_dt = Column(DATETIME,comment="更新时间")


#每日新增
class chinaAdd(Base):
    # 指定表名称
    __tablename__ = "chinaAdd"
    # 主键
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    # 每日新增累计确诊
    confirm = Column(INTEGER, comment="每日新增累计确诊")
    # 每日新增累计死亡
    dead = Column(INTEGER, comment="每日新增累计死亡")
    # 每日新增累计治愈
    heal = Column(INTEGER, comment="每日新增累计治愈")
    # 每日新增现有确诊
    nowConfirm = Column(INTEGER, comment="每日新增现有确诊")
    # 每日新增现有疑似
    suspect = Column(INTEGER, comment="每日新增现有疑似")
    # 每日新增现有重症
    nowSevere = Column(INTEGER, comment="每日新增现有重症")
    # 每日新增境外输入
    importedCase = Column(INTEGER, comment="每日新增境外输入")
    # 每日新增更新时间
    create_dt = Column(DATETIME, comment="更新时间")

#各个国家的数据
class country(Base):
    # 指定表名称
    __tablename__ = "country"
    # 主键
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    #国家名称
    country_name = Column(VARCHAR(64),comment='国家名称')
    #今日新增加
    today_confirm = Column(INTEGER,comment='今日新增加')
    #累计确诊
    total_confirm = Column(INTEGER,comment='累计确诊')
    # 每日新增现有疑似
    suspect = Column(INTEGER, comment="现有疑似")
    # 累计死亡
    dead = Column(INTEGER, comment="累计死亡")
    #当前死亡率
    deadRate = Column(DECIMAL(5,2),comment='当前死亡率')
    # 累计治愈
    heal = Column(INTEGER, comment="累计治愈")
    #治愈率
    healRate = Column(DECIMAL(5,2),comment='治愈率')
    # 更新时间
    create_dt = Column(DATETIME, comment="更新时间")

class province(Base):
    # 指定表名称
    __tablename__ = "province"
    # 主键
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    #省份名称
    province_name = Column(VARCHAR(64),comment='省份名称')
    #今日新增加
    today_confirm = Column(INTEGER,comment='今日新增加')
    #累计确诊
    total_confirm = Column(INTEGER,comment='累计确诊')
    # 每日新增现有疑似
    suspect = Column(INTEGER, comment="现有疑似")
    # 累计死亡
    dead = Column(INTEGER, comment="累计死亡")
    #当前死亡率
    deadRate = Column(DECIMAL(5,2),comment='当前死亡率')
    # 累计治愈
    heal = Column(INTEGER, comment="累计治愈")
    #治愈率
    healRate = Column(DECIMAL(5,2),comment='治愈率')
    # 更新时间
    create_dt = Column(DATETIME, comment="更新时间")

class city(Base):
    # 指定表名称
    __tablename__ = "city"
    # 主键
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    #省份名称
    province_name = Column(VARCHAR(64),comment='省份名称')
    # 省份名称
    city_name = Column(VARCHAR(64), comment='城市名称')
    #今日新增加
    today_confirm = Column(INTEGER,comment='今日新增加')
    #累计确诊
    total_confirm = Column(INTEGER,comment='累计确诊')
    # 每日新增现有疑似
    suspect = Column(INTEGER, comment="现有疑似")
    # 累计死亡
    dead = Column(INTEGER, comment="累计死亡")
    #当前死亡率
    deadRate = Column(DECIMAL(5,2),comment='当前死亡率')
    # 累计治愈
    heal = Column(INTEGER, comment="累计治愈")
    #治愈率
    healRate = Column(DECIMAL(5,2),comment='治愈率')
    # 更新时间
    create_dt = Column(DATETIME, comment="更新时间")

if __name__ == '__main__':
    #1.导入数据库连接驱动
    #2.配置一下连接信息
    mysql_configs = dict(
        db_host="127.0.0.1", #主机地址
        db_name="",    #数据库名称
        db_port=3306,        #数据库端口
        db_user="",      #数据库用户
        db_pwd="",     #数据库密码
        db_code="utf8"       #数据连接编码
    )

    #3.连接格式:mysql+驱动名称://用户:密码@主机:端口/数据库名称?charset=编码
    link = "mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}?charset={db_code}".format(**mysql_configs)

    #4.建立引擎对象 建立表的的时候输出日志
    engine = create_engine(link,encoding="utf-8",echo=True)

    #5.把类模型映射成mysql->数据库->表
    #create database `NZ1904`;
    metadata.create_all(engine)