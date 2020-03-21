'''
@name:datasource.py
'''
import requests,json,time
import pandas as pd
from sqlalchemy import create_engine
from copy import deepcopy
from datetime import date,timedelta
import warnings
warnings.filterwarnings('ignore')

class DataSource(object):
    conn = create_engine('mysql+pymysql://用户名:密码@localhost:3306/数据库')

    #爬取数据
    def getData(self):
        #json字符数据源
        url = 'https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5'
        headers = {
            'user-agent': '.....'
        }
        #获取数据
        r = requests.get(url,headers)
        #数据是str类型,需要对数据转义
        res = json.loads(r.text)
        #获取的是子标签data,数据都在data中
        data_res = json.loads(res['data'])
        return data_res



    #处理数据
    def proData(self):
        #拷贝传值  引用传递
        data_dict = self.getData()

        chinaTotal = deepcopy(data_dict['chinaTotal'])
        chinaAdd = deepcopy(data_dict['chinaAdd'])
        country = deepcopy(data_dict['areaTree'])
        province = deepcopy(data_dict['areaTree'])
        province_city = deepcopy(data_dict['areaTree'])

        #中国综合数据
        chinaTotal['create_dt'] = data_dict['lastUpdateTime']
        chinaTotal = pd.DataFrame([chinaTotal])
        #to_sql()插入到数据表中,如果create_dt是string类型插入会报错
        chinaTotal.create_dt = pd.to_datetime(chinaTotal.create_dt)

        #每日新增
        chinaAdd['create_dt'] = data_dict['lastUpdateTime']
        chinaAdd = pd.DataFrame([chinaAdd])
        chinaAdd.create_dt = pd.to_datetime(chinaAdd.create_dt)

        #各个国家的数据
        #先将中国的子数据集删除掉，只保留国家的数据
        country[0].pop('children')
        datasets = list()
        for c in country:
            info = dict()
            info['country_name'] = c['name']
            info['today_confirm'] = c['today']['confirm']
            info['total_confirm'] = c['total']['confirm']
            info['suspect'] = c['total']['suspect']
            info['dead'] = c['total']['dead']
            info['deadRate'] = c['total']['deadRate']
            info['heal'] = c['total']['heal']
            info['healRate'] = c['total']['healRate']
            info['create_dt'] = data_dict['lastUpdateTime']
            datasets.append(info)
        country = pd.DataFrame(datasets)
        country.create_dt = pd.to_datetime(country.create_dt)

        #各个省份
        province = province[0].pop('children')
        provinces = list()
        for p in range(len(province)):
            province[p].pop('children')
            info = dict()
            info['province_name'] = province[p]['name']
            info['today_confirm'] = province[p]['today']['confirm']
            info['total_confirm'] = province[p]['total']['confirm']
            info['suspect'] = province[p]['total']['suspect']
            info['dead'] = province[p]['total']['dead']
            info['heal'] = province[p]['total']['heal']
            info['deadRate'] = province[p]['total']['deadRate']
            info['healRate'] = province[p]['total']['healRate']
            info['create_dt'] = data_dict['lastUpdateTime']
            provinces.append(info)
        province = pd.DataFrame(provinces)
        province.create_dt = pd.to_datetime(province.create_dt)

        #各个城市的数据
        province_city = province_city[0]['children']

        cites = list()
        for p in range(len(province_city)):
            city = province_city[p]['children']
            for c in range(len(city)):
                info = dict()
                info['province_name'] = province_city[p]['name']
                info['city_name'] = city[c]['name']
                info['today_confirm'] = city[c]['today']['confirm']
                info['total_confirm'] = city[c]['total']['confirm']
                info['suspect'] = city[c]['total']['suspect']
                info['dead'] = city[c]['total']['dead']
                info['heal'] = city[c]['total']['heal']
                info['deadRate'] = city[c]['total']['deadRate']
                info['healRate'] = city[c]['total']['healRate']
                info['create_dt'] = data_dict['lastUpdateTime']
                cites.append(info)
        cites = pd.DataFrame(cites)
        cites.create_dt = pd.to_datetime(cites.create_dt)

        return chinaTotal,chinaAdd,country,province,cites

    #导出数据
    def importData(self):
        #获取数据
        chinaTotal, chinaAdd, country, province, cites = self.proData()

        #执行导入
        chinaTotal.to_sql(name='chinaTotal',con=self.conn,if_exists='append',index=False,chunksize=100)
        chinaAdd.to_sql(name='chinaAdd', con=self.conn, if_exists='append', index=False, chunksize=100)
        country.to_sql(name='country', con=self.conn, if_exists='append', index=False, chunksize=100)
        province.to_sql(name='province', con=self.conn, if_exists='append', index=False, chunksize=100)
        cites.to_sql(name='city', con=self.conn, if_exists='append', index=False, chunksize=100)

if __name__ == '__main__':
    ins = DataSource()
    # 今天爬过的数据就不要再爬取了
    today = date.today()
    conn = create_engine('mysql+pymysql://用户名:z密码@localhost:3306/数据库')
    sql = f"select id from chinatotal where date(create_dt)='{today}'"
    data = pd.read_sql(sql,conn)
    if data.shape[0] == 0:
        ins.importData()
