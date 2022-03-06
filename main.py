import os
import math
import time
import requests
import itertools

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup


class Lottery():
    def __init__(self, url):
        self.url = url
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        self.history_dir = os.path.join('data', 'history', self.__class__.__name__)

        self.payload = {
            '__VIEWSTATE': soup.select_one('#__VIEWSTATE')['value'],
            '__VIEWSTATEGENERATOR': soup.select_one('#__VIEWSTATEGENERATOR')['value'],
            '__EVENTVALIDATION': soup.select_one('#__EVENTVALIDATION')['value']
        }

        self.year_key_name = ''
        self.month_key_name = ''

    def parse_html_soup(self, soup):
        data_list = list()

        for table_index, table in enumerate(soup.select('table.td_hm')):
            draw_term = table.find(id=f'SuperLotto638Control_history1_dlQuery_DrawTerm_{table_index}')
            date = table.find(id=f'SuperLotto638Control_history1_dlQuery_Date_{table_index}')

            data_dict = {'draw_term': draw_term.text, 'date': date.text}

            for number_index in range(1, 8):
                number = table.find(id=f'SuperLotto638Control_history1_dlQuery_SNo{number_index}_{table_index}')
                data_dict[f'number_{number_index}'] = number.text.strip()

            data_list.append(data_dict)

        return pd.DataFrame(data_list)

    def download_history(self, start_date, end_date):
        os.makedirs(self.history_dir, exist_ok=True)

        for pd_period in pd.period_range(start=start_date, end=end_date, freq='M'):
            self.payload[self.year_key_name] = str(pd_period.year - 1911)
            self.payload[self.month_key_name] = str(pd_period.month)

            time.sleep(3)
            r = requests.post(self.url, data=self.payload)
            soup = BeautifulSoup(r.text, 'html5lib')

            self.parse_html_soup(soup).to_csv(os.path.join(self.history_dir, f'{pd_period}.csv'), index=False)

    def get_history_data(self, start_date, end_date):
        data = pd.DataFrame()

        for pd_period in pd.period_range(start=start_date, end=end_date, freq='M'):
            try:
                csv_file = os.path.join(self.history_dir, f'{pd_period}.csv')
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(e)
                continue

            df['date'] = df.apply(lambda x: x['date'].replace(x['date'][0:3], str(int(x['date'][0:3]) + 1911)), axis=1)
            df['date'] = pd.to_datetime(df['date'])

            data = pd.concat([df, data], ignore_index=True)

        try:
            return data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        except KeyError:
            return pd.DataFrame()


class SuperLotto638(Lottery):
    def __init__(self):
        url = 'https://www.taiwanlottery.com.tw/Lotto/SuperLotto638/history.aspx'
        super().__init__(url)

        self.payload['SuperLotto638Control_history1$chk'] = 'radYM'
        self.payload['SuperLotto638Control_history1$btnSubmit'] = '查詢'
        self.year_key_name = 'SuperLotto638Control_history1$dropYear'
        self.month_key_name = 'SuperLotto638Control_history1$dropMonth'

    def lotto_prize(self):
        prize_dict = {
            'match_1_1': 100,
            'match_3_0': 100,
            'match_2_1': 200,
            'match_3_1': 400,
            'match_4_0': 800,
            'match_4_1': 4000,
            'match_5_0': 20000,
            'match_5_1': 150000
        }

        # prize_pool = math.comb(38, 6) * 8 * 100 * 0.55
        #
        # total_1_1 = prize_dict['match_1_1'] * math.comb(6, 1) * math.comb(32, 5)
        # total_3_0 = prize_dict['match_3_0'] * math.comb(6, 3) * math.comb(32, 3) * 7
        # total_2_1 = prize_dict['match_2_1'] * math.comb(6, 2) * math.comb(32, 4)
        # total_3_1 = prize_dict['match_3_1'] * math.comb(6, 3) * math.comb(32, 3)
        # total_4_0 = prize_dict['match_4_0'] * math.comb(6, 4) * math.comb(32, 2) * 7
        # total_4_1 = prize_dict['match_4_1'] * math.comb(6, 4) * math.comb(32, 2)
        # total_5_0 = prize_dict['match_5_0'] * math.comb(6, 5) * math.comb(32, 1) * 7
        # total_5_1 = prize_dict['match_5_1'] * math.comb(6, 5) * math.comb(32, 1)
        #
        # p = prize_pool - total_1_1 - total_3_0 - total_2_1 - total_3_1 - total_4_0 - total_4_1 - total_5_0 - total_5_1
        # prize_dict['match_6_0'] = int(p * 0.11 / 7)
        # prize_dict['match_6_1'] = int(p * 0.89)

        prize_dict['match_6_0'] = 11782100
        prize_dict['match_6_1'] = 667295335

        return prize_dict

    def match_prize(self, history_data, combination_1, combination_2):
        history_data['match_2'] = np.apply_along_axis(lambda x: x in combination_2, 1, history_data[['number_7']])

        df = history_data.merge(pd.DataFrame(pd.Series(combination_1), columns=['combination_1']), how='cross')

        feature_list = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6', 'combination_1']
        df['match_1'] = np.apply_along_axis(lambda x: 12 - len(set(np.hstack(x))), 1, df[feature_list])

        def f_1_1(x): return 1 if x[0] == 1 and x[1] else 0

        def f_3_0(x): return 0 if x[0] != 3 else len(combination_2) - 1 if x[1] else len(combination_2)

        def f_2_1(x): return 1 if x[0] == 2 and x[1] else 0

        def f_3_1(x): return 1 if x[0] == 3 and x[1] else 0

        def f_4_0(x): return 0 if x[0] != 4 else len(combination_2) - 1 if x[1] else len(combination_2)

        def f_4_1(x): return 1 if x[0] == 4 and x[1] else 0

        def f_5_0(x): return 0 if x[0] != 5 else len(combination_2) - 1 if x[1] else len(combination_2)

        def f_5_1(x): return 1 if x[0] == 5 and x[1] else 0

        def f_6_0(x): return 0 if x[0] != 6 else len(combination_2) - 1 if x[1] else len(combination_2)

        def f_6_1(x): return 1 if x[0] == 6 and x[1] else 0

        df['match_1_1'] = np.apply_along_axis(f_1_1, 1, df[['match_1', 'match_2']])
        df['match_3_0'] = np.apply_along_axis(f_3_0, 1, df[['match_1', 'match_2']])
        df['match_2_1'] = np.apply_along_axis(f_2_1, 1, df[['match_1', 'match_2']])
        df['match_3_1'] = np.apply_along_axis(f_3_1, 1, df[['match_1', 'match_2']])
        df['match_4_0'] = np.apply_along_axis(f_4_0, 1, df[['match_1', 'match_2']])
        df['match_4_1'] = np.apply_along_axis(f_4_1, 1, df[['match_1', 'match_2']])
        df['match_5_0'] = np.apply_along_axis(f_5_0, 1, df[['match_1', 'match_2']])
        df['match_5_1'] = np.apply_along_axis(f_5_1, 1, df[['match_1', 'match_2']])
        df['match_6_0'] = np.apply_along_axis(f_6_0, 1, df[['match_1', 'match_2']])
        df['match_6_1'] = np.apply_along_axis(f_6_1, 1, df[['match_1', 'match_2']])

        label_list = ['date', 'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6', 'number_7']
        group_df = df.drop(columns=['draw_term', 'match_1', 'match_2']).groupby(label_list, as_index=False).sum()

        match_list = ['match_1_1', 'match_3_0', 'match_2_1', 'match_3_1', 'match_4_0',
                      'match_4_1', 'match_5_0', 'match_5_1', 'match_6_0', 'match_6_1']
        group_df['revenue'] = group_df[match_list].dot(pd.Series(self.lotto_prize()))

        group_df['cost'] = len(combination_1) * len(combination_2) * 100

        group_df.to_csv('group_df.csv', index=False)

        return

    def backtest(self, start_date, end_date, set_1, set_2):
        history_data = self.get_history_data(start_date, end_date)

        pool_set_1 = set(range(1, 39))
        pool_set_2 = set(range(1, 9))

        if not set_1.issubset(pool_set_1):
            print(f'Not all items are present in {pool_set_1}')
            return

        if not set_2.issubset(pool_set_2):
            print(f'Not all items are present in {pool_set_2}')
            return

        if len(set_1) < 6:
            combination_1 = [set_1.union(i) for i in itertools.combinations(pool_set_1 - set_1, 6 - len(set_1))]
        else:
            combination_1 = [i for i in itertools.combinations(set_1, 6)]

        if len(set_2) == 0:
            combination_2 = [i for i in itertools.combinations(pool_set_2, 1)]
        else:
            combination_2 = [i for i in itertools.combinations(set_2, 1)]

        self.match_prize(history_data, combination_1, combination_2)


if __name__ == '__main__':
    super_lotto638 = SuperLotto638()

    # super_lotto638.download_history('201401', '202202')

    df = super_lotto638.get_history_data('20220115', '20220228')

    # set_1 = set()
    set_1 = {24, 4, 22, 3, 19, 14}
    # set_2 = set()
    set_2 = {2}
    super_lotto638.backtest('20140101', '20220228', set_1, set_2)
