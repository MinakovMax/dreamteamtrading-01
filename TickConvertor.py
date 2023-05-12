import pandas as pd
import datetime as dt
import numpy as np


class TickConvertor:

    def __init__(self):
        pass

    # Write method to fill in the gaps time series
    @staticmethod
    def __fill_time_series(df, start, end, freq):
        # Create a new dataframe with the desired time series
        df_new = pd.DataFrame(index=pd.date_range(start, end, freq=freq))
        # Merge the new dataframe with the original dataframe
        df_new = df_new.merge(df, left_index=True, how='left', right_on='timestamp')
        # Fill in the missing values
        df_new.fillna(method='ffill', inplace=True)
        return df_new

    # Write a method that resets the volumes of ticks with the same ID
    @staticmethod
    def __reset_tick_volumes(df):
        # Create a new dataframe with the desired time series
        df_new = df.copy()
        # Replace all volume values except the first one for duplicate IDs
        df_new.loc[df_new['id'].duplicated(keep='first'), 'vol'] = 0
        # Replace all the operation values except the first one with the One of the duplicate IDs
        df_new.loc[df_new['id'].duplicated(keep='first'), 'oper'] = 'N'

        return df_new

        # Write a method that cuts lines from the dataframe before and after the specified time of day

    @staticmethod
    def __cut_lines(df, start_time, end_time):
        df = df[(df['timestamp'].dt.time >= start_time) & (df['timestamp'].dt.time <= end_time)]
        return df

    # Write method to convert seconds candles to N seconds candles. Use culumns: 'date', 'time', 'open', 'high',
    # 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'tick_count', 'buy_tick_count', 'sell_tick_count'
    @staticmethod
    def __seconds_to_N_seconds(df, N):
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y%m%d %H%M%S')
        df = df.set_index('timestamp')
        df = df.resample(str(N) + 'S').agg(
            {'wprice': 'sum', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
             'buy_volume': 'sum', 'sell_volume': 'sum', 'tick_count': 'sum', 'buy_tick_count': 'sum',
             'sell_tick_count': 'sum'})
        df = df.reset_index()
        df['date'] = df['timestamp'].dt.strftime('%Y%m%d')
        df['time'] = df['timestamp'].dt.strftime('%H%M%S')
        df['time'] = df['time'].astype(int)
        df = df[['date', 'time', 'wprice', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume',
                 'tick_count', 'buy_tick_count', 'sell_tick_count']]
        df['wprice'] = df['wprice'] / df['volume']

        df = df[df['volume'] != 0]
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def __candles_agg_N_trade_seconds(df, N):
        new_df = pd.DataFrame(columns=['date', 'time', 'timestamp', 'is_trade_session', 'change',
                                       'wprice', 'open', 'high', 'low', 'close', 'weight_price',
                                       'volume', 'buy_volume', 'sell_volume',
                                       'tick_count', 'buy_tick_count', 'sell_tick_count'])

        for i in range(0, len(df), N):

            if i + N > len(df):
                N = len(df) - i - 1

            row_value = {'date': df['date'][i], 'time': df['time'][i], 'timestamp': df['timestamp'][i],
                         'is_trade_session': df['is_trade_session'][i], 'change': df['change'][i:i + N].sum(),
                         'wprice': df['wprice'][i:i + N].sum(), 'open': df['open'][i],
                         'high': df['high'][i:i + N].max(), 'low': df['low'][i:i + N].min(),
                         'close': df['close'][i + N], 'weight_price': df['weight_price'][i],
                         'volume': df['volume'][i:i + N].sum(), 'buy_volume': df['buy_volume'][i:i + N].sum(),
                         'sell_volume': df['sell_volume'][i:i + N].sum(),
                         'tick_count': df['tick_count'][i:i + N].sum(),
                         'buy_tick_count': df['buy_tick_count'][i:i + N].sum(),
                         'sell_tick_count': df['sell_tick_count'][i:i + N].sum()}

            new_row = pd.Series(data=row_value,
                                index=['date', 'time', 'timestamp', 'is_trade_session', 'change',
                                       'wprice', 'open', 'high', 'low', 'close', 'weight_price',
                                       'volume', 'buy_volume', 'sell_volume',
                                       'tick_count', 'buy_tick_count', 'sell_tick_count'])

            new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)

        new_df['weight_price'] = new_df['wprice'] / new_df['volume']
        new_df.reset_index(drop=True, inplace=True)

        return new_df

    @staticmethod
    def __tick_to_trade_seconds(file_path, date_start, time_start, date_end, time_end):
        # Convert input dates to datetime format

        start_datetime = dt.datetime.strptime(date_start + time_start, '%Y%m%d%H%M%S')
        end_datetime = dt.datetime.strptime(date_end + time_end, '%Y%m%d%H%M%S')

        # Read the file with ticks. The file must be in the root folder
        Tiks = pd.read_csv(file_path)
        Candles = pd.DataFrame()

        # Check the input dates for compliance with the dates of the tick file
        temp_start_datetime = dt.datetime.strptime(str(Tiks['<DATE>'].loc[0]) + str(Tiks['<TIME>'].loc[0]),
                                                   '%Y%m%d%H%M%S')
        temp_end_datetime = dt.datetime.strptime(str(Tiks['<DATE>'].iloc[-1]) + str(Tiks['<TIME>'].iloc[-1]),
                                                 '%Y%m%d%H%M%S')

        if (start_datetime > temp_end_datetime) or (end_datetime < temp_start_datetime):
            return Candles

        if start_datetime < temp_start_datetime:
            start_datetime = temp_start_datetime

        if end_datetime > temp_end_datetime:
            end_datetime = temp_end_datetime

        # Read the tick-by-tick data from the text file
        df = pd.read_csv(file_path, sep=',', names=['date', 'time', 'last', 'vol', 'id', 'oper'], header=0)

        # Calculate change in price per second
        df['change'] = df['last'].diff()
        df['change'] = df['change'].fillna(0)
        df['change'] = df.loc[df['change'] != 0, ['change']] = 1

        df['wprice'] = df['last'] * df['vol']

        # Combine the date and time columns into a single timestamp column
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y%m%d %H%M%S')

        df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]

        # Group the data by 1-second intervals
        grouped = df.groupby(pd.Grouper(key='timestamp', freq='S'))

        # Aggregate the data to get the open, high, low, close, and weighted average price
        agg = grouped.agg({'change': 'sum', 'wprice': 'sum', 'last': ['first', 'max', 'min', 'last'], 'vol': 'sum'})
        agg.columns = ['change', 'wprice', 'open', 'high', 'low', 'close', 'vol']

        # Calculate the total buy volume by summing the volume for the rows where the oper was "B"
        buy_volume = df[df['oper'] == 'B'].groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].sum()

        volume = df.groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].sum()

        # Merge the total buy volume into the aggregated data
        agg = agg.merge(buy_volume, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol_y': 'buy_volume'}, inplace=True)
        agg.rename(columns={'vol_x': 'volume'}, inplace=True)

        tick_count = df.groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].count()
        agg = agg.merge(tick_count, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol': 'tick_count'}, inplace=True)

        buy_tick_count = df[df['oper'] == 'B'].groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].count()
        agg = agg.merge(buy_tick_count, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol': 'buy_tick_count'}, inplace=True)

        # Calculate the total sell volume by summing the volume for the rows where the oper was "S"
        sell_tick_count = df[df['oper'] == 'S'].groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].count()
        agg = agg.merge(sell_tick_count, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol': 'sell_tick_count'}, inplace=True)

        # Reset the index to convert the timestamps from the index back to a column
        agg = agg.reset_index()

        agg['sell_volume'] = agg['volume'] - agg['buy_volume']
        agg['weight_price'] = agg['wprice'] / agg['volume']

        # If volume is 0, then tick_count is 0
        agg.loc[agg['volume'] == 0, 'tick_count'] = 0
        agg.loc[agg['volume'] == 0, 'sell_tick_count'] = 0
        agg.loc[agg['volume'] == 0, 'buy_tick_count'] = 0

        agg['sell_tick_count'] = agg['sell_tick_count'].fillna(0).astype(int)

        # add separate , to date and time columns
        agg['date'] = agg['timestamp'].dt.strftime('%Y%m%d')
        agg['time'] = agg['timestamp'].dt.strftime('%H%M%S')

        agg['time'] = agg['time'].astype(int)

        agg['is_trade_session'] = 1

        is_premarket = agg['time'] < 100000
        is_aftermarket = np.logical_and(agg['time'] > 184500, agg['time'] < 185000)
        agg.loc[(np.logical_or(is_premarket, is_aftermarket)), 'is_trade_session'] = 0

        # Reorder the columns
        agg = agg[['date', 'time', 'timestamp',
                   'is_trade_session', 'change',
                   'wprice', 'open', 'high', 'low', 'close', 'weight_price',
                   'volume', 'buy_volume', 'sell_volume',
                   'tick_count', 'buy_tick_count', 'sell_tick_count']]

        # We bring the columns to the Int type

        agg['volume'] = agg['volume'].astype(int)
        agg['buy_volume'] = agg['buy_volume'].astype(int)
        agg['sell_volume'] = agg['sell_volume'].astype(int)

        # Delete rows with zero volume
        agg = agg[agg['volume'] != 0]

        agg.reset_index(drop=True, inplace=True)

        return agg

    @staticmethod
    def __tick_to_change_price_tick(file_path, date_start, time_start, date_end, time_end):
        # Load tick data into a pandas DataFrame
        df = pd.read_csv(file_path, sep=',', header=0)
        df.columns = ['Date', 'Time', 'Last', 'Volume', 'ID', 'OPER']

        # Filter data based on the specified start and end dates/times
        start = dt.datetime.strptime(date_start + time_start, '%Y%m%d%H%M%S')
        end = dt.datetime.strptime(date_end + time_end, '%Y%m%d%H%M%S')

        temp_start_datetime = dt.datetime.strptime(str(df['Date'].loc[0]) + str(df['Time'].loc[0]), '%Y%m%d%H%M%S')
        temp_end_datetime = dt.datetime.strptime(str(df['Date'].iloc[-1]) + str(df['Time'].iloc[-1]), '%Y%m%d%H%M%S')

        if (start > temp_end_datetime) or (end < temp_start_datetime):
            return pd.DataFrame()

        if start < temp_start_datetime:
            start = temp_start_datetime

        if end > temp_end_datetime:
            end = temp_end_datetime

        df['Date'] = df['Date'].astype(str)
        df['Time'] = df['Time'].astype(str)

        df['DATETIME'] = df['Date'] + df['Time']
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d%H%M%S')

        # Filter data based on the specified start and end dates/times
        mask = (df['DATETIME'] >= start) & (df['DATETIME'] <= end)
        df = df.loc[mask]
        df['tick_count'] = 1

        # Create a new DataFrame to store the preprocessed data
        df_processed = pd.DataFrame(
            columns=['date', 'time', 'last', 'volume', 'buy_volume', 'sell_volume', 'tick_count', 'buy_tick_count',
                     'sell_tick_count'])

        # Iterate over each row of the original DataFrame
        i = 0
        while i < len(df):
            # Find the end of the sequence of consecutive ticks with the same price
            j = i
            while j < len(df) and df.iloc[j]['Last'] == df.iloc[i]['Last']:
                j += 1

            # Compute the Buy_Volume, Sell_Volume, and Intensity values for the modified tick
            buy_volume = df.iloc[i:j]['Volume'][df.iloc[i:j]['OPER'] == 'B'].sum()
            sell_volume = df.iloc[i:j]['Volume'][df.iloc[i:j]['OPER'] == 'S'].sum()
            volume = buy_volume + sell_volume

            buy_tick_count = df.iloc[i:j]['tick_count'][df['OPER'] == 'B'].sum()
            sell_tick_count = df.iloc[i:j]['tick_count'][df['OPER'] == 'S'].sum()

            tick_count = buy_tick_count + sell_tick_count

            # Append the modified tick to the preprocessed DataFrame
            temp_df = pd.DataFrame({
                'date': [df.iloc[i]['Date']],
                'time': [df.iloc[i]['Time']],
                'last': [df.iloc[i]['Last']],
                'volume': [volume],
                'buy_volume': [buy_volume],
                'sell_volume': [sell_volume],
                'tick_count': [tick_count],
                'buy_tick_count': [buy_tick_count],
                'sell_tick_count': [sell_tick_count]
            })
            df_processed = pd.concat([df_processed, temp_df], ignore_index=True)

            i = j

        return df_processed

    @staticmethod
    def __tick_to_seconds(file_path, date_start, time_start, date_end, time_end, start_session_time=dt.time(9, 55, 00),
                          end_session_time=dt.time(23, 50, 00)):

        # Load tick data into a pandas DataFrame
        start_datetime = dt.datetime.strptime(date_start + time_start, '%Y%m%d%H%M%S')
        end_datetime = dt.datetime.strptime(date_end + time_end, '%Y%m%d%H%M%S')

        tiks = pd.read_csv(file_path)
        candles = pd.DataFrame()

        # Check entry dates
        temp_start_datetime = dt.datetime.strptime(str(tiks['<DATE>'].loc[0]) + str(tiks['<TIME>'].loc[0]),
                                                   '%Y%m%d%H%M%S')
        temp_end_datetime = dt.datetime.strptime(str(tiks['<DATE>'].iloc[-1]) + str(tiks['<TIME>'].iloc[-1]),
                                                 '%Y%m%d%H%M%S')

        if (start_datetime > temp_end_datetime) or (end_datetime < temp_start_datetime):
            return candles

        if start_datetime < temp_start_datetime:
            start_datetime = temp_start_datetime

        if end_datetime > temp_end_datetime:
            end_datetime = temp_end_datetime

        # Read the tick-by-tick data from the text file
        df = pd.read_csv(file_path, sep=',', names=['date', 'time', 'last', 'vol', 'id', 'oper'], header=0)

        # Combine the date and time columns into a single timestamp column
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y%m%d %H%M%S')

        df = TickConvertor.__fill_time_series(df, start_datetime, end_datetime, 'S')
        df = TickConvertor.__reset_tick_volumes(df)

        df['wprice'] = df['last'] * df['vol']

        # Convert the date and time columns to a single datetime column
        df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]

        # Group the data by 1-second intervals
        grouped = df.groupby(pd.Grouper(key='timestamp', freq='S'))

        # Aggregate the data to get the open, high, low, close, and weighted average price
        agg = grouped.agg({'wprice': 'sum', 'last': ['first', 'max', 'min', 'last'], 'vol': 'sum'})
        agg.columns = ['wprice', 'open', 'high', 'low', 'close', 'vol']

        # Calculate the total buy volume by summing the volume for the rows where the oper was "B"
        buy_volume = df[df['oper'] == 'B'].groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].sum()

        # Merge the total buy volume into the aggregated data
        agg = agg.merge(buy_volume, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol_y': 'buy_volume'}, inplace=True)
        agg.rename(columns={'vol_x': 'volume'}, inplace=True)

        tick_count = df.groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].count()
        agg = agg.merge(tick_count, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol': 'tick_count'}, inplace=True)

        buy_tick_count = df[df['oper'] == 'B'].groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].count()
        agg = agg.merge(buy_tick_count, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol': 'buy_tick_count'}, inplace=True)

        # Calculate the total sell volume by summing the volume for the rows where the oper was "S"
        sell_tick_count = df[df['oper'] == 'S'].groupby(pd.Grouper(key='timestamp', freq='S'))['vol'].count()
        agg = agg.merge(sell_tick_count, left_on='timestamp', right_index=True, how='left')
        agg.rename(columns={'vol': 'sell_tick_count'}, inplace=True)

        # Reset the index to convert the timestamps from the index back to a column
        agg = agg.reset_index()

        agg['sell_volume'] = agg['volume'] - agg['buy_volume']
        agg['weight_price'] = agg['wprice'] / agg['volume']

        # If volume is 0, then tick_count is 0
        agg.loc[agg['volume'] == 0, 'tick_count'] = 0
        agg.loc[agg['volume'] == 0, 'sell_tick_count'] = 0
        agg.loc[agg['volume'] == 0, 'buy_tick_count'] = 0

        agg['sell_tick_count'] = agg['sell_tick_count'].fillna(0).astype(int)

        # add separate , to date and time columns
        agg['date'] = agg['timestamp'].dt.strftime('%Y%m%d')
        agg['time'] = agg['timestamp'].dt.strftime('%H%M%S')

        agg['time'] = agg['time'].astype(int)

        agg['is_trade_session'] = 1

        is_premarket = agg['time'] < 100000
        is_aftermarket = np.logical_and(agg['time'] > 184500, agg['time'] < 185000)
        agg.loc[(np.logical_or(is_premarket, is_aftermarket)), 'is_trade_session'] = 0

        # Reorder the columns
        agg = agg[['date', 'time', 'timestamp', 'is_trade_session',
                   'wprice', 'open', 'high', 'low', 'close', 'weight_price',
                   'volume', 'buy_volume', 'sell_volume', 'tick_count', 'buy_tick_count', 'sell_tick_count']]

        # We bring the columns to the Int type

        agg['volume'] = agg['volume'].astype(int)
        agg['buy_volume'] = agg['buy_volume'].astype(int)
        agg['sell_volume'] = agg['sell_volume'].astype(int)

        agg = TickConvertor.__cut_lines(agg, start_session_time, end_session_time)

        return agg

    @staticmethod
    def tick_to_candles_N_seconds(file_path, date_start, time_start, date_end, time_end, N=1,
                                  start_session_time=dt.time(9, 55, 00), end_session_time=dt.time(23, 50, 00)):
        """
        Функция преобразования тиковых данных в N-секундные свечи
        """
        if N == 1:
            return TickConvertor.__tick_to_seconds(file_path, date_start, time_start, date_end, time_end,
                                                   start_session_time, end_session_time)
        elif N > 1:
            candles = TickConvertor.__tick_to_seconds(file_path, date_start, time_start, date_end, time_end,
                                                      start_session_time, end_session_time)
            candles = TickConvertor.__seconds_to_N_seconds(candles, N)
            return candles
        else:
            raise ValueError('N must be greater than 0')

    @staticmethod
    def tick_to_candles_N_trade_second(file_path, date_start, time_start, date_end, time_end, N=1):
        """
        Функция преобразования тиковых данных в N-секундные свечи по торговым секунд
        """
        if N == 1:
            return TickConvertor.__tick_to_trade_seconds(file_path, date_start, time_start, date_end, time_end)
        elif N > 1:
            candles = TickConvertor.__tick_to_trade_seconds(file_path, date_start, time_start, date_end, time_end)
            candles = TickConvertor.__candles_agg_N_trade_seconds(candles, N)
            candles['wprice'] = candles['weight_price'].copy()
            candles.drop(columns=['weight_price'], inplace=True)
            return candles
        else:
            raise ValueError('N must be greater than 0')

    @staticmethod
    def __changed_tick_to_N_row_candles(df, N):

        new_df = pd.DataFrame(columns=['date', 'time',
                                       'wprice', 'open', 'high', 'low', 'close',
                                       'volume', 'buy_volume', 'sell_volume',
                                       'tick_count', 'buy_tick_count', 'sell_tick_count'])

        if N > len(df):
            N = len(df)

        for i in range(0, len(df), N):
            if i + N > len(df):
                N = len(df) - i - 1

            temp_df = df.iloc[i:i + N]

            temp_wprice = temp_df['last'] * temp_df['volume']
            wprice = temp_wprice.sum() / temp_df['volume'].sum()

            new_row_value = {
                'date': temp_df.iloc[0]['date'],
                'time': temp_df.iloc[0]['time'],
                'wprice': wprice,
                'open': temp_df.iloc[0]['last'],
                'high': temp_df['last'].max(),
                'low': temp_df['last'].min(),
                'close': temp_df.iloc[-1]['last'],
                'volume': temp_df['volume'].sum(),
                'buy_volume': temp_df['buy_volume'].sum(),
                'sell_volume': temp_df['sell_volume'].sum(),
                'tick_count': temp_df['tick_count'].sum(),
                'buy_tick_count': temp_df['buy_tick_count'].sum(),
                'sell_tick_count': temp_df['sell_tick_count'].sum()
            }

            new_row = pd.Series(data=new_row_value, index=['date', 'time',
                                                           'wprice', 'open', 'high', 'low', 'close',
                                                           'volume', 'buy_volume', 'sell_volume',
                                                           'tick_count', 'buy_tick_count', 'sell_tick_count'])

            new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)

        return new_df

    @staticmethod
    def tick_to_N_change_tick_candles(file_path, date_start, time_start, date_end, time_end, N=1):

        """
        Функция преобразования тиковых данных в N-секундные свечи
        """
        if N > 0:
            candles = TickConvertor.__tick_to_change_price_tick(file_path, date_start, time_start, date_end, time_end)

            candles = TickConvertor.__changed_tick_to_N_row_candles(candles, N)
            return candles
        else:
            raise ValueError('N must be greater than 0')

    # Write method that converts tick data to N tick candles
    @classmethod
    def tick_to_candles_N_tick(cls, file_path, date_start, time_start, date_end, time_end, N):

        start_datetime = dt.datetime.strptime(date_start + time_start, '%Y%m%d%H%M%S')
        end_datetime = dt.datetime.strptime(date_end + time_end, '%Y%m%d%H%M%S')

        # Read the file with ticks. The file must be in the root folder
        tiks = pd.read_csv(file_path)
        candles = pd.DataFrame()

        # Check the input dates for compliance with the dates of the tick file
        temp_start_datetime = dt.datetime.strptime(str(tiks['<DATE>'].loc[0]) + str(tiks['<TIME>'].loc[0]),
                                                   '%Y%m%d%H%M%S')
        temp_end_datetime = dt.datetime.strptime(str(tiks['<DATE>'].iloc[-1]) + str(tiks['<TIME>'].iloc[-1]),
                                                 '%Y%m%d%H%M%S')

        if (start_datetime > temp_end_datetime) or (end_datetime < temp_start_datetime):
            return candles

        if start_datetime < temp_start_datetime:
            start_datetime = temp_start_datetime

        if end_datetime > temp_end_datetime:
            end_datetime = temp_end_datetime

        # Read the tick-by-tick data from the text file
        df = pd.read_csv(file_path, sep=',', names=['date', 'time', 'last', 'vol', 'id', 'oper'], header=0)

        # Combine the date and time columns into a single timestamp column
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y%m%d %H%M%S')

        df = TickConvertor.__fill_time_series(df, start_datetime, end_datetime, 'S')
        df = TickConvertor.__reset_tick_volumes(df)

        df['wprice'] = df['last'] * df['vol']

        # Convert the date and time columns to a single datetime column
        df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]

        df['tick_count'] = 1

        # Create candles
        candles['date'] = df['date']
        candles['time'] = df['time']
        candles['timestamp'] = df['timestamp']
        candles['wprice'] = df['last'] * df['vol']
        candles['open'] = df['last']
        candles['high'] = df['last']
        candles['low'] = df['last']
        candles['close'] = df['last']
        candles['volume'] = df['vol']
        candles['buy_volume'] = df['vol'].where(df['oper'] == 'B')
        candles['sell_volume'] = df['vol'].where(df['oper'] == 'S')
        candles['buy_tick_count'] = df['tick_count'].where(df['oper'] == 'B')
        candles['sell_tick_count'] = df['tick_count'].where(df['oper'] == 'S')
        candles['tick_count'] = df['tick_count']

        # Group candles by N tick
        candles = candles.groupby(candles.index // N).agg({'date': 'first', 'time': 'first', 'timestamp': 'first',
                                                           'wprice': 'sum', 'open': 'first', 'high': 'max',
                                                           'low': 'min', 'close': 'last',
                                                           'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum',
                                                           'buy_tick_count': 'sum', 'sell_tick_count': 'sum',
                                                           'tick_count': 'sum'})

        # Fill NaN values
        candles['buy_volume'].fillna(0, inplace=True)
        candles['sell_volume'].fillna(0, inplace=True)
        candles['buy_tick_count'].fillna(0, inplace=True)
        candles['sell_tick_count'].fillna(0, inplace=True)

        # Convert columns to integer
        candles['buy_volume'] = candles['buy_volume'].astype(int)
        candles['sell_volume'] = candles['sell_volume'].astype(int)
        candles['buy_tick_count'] = candles['buy_tick_count'].astype(int)
        candles['sell_tick_count'] = candles['sell_tick_count'].astype(int)

        # Calculate weighted average price
        candles['wprice'] = candles['wprice'] / candles['volume']

        candles['is_trade_session'] = 1

        is_premarket = candles['time'] < 100000
        is_aftermarket = np.logical_and(candles['time'] > 184500, candles['time'] < 185000)
        candles.loc[(np.logical_or(is_premarket, is_aftermarket)), 'is_trade_session'] = 0

        # Reset index
        # candles.reset_index(drop=True, inplace=True)

        candles = candles[['date', 'time', 'timestamp', 'is_trade_session',
                           'wprice', 'open', 'high', 'low', 'close',
                           'volume', 'buy_volume', 'sell_volume', 'tick_count', 'buy_tick_count', 'sell_tick_count']]

        candles['date'] = candles['date'].astype(int)
        candles['time'] = candles['time'].astype(int)

        return candles
