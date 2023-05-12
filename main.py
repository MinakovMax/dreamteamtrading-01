import TickConvertor as tc

date_start = '20230122'
time_start = '100000'
date_end = '20230123'
time_end = '105000'
file_path = r'GAZP_230123_230130.csv'

# Main function
def main():
    # Create convertor tick to candles
    Convertor = tc.TickConvertor()

    # Use methods that convert tick to candles N seconds
    first = Convertor.tick_to_candles_N_seconds(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end,time_end=time_end, N=1)
    first.to_csv('first.csv', index=False)

    first = Convertor.tick_to_candles_N_seconds(file_path=file_path, date_start=date_start, time_start=time_start,date_end=date_end, time_end=time_end, N=20)
    first.to_csv('first_20s.csv', index=False)

    # Use methods that convert tick to candles N trade second
    second = Convertor.tick_to_candles_N_trade_second(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end,time_end=time_end, N=1)
    second.to_csv('second.csv', index=False)

    second = Convertor.tick_to_candles_N_trade_second(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end, time_end=time_end, N=20)
    second.to_csv('second_20s.csv', index=False)

    # Use methods that convert tick to candles N change price tick
    third = Convertor.tick_to_N_change_tick_candles(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end,time_end=time_end, N=1)
    third.to_csv('third.csv', index=False)

    third = Convertor.tick_to_N_change_tick_candles(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end, time_end=time_end, N=20)
    third.to_csv('third_20s.csv', index=False)

    # Use methods that convert tick to candles N tick candles
    fourth = Convertor.tick_to_candles_N_tick(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end,time_end=time_end, N=1)
    fourth.to_csv('fourth.csv', index=False)

    fourth = Convertor.tick_to_candles_N_tick(file_path=file_path, date_start=date_start, time_start=time_start, date_end=date_end, time_end=time_end, N=20)
    fourth.to_csv('fourth_20s.csv', index=False)


main()