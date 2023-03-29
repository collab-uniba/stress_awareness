from datetime import datetime, timedelta


def calculate_date_time(timestamp_0, hz, num_rows):
    format = "%d/%m/%Y, %H:%M:%S"

    date_time_0 = datetime.fromtimestamp(timestamp_0)

    #Change datatime format
    date_time_0_str = date_time_0.strftime(format)
    date_time_0 = datetime.strptime(date_time_0_str, format)

    data_times = [date_time_0]
    off_set = 1 / hz
    for i in range(1, num_rows):
        data_time_temp = data_times[i-1] + timedelta(seconds = off_set)
        data_times.append(data_time_temp)

    date = str(data_times[0].date())
    times = [t.time() for t in data_times]
    return date, times


def fix_interval_hr(data):
    #TODO Ha senso questo? 
    # https://www.frontiersin.org/articles/10.3389/fnbeh.2022.856544/full
    for i in range(data.size):
        if data[i] < 40:
            data[i] = 40
        elif data[i] > 180:
            data[i] = 180

    return data


def is_ratio_rr_interval_correct(interval):
    min_rr = min(interval)
    max_rr = max(interval)
    rate = max_rr / min_rr
    
    correct = None
    if rate <= 1.1:
        correct = True
    else:
        correct = False
    
    return correct


def check_ratio_rr_intervals(data):
    window_size = 10 #Seconds
    for i in range(data.size - window_size + 1):
        window_data = data[i:i+window_size]
        correct = is_ratio_rr_interval_correct(window_data)

        if correct == False:
            print(i)
    

