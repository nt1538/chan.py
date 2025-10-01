import os
from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit
from DataAPI.CommonStockAPI import CCommonStockApi


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else str2float(data[i])
    return dict(zip(column_name, data))


def parse_time_column(inp):
    """
    Parse various time formats:
    - "8/11/2019 18:05" (M/D/YYYY H:MM)
    - "2021-09-13" (YYYY-MM-DD)
    - "20210902113000000" (YYYYMMDDHHMMSS000)
    - "2021-09-13 18:05:30" (YYYY-MM-DD HH:MM:SS)
    """
    inp = inp.strip()
    
    # Handle M/D/YYYY H:MM format (your CSV format)
    if '/' in inp and ' ' in inp:
        try:
            date_part, time_part = inp.split(' ')
            month, day, year = date_part.split('/')
            hour, minute = time_part.split(':')
            return CTime(int(year), int(month), int(day), int(hour), int(minute))
        except ValueError:
            pass
    
    # Handle M/D/YYYY format (date only)
    elif '/' in inp and ' ' not in inp:
        try:
            month, day, year = inp.split('/')
            return CTime(int(year), int(month), int(day), 0, 0)
        except ValueError:
            pass
    
    # Original formats
    elif len(inp) == 10 and '-' in inp:
        # 2021-09-13
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = minute = 0
    elif len(inp) == 17:
        # 20210902113000000
        year = int(inp[:4])
        month = int(inp[4:6])
        day = int(inp[6:8])
        hour = int(inp[8:10])
        minute = int(inp[10:12])
    elif len(inp) == 19 and '-' in inp:
        # 2021-09-13 18:05:30
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = int(inp[11:13])
        minute = int(inp[14:16])
    else:
        raise Exception(f"unknown time column format from csv: '{inp}'. Supported formats: M/D/YYYY H:MM, YYYY-MM-DD, YYYYMMDDHHMMSS000, YYYY-MM-DD HH:MM:SS")
    
    return CTime(year, month, day, hour, minute)


def time_to_date_string(ctime):
    """Convert CTime to YYYY-MM-DD format for comparison"""
    return f"{ctime.year:04d}-{ctime.month:02d}-{ctime.day:02d}"


class CSV_API(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        self.headers_exist = True  # 第一行是否是标题，如果是数据，设置为False
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,  # Added volume support
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]  # 每一列字段
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        super(CSV_API, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        k_type = self.k_type.name[2:].lower()
        file_path = f"{cur_path}/data/{self.code}_{k_type}.csv"
        if not os.path.exists(file_path):
            raise CChanException(f"file not exist: {file_path}", ErrCode.SRC_DATA_NOT_FOUND)

        total_lines = 0
        filtered_lines = 0
        
        for line_number, line in enumerate(open(file_path, 'r')):
            if self.headers_exist and line_number == 0:
                continue
                
            total_lines += 1
            data = line.strip("\n").split(",")
            if len(data) < len(self.columns):
                # Handle missing volume column
                if len(data) == len(self.columns) - 1:
                    data.append("0")  # Add default volume of 0
                else:
                    raise CChanException(f"file format error: {file_path}", ErrCode.SRC_DATA_FORMAT_ERROR)
            
            # Parse the time for proper date comparison
            time_str = data[self.time_column_idx]
            try:
                parsed_time = parse_time_column(time_str)
                date_str = time_to_date_string(parsed_time)
                
                # Proper date filtering using parsed dates
                if self.begin_date is not None and date_str < self.begin_date:
                    continue
                if self.end_date is not None and date_str > self.end_date:
                    continue
                    
                filtered_lines += 1
                yield CKLine_Unit(create_item_dict(data, self.columns))
                
            except Exception as e:
                print(f"Warning: Failed to parse time '{time_str}' at line {line_number}: {e}")
                continue
        
        print(f"CSV_API: Processed {total_lines} lines, filtered to {filtered_lines} lines")
        if filtered_lines == 0:
            print(f"Warning: No data in date range {self.begin_date} to {self.end_date}")

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass