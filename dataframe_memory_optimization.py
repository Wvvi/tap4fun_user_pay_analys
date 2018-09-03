#!/usr/bin/python3
import pandas as pd
from pandas import DataFrame, Series
import pprint
import pickle
import gc


class DtypesConvert:

    def __init__(self, data):
        self.data = data
        self.memory_diff = None
        self.dtype_diff = None
        self.column_types = None

    def mem_usage(self, pandas_obj):
        if isinstance(pandas_obj, pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else:  # we assume if not a df it's a series
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = round(usage_b/1024**2, 3)
        return usage_mb

    def check_object(self, object_df):
        converted_obj = DataFrame()
        for col in object_df.columns:
            num_unique_values = len(object_df[col].unique())
            num_total_values = len(object_df[col])
            if num_unique_values/num_total_values < 0.5:
                converted_obj.loc[:, col] = object_df.astype('category')
            else:
                converted_obj.loc[:, col] = object_df[col]
        return converted_obj

    def dtype_memory(self):
        optimized_df = self.data
        origin_memory_all = 0
        converted_memory_all = 0
        decline_ratio_all = 0
        memory_diff = DataFrame(columns=['before', 'after'])
        dtype_diff = []
        for dtype in ['int', 'float', 'object']:
            selected_dtype = self.data.select_dtypes(include=[dtype])
            if not selected_dtype.empty:
                mean_usage_mb = self.mem_usage(selected_dtype)
                print('Average memory usage for {} columns:{:03.2f}MB'.format(
                    dtype, mean_usage_mb))
                if dtype == 'int':
                    converted_df = selected_dtype.apply(
                        pd.to_numeric, downcast='unsigned')
                elif dtype == 'float':
                    converted_df = selected_dtype.apply(
                        pd.to_numeric, downcast='float')
                elif dtype == 'object':
                    converted_df = self.check_object(selected_dtype)
                optimized_df[converted_df.columns] = converted_df
                origin_memory = self.mem_usage(selected_dtype)
                converted_memory = self.mem_usage(converted_df)
                memory_diff.loc[dtype, 'before'] = origin_memory
                memory_diff.loc[dtype, 'after'] = converted_memory
                memory_diff.loc[dtype, 'decline_ratio(%)'] = round(
                    (origin_memory-converted_memory)/origin_memory*100, 2)
                decline_ratio = (
                    origin_memory - converted_memory)/origin_memory*100
                origin_memory_all += origin_memory
                converted_memory_all += converted_memory
                decline_ratio_all += decline_ratio
                print('--------------------------------------------------')
                print('Origin memory usage for {} columns:{:03.2f}'.format(
                    dtype, origin_memory))
                print('Converted memory usage for {} columns: {:03.2f}'.format(
                    dtype, converted_memory))
                print('Decline ratio:{:03.2f}%'.format(decline_ratio))
                compare_df = pd.concat(
                    [selected_dtype.dtypes, converted_df.dtypes], axis=1)
                compare_df.columns = ['before', 'after']
                compare_df = compare_df.apply(Series.value_counts)
                dtype_diff.append(compare_df)
                print(compare_df)
        dtype_diff = pd.concat(dtype_diff)
        self.memory_diff = memory_diff
        self.dtype_diff = dtype_diff
        print('---------------------------------------------------')
        print('Total origin memoty usage:{:03.2f}'.format(origin_memory_all))
        print('Total convert memotyusage:{:03.2f}'.format(
            converted_memory_all))
        print('Total decline ratio:{:03.2f}%'.format(decline_ratio_all))
        print('-------------------------------------------------------')
        print(memory_diff)
        print(dtype_diff)
        return optimized_df

    def dump_and_load_file(self, file_path, columns_types=None):
        if columns_types:
            with open(file_path, 'wb') as file:
                columns_types = pickle.dump(columns_types, file)
        else:
            with open(file_path, 'rb') as file:
                columns_types = pickle.load(file)
                return columns_types

    def dataframe_dtype_converted(self, optimized_df, file_path):
        dtypes = optimized_df.dtypes
        dtypes_col = dtypes.index
        dtypes_type = [i.name for i in dtypes.values]
        column_types = dict(zip(dtypes_col, dtypes_type))
        self.column_types = column_types
        self.dump_and_load_file('column_types.pkl', column_types)
        preview = {key: value for key, value in list(
            column_types.items())[:10]}
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(preview)
        read_and_optimized = pd.read_csv(file_path, dtype=column_types)
        print(self.mem_usage(read_and_optimized))
        print(read_and_optimized.info())
        column_types_load = self.dump_and_load_file('column_types.pkl')
        print(len(column_types_load))
        return read_and_optimized


if __name__ == '__main__':
    file_path = 'tap4fun_compitition_data/tap_fun_train.csv'
    reader = pd.read_csv(file_path, chunksize=10000)
    data = reader.get_chunk()
    C = DtypesConvert(data)
    optimized_df = C.dtype_memory()
    optimized_data = C.dataframe_dtype_converted(optimized_df, file_path)
    #C.dataframe_dtype_converted(optimized_df, file_path)
