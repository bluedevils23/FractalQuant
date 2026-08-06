from .fz_base import Factor
import os
import polars as pl
from typing import Optional, List
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

class MinFreqFactor(Factor):
    def __init__(self, factor_name, factor_exposure=None):
        """
        由分钟行情构造的日频因子类：从因子类中继承coverage/ic_test/group_test
        :param factor_name: 因子的名字
        :param factor_exposure: 因子暴露
        """
        super().__init__(factor_name, factor_exposure)

    @staticmethod
    def _normalize_daily_result(result):
        """Collect and validate one-factor daily exposure output."""
        if result is None:
            return None
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        if not isinstance(result, pl.DataFrame):
            raise TypeError(
                f'因子计算结果必须是DataFrame、LazyFrame或None，实际为{type(result).__name__}'
            )
        missing_keys = {'code', 'date'} - set(result.columns)
        if missing_keys:
            raise ValueError(f'因子计算结果缺少键列: {sorted(missing_keys)}')
        if result.is_empty():
            return result
        duplicate_keys = (
            result.group_by(['code', 'date'])
            .len()
            .filter(pl.col('len') > 1)
        )
        if duplicate_keys.height:
            raise ValueError('因子计算结果包含重复的code/date键')
        value_columns = [
            column for column in result.columns
            if column not in {'code', 'date'}
        ]
        if len(value_columns) != 1:
            raise ValueError(
                f'因子计算结果必须且只能包含一个值列，实际为{value_columns}'
            )
        return result.sort(['date', 'code'])

    @staticmethod
    def _process_single_file(file_name, folder_path, calculate_method, pv_data):
        """处理单个文件"""
        try:
            file_path = os.path.join(folder_path, file_name)
            if pv_data is None:  # 不需要日频量价数据
                result = calculate_method(pl.read_parquet(file_path))
            else:  # 需要日频量价数据
                result = calculate_method(pl.read_parquet(file_path), pv_data)
            return MinFreqFactor._normalize_daily_result(result)
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            return None

    @staticmethod
    def _read_exposure(
            factor_name: str, path: Optional[str|None], default_path: str
    ) -> Optional[pl.DataFrame|None]:
        """
        读取因子暴露数据
        :param factor_name: 因子名
        :param path: 因子保存的路径，可以为文件或文件夹
        :param default_path: 默认路径，在path为空时使用
        :return:
        """
        factor_exposure = None
        if path is None:
            path = default_path
        if path.endswith('.parquet'):  # 传递的path参数为因子暴露所在的位置
            factor_exposure = pl.read_parquet(path)
        else:  # 传递的path参数为文件夹
            exposures = os.listdir(path)
            if f'{factor_name}.parquet' in exposures:  # 存在已计算的因子暴露
                path = os.path.join(path, f'{factor_name}.parquet')
                factor_exposure = pl.read_parquet(path)
        return factor_exposure

    def cal_daily_exposure_by_min_data(
            self,
            calculate_method,
            path: str = None,
            need_daily_data: bool = False,
            n_jobs: int = None
    ):
        r"""
        使用分钟行情计算日频因子暴露。如果已有结果则更新至最新交易日。
        :param calculate_method: 因子计算方法
        :param path: 日频因子暴露的保存路径
        :param need_daily_data: 是否需要日频量价数据，默认为False
        :param n_jobs:
        """
        factor_exposure = self._read_exposure(
            factor_name=self.factor_name,
            default_path=r'D:\QuantData\DailyFreqFactor\FangZheng Factor',
            path=path
        )

        folder_path = r'D:\QuantData\KLine_cleaned'  # 分钟频价量数据
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        pv_data_index = pl.DataFrame(
            data={'file_name': file_names},
            schema={'file_name': pl.String},
        ).with_columns(
            pl.col('file_name')
            .str.head(8)
            .str.to_date(format='%Y%m%d')
            .alias('date')
        )
        if factor_exposure is not None:  # 如果有已计算的因子暴露
            end_date = factor_exposure['date'].max()
            pv_data_index = pv_data_index.filter(pl.col('date') > end_date)

        valid_results = []
        if len(pv_data_index) > 0:
            if need_daily_data & (n_jobs is None):  # 如果需要日频量价数据
                daily_pv = pl.read_parquet(r'D:\QuantData\Price_volume.parquet')
                n_jobs = 1
            elif need_daily_data & (n_jobs is not None):
                daily_pv = pl.read_parquet(r'D:\QuantData\Price_volume.parquet')
                n_jobs = n_jobs
            elif (~need_daily_data) & (n_jobs is None):
                daily_pv = None
                n_jobs = -1
            elif (~need_daily_data) & (n_jobs is not None):
                daily_pv = None
                n_jobs = n_jobs
            else:
                raise ValueError(
                    f'不正确的need_daily_data或n_jobs参数值, {need_daily_data}与{n_jobs}｝'
                )
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._process_single_file)(
                    file_name,
                    folder_path,
                    calculate_method,
                    daily_pv
                )
                for file_name in tqdm(pv_data_index['file_name'], desc='Processing')
            )
            valid_results = [r for r in results if r is not None]

        if factor_exposure is None and valid_results:
            self.factor_exposure = (
                pl.concat(valid_results, how='vertical')
                .unique(subset=['code', 'date'], keep='last')
                .sort(['date', 'code'])
            )
        elif factor_exposure is None:
            self.factor_exposure = None
        elif len(valid_results) > 0:
            update_exposure = pl.concat(valid_results, how='vertical')
            self.factor_exposure = (
                pl.concat(
                    items=[factor_exposure, update_exposure],
                    how='vertical'
                )
                .unique(subset=['code', 'date'], keep='last')
                .sort(['date', 'code'])
            )
        else:
            self.factor_exposure = factor_exposure

    def cal_exposure_by_min_data(
            self,
            calculate_method,
            path: str = None,
            need_daily_data: bool = False,
            n_jobs: int = None
    ):
        """兼容旧API；结果仍为每个code/date一行的日频暴露。"""
        warnings.warn(
            'cal_exposure_by_min_data已更名为cal_daily_exposure_by_min_data；'
            '返回结果是日频暴露。',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cal_daily_exposure_by_min_data(
            calculate_method=calculate_method,
            path=path,
            need_daily_data=need_daily_data,
            n_jobs=n_jobs,
        )

    def cal_exposure_by_intermediate(
            self,
            calculate_method,
            intermediate: Optional[str|List[str]],
            path: str=None,
            intermediate_path: str=None,
            depend_past_days: int=20,
            need_daily_pv_data: bool=False,
            depend_columns: List[str]=None
    ):
        """
        使用中间因子计算因子暴露。
        :param calculate_method: 计算方法
        :param intermediate: 中间变量。可以传递为"PanDeng"或["ZhaoMoChenWu", "WuBiGuMu"]
        :param path: 日频因子暴露的保存路径
        :param intermediate_path: 日频中间因子的保存路径
        :param depend_past_days: 依赖过去多少天的值，默认为20
        :param need_daily_pv_data: 是否需要日频量价数据
        :param depend_columns: 依赖的列，在需要日频量价数据时可以传递
        """
        factor_exposure = self._read_exposure(
            factor_name=self.factor_name,
            default_path=r'D:\QuantData\DailyFreqFactor\FangZheng Factor',
            path=path
        )

        minute_freq_factors = [
            file_name.removesuffix('.parquet')
            for file_name in os.listdir(r'D:\QuantData\MinuteFreqFactor')
        ]  # 分钟频因子集
        daily_freq_factors = [
            file_name.removesuffix('.parquet')
            for file_name in os.listdir(r'D:\QuantData\DailyFreqFactor')
        ]  # 日频因子集
        if intermediate_path is None:
            special_factors = []
        else:
            special_factors = [
                file_name.removesuffix('.parquet')
                for file_name in os.listdir(intermediate_path)
            ]  # 指定因子集

        def _get_intermediate_path(intermediate_name: str, folder_path: str=None) -> str:
            """获得中间变量的路径"""
            if intermediate_name in minute_freq_factors:
                if folder_path is None:
                    folder_path = (
                        r'D:\QuantData\MinuteFreqFactor'
                    )
                return os.path.join(
                    folder_path,
                    f'{intermediate_name}.parquet'
                )
            elif intermediate_name in daily_freq_factors:
                if folder_path is None:
                    folder_path = (
                        r'D:\QuantData\DailyFreqFactor'
                    )
                return os.path.join(
                    folder_path,
                    f'{intermediate_name}.parquet'
                )
            elif intermediate_name in special_factors:
                if folder_path is None:
                    folder_path = intermediate_path
                return os.path.join(
                    folder_path,
                    f'{intermediate_name}.parquet'
                )
            else:
                raise ValueError(f'中间变量{intermediate_name}未计算')

        if isinstance(intermediate, str):  # 中间变量仅为一个因子
            intermediate_df = pl.read_parquet(_get_intermediate_path(
                intermediate, intermediate_path
            ))
        elif isinstance(intermediate, list):  # 中间变量有多个因子
            intermediate_df = []
            for single_intermediate in intermediate:
                intermediate_df.append(
                    pl.read_parquet(_get_intermediate_path(
                        single_intermediate, intermediate_path
                    ))
                )
            intermediate_df = pl.concat(intermediate_df, how='align')
        else:
            raise ValueError(f'中间变量{intermediate}格式不正确')

        if need_daily_pv_data:  # 如果需要日频量价数据
            intermediate_df = pl.concat(
                [
                    intermediate_df,
                    self._read_daily_pv_data(depend_columns)
                ], how='align'
            )

        if factor_exposure is None:  # 不存在已计算的因子暴露
            factor_exposure = self._normalize_daily_result(
                calculate_method(intermediate_df)
            )
        else:  # 存在已计算的因子暴露
            if depend_past_days is None:
                latest_date = factor_exposure['date'].max()
            else:
                latest_date = factor_exposure.select(
                    pl.col('date')
                    .unique()
                    .top_k(depend_past_days + 1)
                )['date'].min()
            intermediate_df = intermediate_df.filter(
                pl.col('date') > latest_date
            )
            newly_factor_exposure = self._normalize_daily_result(
                calculate_method(intermediate_df)
            )
            if newly_factor_exposure is not None:
                factor_exposure = (
                    pl.concat([factor_exposure, newly_factor_exposure])
                    .unique(subset=['code', 'date'], keep='last')
                    .sort(['date', 'code'])
                )

        self.factor_exposure = factor_exposure
