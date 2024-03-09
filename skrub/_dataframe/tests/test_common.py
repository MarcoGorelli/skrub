"""
Note: most tests in this file use the ``df_module`` fixture, which is defined
in ``skrub.conftest``. See the corresponding docstrings for details.
"""

import inspect
from datetime import datetime

import narwhals as nws
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skrub._dataframe import _common as ns


def test_not_implemented():
    # make codecov happy
    has_default_impl = {
        "is_dataframe",
        "is_column",
        "collect",
        "is_lazyframe",
        "pandas_convert_dtypes",
        "to_column_list",
    }
    for func_name in sorted(set(ns.__all__) - has_default_impl):
        func = getattr(ns, func_name)
        n_params = len(inspect.signature(func).parameters)
        params = [None] * n_params
        with pytest.raises(NotImplementedError):
            func(*params)


#
# Inspecting containers' type and module
# ======================================
#


def test_get_implementation(df_module):
    assert nws.get_implementation(df_module.empty_dataframe) == df_module.name


def test_dataframe_module_name(df_module):
    assert nws.get_implementation(df_module.empty_dataframe) == df_module.name
    assert getattr(nws, f"is_{df_module.name}")(df_module.empty_dataframe)
    assert nws.get_implementation(df_module.empty_column) == df_module.name
    assert getattr(nws, f"is_{df_module.name}")(df_module.empty_column)


def test_is_dataframe(df_module):
    assert nws.is_dataframe(df_module.empty_dataframe)
    assert not nws.is_dataframe(df_module.empty_column)
    assert not nws.is_dataframe(np.eye(3))
    assert not nws.is_dataframe({"a": [1, 2]})


def test_is_lazyframe(df_module):
    assert not ns.is_lazyframe(df_module.empty_dataframe)
    if hasattr(df_module, "empty_lazyframe"):
        assert ns.is_lazyframe(df_module.empty_lazyframe)


def test_is_column(df_module):
    assert ns.is_column(df_module.empty_column)
    assert not ns.is_column(df_module.empty_dataframe)
    assert not ns.is_column(np.eye(3))
    assert not ns.is_column({"a": [1, 2]})


#
# Conversions to and from other container types
# =============================================
#


def test_to_numpy(df_module, example_data_dict):
    array = (
        nws.translate_frame(df_module.example_dataframe, is_eager=True)
        .frame["int-col"]
        .to_numpy()
    )
    assert array.dtype == float
    assert_array_equal(array, np.asarray(example_data_dict["int-col"], dtype=float))

    array = (
        nws.translate_frame(df_module.example_dataframe, is_eager=True)
        .frame["str-col"]
        .to_numpy()
    )
    assert array.dtype == object
    assert_array_equal(array, np.asarray(example_data_dict["str-col"]))


def test_to_pandas(df_module, all_dataframe_modules):
    pd_module = all_dataframe_modules["pandas"]
    if df_module.name == "pandas":
        assert (
            nws.translate_frame(
                df_module.example_dataframe, is_eager=True
            ).frame.to_pandas()
            is df_module.example_dataframe
        )
        assert (
            nws.translate_series(df_module.example_column).series.to_pandas()
            is df_module.example_column
        )
    pd_module.assert_frame_equal(
        nws.translate_frame(df_module.example_dataframe, is_eager=True)
        .frame.to_pandas()
        .drop(["datetime-col", "date-col"], axis=1),
        pd_module.example_dataframe.drop(["datetime-col", "date-col"], axis=1),
    )
    pd_module.assert_column_equal(
        nws.translate_series(df_module.example_column).series.to_pandas(),
        pd_module.example_column,
    )


def test_make_dataframe_like(df_module, example_data_dict):
    df = ns.make_dataframe_like(df_module.empty_dataframe, example_data_dict)
    df_module.assert_frame_equal(df, df_module.make_dataframe(example_data_dict))
    assert ns.dataframe_module_name(df) == df_module.name


def test_make_column_like(df_module, example_data_dict):
    col = ns.make_column_like(
        df_module.empty_column, example_data_dict["float-col"], "mycol"
    )
    df_module.assert_column_equal(
        col, df_module.make_column(values=example_data_dict["float-col"], name="mycol")
    )
    assert ns.dataframe_module_name(col) == df_module.name


def test_all_null_like(df_module):
    col = ns.all_null_like(df_module.example_column)
    assert ns.is_column(col)
    assert ns.shape(col) == ns.shape(df_module.example_column)
    df_module.assert_column_equal(
        ns.is_null(col), df_module.make_column("float-col", [True] * ns.shape(col)[0])
    )


def test_concat_horizontal(df_module, example_data_dict):
    df1 = df_module.make_dataframe(example_data_dict)
    df1, plx = nws.translate_frame(df1)
    df2 = df1.rename({col: f"{col}1" for col in df1.columns})
    df = plx.concat([df1, df2], how="horizontal")
    assert df.columns == df1.columns + df2.columns


def test_to_column_list(df_module, example_data_dict):
    cols = nws.translate_frame(
        df_module.example_dataframe, is_eager=True
    ).frame.iter_columns()
    # cols = ns.to_column_list(df_module.example_dataframe)
    for c, name in zip(cols, example_data_dict.keys()):
        assert c.name == name


def test_collect(df_module):
    if df_module.name == "polars":
        df_module.assert_frame_equal(
            nws.translate_frame(df_module.example_dataframe.lazy(), is_lazy=True)
            .frame.collect()
            .to_native(),
            df_module.example_dataframe,
        )
    else:
        df_module.assert_frame_equal(
            nws.translate_frame(df_module.example_dataframe, is_lazy=True)
            .frame.collect()
            .to_native(),
            df_module.example_dataframe,
        )


#
# Querying and modifying metadata
# ===============================
#


def test_shape(df_module):
    assert nws.translate_frame(
        df_module.example_dataframe, is_eager=True
    ).frame.shape == (4, 6)
    assert nws.translate_frame(
        df_module.empty_dataframe, is_eager=True
    ).frame.shape == (0, 0)
    assert nws.translate_series(df_module.example_column).series.shape == (4,)
    assert nws.translate_series(df_module.empty_column).series.shape == (0,)


@pytest.mark.parametrize("name", ["", "a\nname"])
def test_name(df_module, name):
    assert (
        nws.translate_series(df_module.make_column(name=name, values=[0])).series.name
        == name
    )


def test_column_names(df_module, example_data_dict):
    col_names = nws.translate_frame(df_module.example_dataframe).frame.columns
    assert isinstance(col_names, list)
    assert col_names == list(example_data_dict.keys())
    assert nws.translate_frame(df_module.empty_dataframe).frame.columns == []


def test_rename(df_module):
    col = nws.translate_series(df_module.make_column(name="name", values=[0])).series
    col1 = col.rename("name 1")
    assert col.name == "name"
    assert col1.name == "name 1"


def test_set_column_names(df_module, example_data_dict):
    df = df_module.make_dataframe(example_data_dict)
    df = nws.translate_frame(df).frame
    old_names = df.columns
    new_df = df.rename({old: f"col {old}" for old in old_names})
    new_names = new_df.columns
    assert df.columns == old_names
    assert new_df.columns == new_names


#
# Inspecting dtypes and casting
# =============================
#


def test_dtype(df_module):
    df = ns.pandas_convert_dtypes(df_module.example_dataframe)
    assert ns.dtype(ns.col(df, "float-col")) == df_module.dtypes["float64"]
    assert ns.dtype(ns.col(df, "int-col")) == df_module.dtypes["int64"]


def test_cast(df_module):
    df, plx = nws.translate_frame(df_module.example_dataframe)
    result = df.select(int_col_cast_to_float=plx.col("int-col").cast(plx.Float64))
    assert result.schema == {"int_col_cast_to_float": plx.Float64}


def test_pandas_convert_dtypes(df_module):
    if df_module.name == "pandas":
        df_module.assert_frame_equal(
            ns.pandas_convert_dtypes(df_module.example_dataframe),
            df_module.example_dataframe.convert_dtypes(),
        )
        df_module.assert_column_equal(
            ns.pandas_convert_dtypes(df_module.example_column),
            df_module.example_column.convert_dtypes(),
        )
    else:
        assert (
            ns.pandas_convert_dtypes(df_module.example_dataframe)
            is df_module.example_dataframe
        )
        assert (
            ns.pandas_convert_dtypes(df_module.example_column)
            is df_module.example_column
        )


def test_is_bool(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    df, plx = nws.translate_frame(df)
    assert df.schema["bool-col"] == plx.Boolean
    assert df.schema["int-col"] != plx.Boolean


def test_is_numeric(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    df, plx = nws.translate_frame(df)
    for num_col in ["int-col", "float-col"]:
        assert df.schema[num_col].is_numeric()
    for col in ["str-col", "datetime-col", "date-col", "bool-col"]:
        assert not df.schema[col].is_numeric()


# todo: maybe should add a convert_dtypes function?
@pytest.mark.xfail()
def test_to_numeric(df_module):
    plx = nws.get_namespace(df_module.name)
    s = plx.Series("", list(range(5))).cast(plx.String)
    assert s.dtype == plx.String
    for dtype in [plx.Int64, plx.Float64]:
        try:
            as_num = s.cast(dtype)
        except:
            pass
        else:
            break
    assert as_num.dtype.is_numeric()
    assert as_num.dtype == plx.Int64
    s = s.to_native()
    df_module.assert_column_equal(
        as_num.to_native(),
        ns.pandas_convert_dtypes(df_module.make_column("_", list(range(5)))),
    )
    assert (
        ns.dtype(ns.to_numeric(s, dtype=df_module.dtypes["float32"]))
        == df_module.dtypes["float32"]
    )
    assert ns.dtype(ns.to_float32(s)) == df_module.dtypes["float32"]
    s = df_module.make_column("_", map("_{}".format, range(5)))
    with pytest.raises(ValueError):
        ns.to_numeric(s)
    df_module.assert_column_equal(
        ns.to_numeric(s, strict=False),
        ns.all_null_like(s, dtype=df_module.dtypes["int64"]),
    )
    assert (
        ns.dtype(ns.to_numeric(s, strict=False, dtype=df_module.dtypes["float32"]))
        == df_module.dtypes["float32"]
    )


def test_is_string(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    df, plx = nws.translate_frame(df)
    assert df.schema["str-col"] == plx.String
    for col in ["int-col", "float-col", "datetime-col", "date-col", "bool-col"]:
        assert df.schema[col] != plx.String


@pytest.mark.xfail()  # todo?
def test_to_string(df_module):
    plx = nws.get_namespace(df_module.name)
    s = plx.Series("", list(range(5))).cast(plx.String)
    assert s.dtype == plx.String


def test_is_object(df_module):
    if df_module.name == "polars":
        import polars as pl

        s = pl.Series("", [1, "abc"], dtype=pl.Object)
    else:
        s = df_module.make_column("", [1, "abc"])
    assert ns.is_object(ns.pandas_convert_dtypes(s))

    s = df_module.make_column("", ["1", "abc"])
    assert not ns.is_object(ns.pandas_convert_dtypes(s))


def test_is_anydate(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    date_cols = ["datetime-col"]
    if df_module.name != "pandas":
        # pandas does not have a Date type
        date_cols.append("date-col")
    for date_col in date_cols:
        assert ns.is_any_date(ns.col(df, date_col))
    for col in ["str-col", "int-col", "float-col", "bool-col"]:
        assert not ns.is_any_date(ns.col(df, col))


def test_to_datetime(df_module):
    s = df_module.make_column("", ["01/02/2020", "02/01/2021", "bad"])
    with pytest.raises(ValueError):
        ns.to_datetime(s, "%m/%d/%Y", True)
    df_module.assert_column_equal(
        ns.to_datetime(s, "%m/%d/%Y", False),
        df_module.make_column("", [datetime(2020, 1, 2), datetime(2021, 2, 1), None]),
    )
    df_module.assert_column_equal(
        ns.to_datetime(s, "%d/%m/%Y", False),
        df_module.make_column("", [datetime(2020, 2, 1), datetime(2021, 1, 2), None]),
    )
    s = df_module.make_column("", ["2020-01-02", "2021-04-05"])
    df_module.assert_column_equal(
        ns.to_datetime(s, None, True),
        df_module.make_column("", [datetime(2020, 1, 2), datetime(2021, 4, 5)]),
    )
    dt_col = ns.col(df_module.example_dataframe, "datetime-col")
    assert ns.to_datetime(dt_col, None) is dt_col
    if df_module.name != "pandas":
        return
    s = df_module.make_column("", ["2020-01-01 04:00:00+02:00"])
    dt = ns.to_datetime(s, None)
    assert str(dt[0]) == "2020-01-01 02:00:00+00:00"


def test_is_categorical(df_module):
    if df_module.name == "pandas":
        import pandas as pd

        s = pd.Series(list("aab"))
        assert not ns.is_categorical(s)
        s = pd.Series(list("aab"), dtype="category")
        assert ns.is_categorical(s)
    elif df_module.name == "polars":
        import polars as pl

        s = pl.Series(list("aab"))
        assert not ns.is_categorical(s)
        s = pl.Series(list("aab"), dtype=pl.Categorical)
        assert ns.is_categorical(s)
        s = pl.Series(list("aab"), dtype=pl.Enum("ab"))
        assert ns.is_categorical(s)


def test_to_categorical(df_module):
    s = df_module.make_column("", list("aab"))
    assert not ns.is_categorical(s)
    s = ns.to_categorical(s)
    assert ns.is_categorical(s)
    if df_module.name == "polars":
        import polars as pl

        assert s.dtype == pl.Categorical
        assert list(s.cat.get_categories()) == list("ab")
    if df_module.name == "pandas":
        import pandas as pd

        assert s.dtype == pd.CategoricalDtype(list("ab"))


#
# Inspecting, selecting and modifying values
# ==========================================
#


def test_is_in(df_module):
    s = df_module.make_column("", list("aabc") + ["", None])
    s = ns.pandas_convert_dtypes(s)
    df_module.assert_column_equal(
        ns.pandas_convert_dtypes(
            nws.translate_series(s).series.is_in(["a", "c"]).to_native()
        ),
        ns.pandas_convert_dtypes(
            ns.pandas_convert_dtypes(
                df_module.make_column("", [True, True, False, True, False, None])
            )
        ),
    )


def test_is_null(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    result = nws.translate_series(s).series.is_null()
    df_module.assert_column_equal(
        result.to_native(), df_module.make_column("", [False, True, False, True, False])
    )


def test_drop_nulls(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    df_module.assert_column_equal(
        nws.translate_series(s).series.drop_nulls().to_native(),
        ns.pandas_convert_dtypes(df_module.make_column("", [0, 2, 4])),
    )


def test_unique(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    s = nws.translate_series(s).series
    assert s.drop_nulls().n_unique() == 3
    df_module.assert_column_equal(
        s.drop_nulls().unique().to_native(),
        ns.pandas_convert_dtypes(df_module.make_column("", [0, 2, 4])),
    )


def test_where(df_module):
    plx = nws.get_namespace(df_module.name)
    s = ns.pandas_convert_dtypes(plx.Series("", [0, 1, 2]).to_native())
    out, pl = nws.translate_series(s)
    mask = nws.translate_series(plx.Series("", [True, False, True]).to_native()).series
    other = nws.translate_series(plx.Series("", [10, 11, 12]).to_native()).series
    out = out.zip_with(mask, other)
    df_module.assert_column_equal(
        out.to_native(), ns.pandas_convert_dtypes(df_module.make_column("", [0, 11, 2]))
    )


def test_sample(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, 1, 2]))
    sample = ns.sample(s, 2)
    assert ns.shape(sample)[0] == 2
    vals = set(ns.to_numpy(sample))
    assert len(vals) == 2
    assert vals.issubset([0, 1, 2])


@pytest.mark.skip()  # need to add replace?
def test_replace(df_module):
    plx = nws.get_namespace(df_module.name)
    s = ns.pandas_convert_dtypes(
        plx.Series("", "aa ab ac ba bb bc".split() + [None]).to_native()
    )
    s = nws.translate_series(s).series
    out = s.replace(s, "ac", "AC")
    expected = ns.pandas_convert_dtypes(
        plx.Series("", "aa ab ac ba bb bc".split() + [None]).to_native()
    )
    df_module.assert_column_equal(out.to_native(), expected.to_native())

    # out = ns.replace_regex(s, "^a", r"A_")
    # expected = ns.pandas_convert_dtypes(
    #     df_module.make_column("", "A_a A_b A_c ba bb bc".split() + [None])
    # )
    # df_module.assert_column_equal(out, expected)
