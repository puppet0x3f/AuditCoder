"""Minimal frozen compatibility layer for the selected official SciCode tests."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import scipy
import scipy.sparse
import sympy


def process_hdf5_list(group: h5py.Group) -> list[Any]:
    return [group[key][()] for key in group.keys()]


def process_hdf5_sparse_matrix(group: h5py.Group) -> scipy.sparse.spmatrix:
    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        return scipy.sparse.coo_matrix(
            (data, (group["row"][()], group["col"][()])), shape=shape
        )
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    if "blocksize" in group:
        return scipy.sparse.bsr_matrix(
            (data, indices, indptr),
            shape=shape,
            blocksize=tuple(group["blocksize"][()]),
        )
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_dict(group: h5py.Group) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            result[key] = process_hdf5_sparse_matrix(obj["sparse_matrix"])
        elif isinstance(obj[()], bytes):
            result[key] = obj[()].decode("utf-8", errors="strict")
        else:
            try:
                result[float(key)] = obj[()]
            except ValueError:
                result[key] = obj[()]
    return result


def process_hdf5_datagroup(group: h5py.Group) -> Any:
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        return process_hdf5_dict(group)
    raise ValueError("empty SciCode HDF5 data group")


def process_hdf5_to_tuple(
    step_id: str, test_num: int, h5py_file: str
) -> list[Any]:
    data = []
    with h5py.File(h5py_file, "r") as handle:
        for test_id in range(test_num):
            group_path = f"{step_id}/test{test_id + 1}"
            node = handle[group_path]
            if not isinstance(node, h5py.Group):
                raise FileNotFoundError(f"Path {group_path} is not a group")
            keys = list(node.keys())
            if len(keys) == 1:
                subgroup = node[keys[0]]
                if isinstance(subgroup, h5py.Dataset):
                    value = subgroup[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="strict")
                    data.append(value)
                else:
                    data.append(process_hdf5_datagroup(subgroup))
            else:
                values = []
                for key in keys:
                    subgroup = node[key]
                    if isinstance(subgroup, h5py.Dataset):
                        value = subgroup[()]
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="strict")
                        values.append(value)
                    else:
                        values.append(process_hdf5_datagroup(subgroup))
                data.append(tuple(values))
    return data


def process_symbol_in_dict(value: dict[Any, Any]) -> dict[Any, Any]:
    result = {}
    for key, item in value.items():
        result[key] = str(item) if isinstance(item, sympy.Symbol) else item
        if isinstance(key, sympy.Symbol):
            result[str(key)] = value[key]
            result.pop(key)
    return result


def are_dicts_close(
    dict1: dict[Any, Any],
    dict2: dict[Any, Any],
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> bool:
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, (sympy.Symbol, str)):
            if value1 != value2:
                return False
        elif isinstance(value1, scipy.sparse.spmatrix):
            if not np.allclose(value1.toarray(), value2.toarray(), atol=atol, rtol=rtol):
                return False
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if value1 != value2:
                    return False
    return True


def cmp_tuple_or_list(var1: Any, var2: Any) -> bool:
    if len(var1) != len(var2):
        return False
    for value1, value2 in zip(var1, var2):
        if isinstance(value1, dict):
            if not are_dicts_close(value1, value2):
                return False
        elif isinstance(value1, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            if not np.allclose(value1.toarray(), value2.toarray()):
                return False
        elif isinstance(value1, bool):
            if value1 != value2:
                return False
        else:
            try:
                if not np.allclose(value1, value2):
                    return False
            except ValueError:
                if value1 != value2:
                    return False
    return True
