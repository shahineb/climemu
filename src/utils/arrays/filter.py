def filter_var(var_name):
    def keep_only_var(ds):
        if var_name in ds.data_vars:
            return ds[[var_name]]
        else:
            return None
    return keep_only_var