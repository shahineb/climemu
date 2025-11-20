import xarray as xr

def filter_datatree(tree: xr.DataTree, keep: list[str]) -> xr.DataTree:
    new = xr.DataTree(name=tree.name)
    for key in tree.groups:
        grp = key.strip("/")
        if grp in keep:
            new[grp] = tree[grp]
    return new