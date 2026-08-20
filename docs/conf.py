project = "earthsampler"
copyright = "2026, Shahine Bouabid"
author = "Shahine Bouabid"

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.napoleon",
]

# sphinx-autoapi: scan source statically (no import needed)
autoapi_dirs = ["../src/earthsampler"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]
autoapi_ignore = [
    "*/utils/*",
    "*/abstractemulator*",
]
autoapi_python_class_content = "class"
autoapi_member_order = "bysource"

# Napoleon: parse Google-style docstrings
napoleon_google_docstring = True

# MyST: enable Markdown pages
myst_enable_extensions = ["colon_fence"]

# Theme
html_theme = "furo"

# Suppress noisy warnings
suppress_warnings = ["autoapi.python_import_resolution"]

# Hide documented functions that have ":meta private:" in their docstring
def _skip_private_meta(app, what, name, obj, skip, options):
    if not skip and hasattr(obj, "docstring") and ":meta private:" in obj.docstring:
        return True
    return skip

def setup(app):
    app.connect("autoapi-skip-member", _skip_private_meta)
