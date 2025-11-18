# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import subprocess
from pbr.version import VersionInfo

## Version
_v = VersionInfo('neurapy_ai_utils').semantic_version()
__version__ = _v.release_string()
version = '.'.join(__version__.split('.')[:3])

# -- Project information -----------------------------------------------------
def get_git_short_hash():
    try:
        rc = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        rc = rc.decode("utf-8").strip()
        return rc
    except subprocess.CalledProcessError:
        return "unknown"
    
# -- Project information -----------------------------------------------------

project = "neurapy_ai_utils"
copyright = "2022, Neura Robotics GmbH"
author = "Neura Robotics GmbH"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'rst2pdf.pdfbuilder',
    'sphinxcontrib.confluencebuilder',
    'sphinx_rtd_theme',
    'sphinx_mdinclude'
]

templates_path = ['_templates']
exclude_patterns = [    
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "docker.in.rst",
    "getting_started.in.rst",
    "jupyter/*/*.ipynb",
    "python_api_in/*.rst",
    ]
html_sidebars = {
    '**': [
        'versioning.html',
    ],
}

# -- Options for LaTeX output --------------------------------------------

latex_logo = 'logo.jpg'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
# Make sure the target is unique
autosectionlabel_prefix_document = True

# Include methods that start with an _
napoleon_include_private_with_doc = False

# # Add Myst extensions
myst_enable_extensions = ["colon_fence", "html_image"]

# # Needed for font awesome support using the sphinx-design extension.
# html_css_files = [
#     "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
# ]

numpydoc_show_class_members = False

# For including README.md
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# For includin __init__()
autoclass_content = "both"

# The primary toctree document.
primary_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# html_css_files = [
#     'custom.css',
# ]
html_logo = 'logo.jpg'
html_show_sourcelink = False

##
html_favicon = "_static/neura_logo.svg"

# Display selection of all documentation versions.
html_context = {'display_all_docs_versions': True}

# added by Jaesik to hide "View page source"
html_show_sourcelink = False

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "NeuraPyAIUtilsdoc"

# current_hash = get_git_short_hash() # 9ddfd28
# version = "primary ({})".format(current_hash)
release = version

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# 'groupwise': __init__ first, then methods, then
#            properties. Within each, sorted alphabetically.
autodoc_member_order = "groupwise"

# Show TODO elements in the documentation
todo_include_todos = True

confluence_publish = True
confluence_space_key = 'SAppEng'
confluence_parent_page = 'Sphinx export test'
confluence_server_pass = "<token>"
#confluence_ask_password = True
#confluence_publish_dryrun = True

# (for Confluence Cloud)
confluence_server_url = 'https://neurarobotics.atlassian.net/wiki/'
confluence_server_user = 'user@neura-robotics.com'
